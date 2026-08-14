# DELEGATION (cluster): CoDA produces noise on GH200 — diagnose, fix, rerun 19 cells

> 2026-08-14, workstation. Every CoDA cell in the campaign is garbage:
> mmlu 24.15 (= chance on 4-way MC), 0.00 on all eight generation
> benchmarks. Both the MC path and the generation path are broken, so the
> fault is at FORWARD level (logits are noise), not in decoding. This doc
> carries known-good goldens captured on the workstation (x86, torch
> 2.7.1, tf 5.13 — where CoDA works) so every hypothesis is a diff, not a
> guess.

## What we can already rule OUT (deduction from the cells existing at all)

run_cell's build gates RAISE on failure, and all 19 CoDA cells completed.
Therefore, ON THE CLUSTER:
- `assert_finite_rope` passed → inv_freq is FINITE (but may be WRONG —
  finite-but-wrong passes the gate; see prime suspect below).
- The UNTIED-HEAD GATE passed → lm_head is not storage-tied and not
  bitwise-equal to embeddings.
- The mask_token_id check passed.

## Workstation goldens (diff against these, same commands)

Probe script — run this verbatim on a GH200 GPU (PYTHONPATH=repo root):

    import torch
    from models import load
    from models.samplers import DecodeConfig, generate
    a = load("coda-1.7b-base")
    print("shims:", a.shims)
    sums = [float(b.sum()) for n, b in a.model.named_buffers() if n.endswith("inv_freq")]
    print("inv_freq:", len(sums), f"{sums[0]:.6f}")
    h = a.model.lm_head.weight
    print(f"lm_head sum={float(h.float().sum()):.4f} std={float(h.float().std()):.6f}")
    for prompt in ("def add(a, b):\n    ", "The capital of France is"):
        ids = a.encode(prompt)
        raw = a.raw_logits(ids)
        print(prompt[:12], f"mean={float(raw.float().mean()):.4f}",
              f"std={float(raw.float().std()):.4f}",
              "argmax[-5:]:", raw[0].argmax(-1)[-5:].tolist())
        out = generate(a, ids, DecodeConfig(gen_length=32, block_length=32, steps_per_block=32))
        print("  gen:", out.text[:70])

Golden values (workstation, KNOWN GOOD):

    shims = 5 entries, verbatim:
      CoDAModel._supports_sdpa = True (restores pre-4.54 dispatch)
      rope_scaling ['rope_theta', 'rope_type'] -> None (tf5 synthesised; checkpoint ships null)
      tie_word_embeddings True -> False (CoDA is untied; tf5 default, absent from config.json)
      all_tied_weights_keys={} on CoDALanguageModel (post_init never called in 4.x either)
      post-load CoDARotaryEmbedding.inv_freq recompute (meta-buffer repair)
    inv_freq: 1 buffer, sum = 5.150445        <- EXACT, hardware-independent
    lm_head:  sum = -10066.6621  std = 0.188063   <- EXACT, pure checkpoint data
    'def add' prompt:  logits mean=-4.6345 std=8.0565  argmax[-5:] = [11, 279, 11, 11, 279]
    'capital' prompt:  logits mean=-4.5937 std=7.9022  argmax[-5:] = [11, 279, 279, 279, 279]
    gens are REPETITIVE-BUT-LEXICAL (base-model babble, real tokens):
      ' bdef add(a, b):\n     bdefdef add(a, b):...'
      '\nThe capital capital of France is the capital of France...'
    macro reference: workstation battery cell generated CORRECT code
      ('    return [string[:i] for i in range(len(string) + 1)]') — noise
      on the cluster vs coherent code here is the discriminator.

## Diagnosis ladder (stop at the first mismatch)

0. `git pull`; confirm tf==5.13.0 and record torch/aarch64 versions.
1. `python tests/load_matrix.py --only coda-1.7b-base` — purpose-built
   silent-failure asserts (finite+sane logits, canonicalization must be
   EXACTLY cat([l[:,:1], l[:,:-1]]), greedy never proposes MASK).
2. Run the probe above. Decision tree:
   - **shims list != 5 entries / different text** → a shim's detection
     predicate misfired under this tf/torch build; fix models/coda.py
     apply_shims for the actual class/config state you find.
   - **inv_freq sum != 5.150445** → PRIME SUSPECT CONFIRMED: the
     post-load repair recomputed finite-but-wrong values. The repair
     calls CoDA's OWN `default_rope_frequencies` — check its dtype/device
     behavior under torch 2.10/aarch64 (suspect: computed in bf16 or on
     meta; recompute in float32 on CPU, then `.to(device)`, then
     re-register). This failure mode passes assert_finite_rope by
     construction, which is exactly why the row died silently.
   - **lm_head sum/std off** → the $SCRATCH snapshot is suspect (that
     cache had known layout weirdness: Dream repos in a legacy layout,
     LLaDA absent entirely). `rm` the CoDA snapshot and re-run
     `run.py --prewarm manifests/part2-big-table.jsonl`, re-probe.
   - **weights+rope exact but logits stats way off** (std collapsed or
     exploded) → forward-path: CoDA hardcodes is_causal=False through
     SDPA — A/B torch's sdpa backends (math already pinned in cells;
     probe both), A/B bf16 vs fp32, and check the (logits, loss) tuple
     branch still takes the eval-mode path under this build.
   - **everything matches, gens still noise under run.py only** →
     decode-side (unlikely: chance-level mmlu says forward), diff a
     RecordingLM call against the probe.
3. Record findings + the fix in a _claude doc (this file's stamp as the
   parent), commit.

## Rerun protocol (after the probe matches goldens)

1. On the cluster store: DELETE the 19 garbage CoDA cells first —
   idempotency will otherwise SKIP them:
       rm _runs/grid_v1/*/coda-1.7b-base@main__*.jsonl
2. `python run.py --pending manifests/part2-big-table.jsonl` → exactly 19
   indices (18 sharded generation cells + 1 mc-nelbo mmlu).
3. Launch via slurm_launch.sh as usual; re-sync to the workstation when
   green (rsync into _runs/grid_v1_vista/, same exclude rules).

## Acceptance

- load_matrix --only coda-1.7b-base: PASS.
- Probe: inv_freq and lm_head EXACT (they are pure data); logit stats
  within ~1%; argmax lists identical or off by <=1 borderline position;
  generations lexical, not noise.
- Rerun sanity: mmlu clearly above chance (expect 30s-40s), HumanEval in
  the vicinity of the PROBE-era 23.17 head-to-head (canvas is 1024 now;
  small movement fine), nothing at exactly 0.00.
