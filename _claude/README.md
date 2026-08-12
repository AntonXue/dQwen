# `_claude/` — dQwen working log & progression

**Project goal.** A **release-quality evaluation package** for the dQwen3.5 / dQwen3
family and its diffusion-LM comparators (LLaDA, Dream / Dream-Coder, SDAR), all under
**one environment**, with lm-eval-harness on top. Two things it must support:

1. Load any model and call **its own shipped sampler** — "the number with LLaDA's own
   sampler".
2. Run **our sampler** on the same weights — "the number with ours". Model x sampler
   is a cross-product, not one number per model.

Plus reproducing the numbers those papers claim, so our comparison tables are
defensible rather than cited.

**Why this dir.** The finer-grained running timeline — what was run, the numbers, the
decisions and their rationale, the negatives — at low ceremony, so a fresh Claude (or
future-me) on a different machine can reconstruct the *progression* without git
archaeology or chat scrollback. Mirrors `~/foo/ADLMC/_claude/`.

## Conventions
- **Filename:** `yyyymmdd-hhmmss-short-description.md`. One topic per file.
- **Append-only:** don't rewrite old entries; add a new file that supersedes and
  reference the old one by name. The directory IS the timeline.
- **Write for the next reader:** 1–2 lines of context, then findings. Tables > prose
  for numbers. Always say *what was run*, *where artifacts live* (paths / HF repo ids),
  and *what to do next*.
- **Record refuted priors explicitly.** Several load-bearing design choices here exist
  because a measurement contradicted an earlier belief; without the refutation written
  down, the next reader re-proposes the wrong thing.
- Cold-start reading order: this dir in filename order → `third_party/LOCKFILE.md`
  (which upstream implementation produced which published table) → the auto-memory
  index at `~/.claude/projects/-home-ayx98-foo-ADLMC/memory/MEMORY.md`.

## Sister projects
- `~/foo/ADLMC/` — the research repo this is extracted from. Training, ablations, and
  the eval prehistory (`ablations/paper_evals/`, `eval/`). Its `_claude/` holds the
  dQwen3.5 family training history and the eval-standardization decisions.

## Index (update when adding files)

| file | what it holds |
|---|---|
| `20260721-184634-dqeval-bootstrap-port-evidence-adapters-samplers.md` | ⭐⭐ **Repo bootstrap + the port evidence.** Our models need tf 5.13, comparators pin 4.4x–4.5x, and the sets are **disjoint** — so LLaDA/Dream/SDAR were **ported** with ~90 lines of shims (3/4/3, zero invented). **Decomposed the drift: 100% torch, 0% transformers ⇒ the port is BITWISE-exact and the acceptance gate is bitwise, not a tolerance** (an earlier tolerance recommendation came from a confounded comparison — do not reintroduce). **Silent failure class** hit 2/3 families: tf5 meta-device wipes non-persistent rotary `inv_freq`, tf5's repair lives in base `_init_weights` which Dream and SDAR both override ⇒ model loads, forwards, returns noise, raises nothing (argmax 20–40%) — hence mandatory goldens + `assert_finite_rope`. **Two logit surfaces** (`raw_logits`/`logits`): Dream alone is AR-aligned and shifts in its own sampler AND eval wrapper, so canonicalising for its native sampler would double-shift silently. **Two priors refuted:** Dream's sampler is NOT HF-coupled (plain class, 6/7 bit-identical), and SDAR DOES ship a sampler (`JetAstra/SDAR/generate.py`, runnable on HF weights via `store_kv`). **SDAR minefield:** four disagreeing implementations, thresholds 0.85/0.75/**0.9**, published table came from **LMDeploy**, and their own script can't do greedy ⇒ our SDAR numbers are re-measurements. **Built + GPU-verified:** all four families load through one adapter contract (incl. dQwen3.5 at `revision=step30000-swa`), and the unified bs=1 sampler runs on all four — **token-for-token identical to LLaDA's own `generate.py`** at matched config, which validates "native decode is a preset, not a code path". |
| `20260805-190500-v1-v3-mixture-ab-mbpp-format-effect-and-footgun-guards.md` | ⭐⭐ **0.8B v1-vs-v3 mixture A/B (budget-matched, both 50k) + the MBPP prompt-format effect.** v3 wins every benchmark and **code ROSE ~4.4pp while code share FELL 71%→50%** — code was bought with targeting (`code_sft` 12%, verified concepts), not volume; the campaign's assumed "sacrifice code for general gains" trade may not exist. v1 arm **reproduces July on all four benchmarks (two bitwise)**, which is what makes the deltas quotable. **MBPP's `[BEGIN]`/`[DONE]` scaffold is 0.000% of both training corpora** ⇒ no scaffold-leakage story is available, and swapping it for a ```python fence is worth **+4.00pp (v1) / +6.00pp (v3)** — v1-fenced ≈ v3-scaffolded, i.e. **prompt format ≈ the entire mixture overhaul**. Mechanism is the *task→fenced-solution pattern* (0%→17% of tokens), NOT fences (v1 already had 42.9%). ⚠ fenced MBPP is NOT comparable to published LLaDA/Dream numbers. **Priors refuted:** cross-process nondeterminism does NOT perturb these evals (pinning `CUBLAS_WORKSPACE_CONFIG` changed nothing — the real cause was `--num-fewshot` silently falling back to a 0-shot task default); no Hub drift; v1 does not ramble on fences. **Harness hardened:** append now needs `--allow-append` (fails pre-load), few-shot warning per task, `decode`+`model_args` provenance in every result json, config banner at log head, and a **`.gitignore` bug** that left the whole agent worktree untracked-but-visible. |
| `20260807-035500-coda-port-five-shims-and-same-backbone-head-to-head.md` | ⭐⭐ **CoDA ported (5th family) + the first SAME-BACKBONE head-to-head.** `Salesforce/CoDA-v0-Base` is adapted from Qwen3-1.7B — same backbone as our `dQwen3-1.7B-Base` — so it is the only comparator isolating RECIPE rather than starting weights. Unified protocol: **ours 42.68 HE vs CoDA 23.17 (+19.51)**, and that is with our superseded v1-era mixture. ⚠ their published 29.3 uses 768 tokens at ONE token/step (768 forwards) vs our ≤512 with up to 32 committed/step — our protocol undersells them ~6pp, so run the native-ish cell before publishing. ⚠ CoDA **untied its head** (verified from the raw checkpoint: max|Δ| 4.41, cos 0.48) ⇒ **2.032B**, not 1.7B. Their arXiv table is mislabelled (README says "Humaneval Instruct" for base rows) and their MBPP+ > MBPP because "MBPP+" is EvalPlus's 378-problem *sanitized* subset, NOT a superset ⇒ **their MBPP is not comparable to ours; HumanEval is the only clean target**. **Five shims, all restorations, three NEW beyond `ADLMC/coda/old/environment.md`** (which was verified on 4.57; they break only on 5.x): `_supports_sdpa`, `rope_scaling→None`, **`tie_word_embeddings→False` (silent head-destruction if unrestored)**, `all_tied_weights_keys={}`, post-load `inv_freq` recompute. CoDA is the SECOND AR-aligned family (same verbatim shift as Dream) — docstring corrected. Lesson: a 3× slowdown I blamed on decode was **GPU contention** — check `nvidia-smi` first. |
| `20260812-094500-runs-layout-contract-raw-vs-launches.md` | ⭐ **Layout contract (Anton's rulings): `_runs` = raw only; every launch dir stamped `yyyymmdd-hhmmss-<desc>` (retroactive stamps from dir birth times); coalescing = read-only `summarize.py`; NO second top-level tree** (a brief `_launches/` split was reverted — repair happens IN `_runs`). grid_v1 cell filenames stay deterministic ON PURPOSE (identity=filename ⇒ idempotent reruns + multi-node shards); launch time = `launched_at` in the meta record. Full old→new rename map inside (battery, battery4b, coda, ar_twin, stepsweep, smoke_tau, ar_overnight, heplus→regrades). ar_overnight held only logs — data was always grid_v1. Auditor + budget grid re-verified after the moves. |
| `20260812-023000-grid-deploy-gates-passed.md` | ⭐⭐ **grid_v1 runner DEPLOY-READY** (run.py + RUNBOOK.md are the front door): spot-check 10/10 (6 decode families × 6 benchmarks × ours/comparator/AR × 3 revisions × 6 stripe shapes); Gate A shard merge LOSSLESS (164/164, 0 overlap) w/ HE reproduction +1.83pp attributed to cross-era numerics (same-env kernel probe: 0 flips ⇒ deterministic; expect ~±2pp on PROBE→GATE restamps, paired same-env comparisons exact); Gate B GSM8K template fidelity 163/165 per-doc, stripe aggregate identical. τ grid {0.5–0.9}; smoke frontiers: block Pareto-dominates standard (~4× forwards at ≤ accuracy). Merge-tool notes: 2 sample records/doc on gsm8k (strict = per-doc min); global id = k + doc_id·n. NEXT: make_manifest row lists, then launch. |
