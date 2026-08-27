# HANDOFF → next cluster Claude: throughput sweep mid-battery, idev node being killed

> 2026-08-27 ~02:58, written on c608-072 (idev 943292) minutes before Anton
> kills the node. The campaign is `_claude/20260826-235257-HANDOFF-…` (read it
> first); this doc is the delta: what already ran, exactly how to resume, and
> the Tier-1 results so far. Everything below lives on shared filesystems and
> survives the node; the running chain + its console die with it.

## 0. State at write time (02:58) — most of the battery is DONE

Out dir: `_runs/20260827-throughput-gh200/` (JSONLs + console snapshots +
`phase34.sh`, the chain script). Stage order = `phase34.sh` top to bottom.

| model | deployment | fla+math | fallback+flash | naive |
|---|---|---|---|---|
| dqwen3.5-2b (Tier 1) | ✅ | ✅ | ✅ | ✅ |
| dqwen3-1.7b control (Tier 1) | ✅ | n/a | n/a | ✅ |
| dqwen3.5-0.8b | ✅ | ✅ | ✅ | ✅ |
| coda-1.7b | ✅ | n/a | n/a | ✅ |
| dqwen3.5-4b | ✅ | ✅ | ✅ | ✅ |
| dream-7b | ✅ | n/a | n/a | ✅ |
| dream-coder-7b | ✅ | n/a | n/a | ✅ |
| llada-8b | ⏳ IN FLIGHT at kill | n/a | n/a | ❌ |
| dqwen3.5-9b | ❌ | ❌ | ❌ | ❌ |

(n/a = pure-attention model, GDN arms don't exist; per the campaign handoff
they get deployment + naive only.) Arms within one JSONL are distinguished by
the per-record provenance pair **(`sdpa_arm`, `causal_conv1d`)**:
fast/True=deployment, math/True=fla+math, fast/False=fallback+flash,
math/False=naive.

## 1. ⚠ THE ONE TRAP: throughput.py APPENDS, it never skips

Unlike run.py cells there is no completeness sentinel. Re-running a finished
(model, arm) duplicates lines. Resume protocol:

1. The stages ran strictly sequentially, so **at most ONE arm is partial: the
   in-flight one at kill time = the (`sdpa_arm`,`causal_conv1d`) signature of
   the LAST LINE of the newest-mtime `throughput_*.jsonl`** (cross-check
   `phase34_console_snapshot.log`, but that snapshot may be stale — trust the
   JSONL rule). Likely `llada-8b-base` deployment (fast/True), possibly later.
2. Trim that arm before rerunning it, e.g.:
   ```python
   import json, pathlib
   p = pathlib.Path("_runs/20260827-throughput-gh200/throughput_llada-8b-base.jsonl")
   keep = [l for l in p.read_text().splitlines()
           if not ((r:=json.loads(l))["sdpa_arm"]=="fast" and r["causal_conv1d"]==True)]
   p.write_text("\n".join(keep) + ("\n" if keep else ""))
   ```
3. Delete the COMPLETED stages from a copy of `phase34.sh` and run the rest.
   Remaining if llada-deployment was in flight: llada deployment (rerun),
   llada naive, 9b deployment, 9b fla+math, 9b fallback+flash, 9b naive —
   all at `--iters 10` (the 7–9B rule; iters is recorded per record so mixed
   counts are self-documenting). ~45–70 min total.

## 2. Environment (all persistent, nothing to rebuild)

- **tp-venv** (deployment/fla arms): `$WORK/tp-venv` — venv over the qwen35
  conda env (`--system-site-packages`), plus `causal-conv1d==1.7.0` built
  from source (module `cuda/12.6` nvcc, sm_90, torch cu129 minor mismatch =
  warning-class). **Hard gate PASSED on c608-072**: `fla: True,
  causal_conv1d: True`, 2B forward, zero "fast path is not available".
  Re-gate on the new node (~2 min, model cached):
  `CUDA_VISIBLE_DEVICES=0 $WORK/tp-venv/bin/python -c "import torch,models;
  m=models.load('dqwen3.5-2b-base-v3',revision='step50000-swa');
  print(m.raw_logits(torch.randint(0,1000,(1,64),device=m.device)).shape)"`
  → must NOT print the fast-path warning.
- **Stock qwen35 env** = the fallback arm as-is (`fla` present,
  `causal_conv1d` absent): `/scratch/11079/antonxue/miniconda3/envs/qwen35`.
- **throughput.py carries an UNCOMMITTED local patch**: math-arm re-pin AFTER
  `models.load()` (LLaDA's remote `__init__` re-enables flash sdp,
  `modeling_llada.py:1056` — same trap run.py pins for). Verified live: all
  math-arm records show `flash_sdp: false`. KEEP IT; commit with the results.
  llada-naive is the arm that actually exercises it — spot-check its records.
- New node: **append** a fresh stanza to `node_provenance.txt` (job id,
  `nvidia-smi -q -d CLOCK`). Cells will now come from two GH200 nodes — same
  hardware class, note both job ids in the results doc.

## 3. Tier-1 results (the boss answer — already safe, reproduce from JSONLs)

Hybrid 2B / control 1.7B tok/s ratio (>1 = hybrid faster), deployment arm,
full mode: **0.67× @ B1/L1024 (launch-bound worst case) → crossover ~B1/L4096
(1.09×) or any batching → 1.31× @ B16/L4096 → 1.42–1.48× @ L8192** (trunk
mode: 0.71× → 1.43× → 1.64×). GH200 reproduces the H100 prototype's shape.
Kernel-robustness: under BOTH-naive kernels the hybrid's win GROWS
(2.74×/3.34× at L4096) — unfused attention murders the control while most
hybrid layers aren't attention. Deployment arm dominates every 2B cell (no
kernel combo beats it) and at B1/L1024 all four arms agree within ~6%
(launch-bound = kernel-insensitive). Kernel decomposition @B16/L4096 trunk:
FLA+conv1d ≈ 1.31× on the hybrid, flash-vs-math sdpa ≈ 1.75×.
⚠ One prototype claim does NOT reproduce: "much flatter memory" — hybrid
peak mem is HIGHER (248k head dominates full mode: ~65 GB logits tensor at
B32/L4096; trunk mode hybrid ≈ +35% from GDN intermediates). Both linear in
tokens; report as measured, head GEMM is quotable separately.
OOM envelope so far: only B32×L8192 corners (2B, control, others per JSONL).

## 4. After the battery: remaining deliverables (campaign handoff §5)

1. **full ≈ embed+trunk+head spot-check per model — NOT DONE YET** (harness
   sanity check; do it from the JSONLs, no GPU needed).
2. Results doc: Tier-1 2×2 at slice points, deployment crossover per model,
   OOM boundary, both nodes' provenance, the iters-10 note, the memory
   finding, the re-pin patch note.
3. Commit: throughput.py patch + `_runs/20260827-throughput-gh200/` docs +
   this dir's README index row (added at write time).
4. rsync hand-back per the campaign handoff (additive + md5s). Anton has NOT
   yet ruled on the optional flash-attn-package spot-check slice (installed
   but unused; §6 exclusion) — ask before adding it.

## 5. Console snapshots (stale copies, node /tmp originals are gone)

`tier1_deployment_console.log`, `tier1_crossarms_console.log`,
`phase34_console_snapshot.log` in the out dir — stage stamps + per-cell
lines through ~02:57. The 0.8B→dream-coder stages all ended with clean
stage-boundary stamps and zero `!!! FAILED` markers.
