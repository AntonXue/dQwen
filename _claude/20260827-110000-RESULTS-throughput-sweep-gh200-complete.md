# RESULTS: GH200 throughput sweep COMPLETE — GDN layers cost throughput ONLY single-stream/short-context; 1.3×→2.2× ahead everywhere else, growing with L

> 2026-08-27, Vista (cluster Claude). Answers the boss ask in HANDOFF
> 20260826-235257; supersedes its H100-prototype numbers. Two exclusive GH200
> nodes: c608-072 (idev 943292) + c639-072 (945525), clock snapshots in
> `node_provenance.txt`. Anton's protocol rulings folded in live: uniform
> **5 warmup + 50 timed** per cell in the release battery ("v2"); re-running
> everything was cheaper than defending mixed iteration counts.

## Datasets (all under `_runs/`, JSONL one line per cell, arms distinguished
## by per-record (`sdpa_arm`, `causal_conv1d`); later records carry `node`)

| dir | what |
|---|---|
| `20260827-throughput-gh200-i50/` | **THE RELEASE DATASET.** 9 models × arms × modes, uniform i50, plus envelope extensions (L16k/32k, crossover refinement, batch B64/128). 42 stages, 7h20m, zero failures. Mostly c639. |
| `20260827-throughput-gh200/` | v1 quick pass (i20; 7–9B at i10), 8 models complete (9B partial was deleted when superseded). Mostly c608. KEPT as the iters-sensitivity + inter-node cross-check. Console logs + `phase34.sh` + provenance live here. |
| `…-stability-{a,b,c}/` | Tier-1 deployment slice repeated at session start/middle/end on c639. |
| `…-fa2/` | flash-attn-2 spot check — **inapplicable**, see §6. |

## 1. The answer (Tier-1: dqwen3.5-2B hybrid vs width-matched dqwen3-1.7B
## full-attention control, deployment kernels, tok/s ratio, i50)

| B, L | full | trunk (vocab-head confound removed) |
|---|---|---|
| 1, 1024 | 0.67 | 0.66 |
| 1, 4096 | 1.19 | 1.25 |
| 16, 1024 | 1.16 | 1.25 |
| 16, 4096 | 1.31 | 1.44 |
| 16, 8192 | 1.48 | 1.63 |
| 4, 16384 | 1.78 | 1.97 |
| 4, 32768 | **2.21** | **2.42** |

- **Launch-bound regime** (B=1, L≲3.5k): hybrid at 0.66–0.85× — the honest
  cost of GDN's extra kernel launches. All four kernel arms agree within ~6%
  there: no tuning rescues it, none is needed to explain it.
- **Crossover measured, not interpolated**: B=1 ratio walks 0.77 (L3072) →
  0.85 (L3584) → 1.19 (L4096) — a sharp transition, not a smooth glide; in
  batch, B=2 at L2048 already crosses (1.10). Any batching OR L≥4k ⇒ hybrid
  ahead.
- **Compute-bound**: advantage grows monotonically with L, 2.2×/2.4× at 32k —
  the linear-vs-quadratic attention story on real deployment kernels.
- **Batch saturation** (B∈{32,64,128}, L≤1024): both plateau — hybrid ~120k
  tok/s, control ~105k (+13–15% serving-throughput ceiling).

## 2. Kernel-robustness (the "unfair setup" defense)

2B full-mode tok/s across arms (i50): deploy ≥ every other arm at every
point. Decomposition @B16/L4096: FLA+conv1d fast path ≈ 1.26× over fallback
(114.7k vs 90.9k), flash-vs-math sdpa ≈ 1.69× (114.7k vs 68.0k). And the
control NEEDS fast attention more than the hybrid does: under both-naive
kernels the Tier-1 ratio GROWS to 2.7–3.3× (control B16/L4096 collapses to
17.6k tok/s; most hybrid layers aren't attention). The hybrid's win is not a
kernel artifact; fast kernels flatter the control.

## 3. Context rows (deployment, full mode, tok/s, i50)

| model | B16/L4096 | B1/L1024 |
|---|---|---|
| dqwen3.5-0.8b | 195,458 | 28,013 |
| dqwen3.5-2b | 114,739 | 27,260 |
| dqwen3-1.7b control | 87,878 | 40,738 |
| coda-1.7b | 85,054 | 39,442 |
| dqwen3.5-4b | 52,125 | 19,382 |
| dream-7b / dream-coder-7b | 34,292 / 34,284 | 34,134 / 34,181 |
| dqwen3.5-9b | 33,103 | 19,617 |
| llada-8b | 28,383 | 30,234 |

9B long-context: 23.7k tok/s at B1/L32768, 32 GB — the 96 GB envelope is
nowhere near binding for single-stream long context.

## 4. Memory (the prototype's "much flatter memory" did NOT reproduce)

Full-mode peak memory is vocab-head-dominated (248k vs 151k: the logits
tensor alone is ~65 GB at B32/L4096 for the hybrid vs ~40 GB control) and
even trunk-mode hybrid peaks ~35% higher (GDN conv/gating intermediates).
Both grow linearly in tokens (flash SDPA: no L² materialization). Report the
head GEMM separately (`head` mode cells) — that's what it exists for.
OOM envelope: deployment OOMs only at grid corners (B32/L8192 for ≤4B;
9B additionally B32/L4096, B16/L8192, B4/L32768). Datum: the NAIVE arm OOMs
at B16/L4096 on llada and dream-coder — unfused attention's transient alone
kills 7–8B full-attention models at shapes the hybrids handle trivially.

## 5. Measurement hygiene (what a reviewer will ask)

- **Identity check**: full ≈ trunk + head within 1% on all 7 trunk-capable
  models (trunk unsupported on LLaDA AND CoDA — remote code takes no embeds
  kwarg; CoDA was not predicted by the campaign handoff, recorded honestly).
- **Stability**: batched cells reproduce to 0.3–0.8% across five
  measurements spanning both nodes and 7 h (B16/L4096: 570.1 vs 571.2 ms
  cross-node = 0.2%). B=1 cells jitter 28–43% — launch-bound latency is
  inherently noisy; quote the regime, not a third digit.
- **Iters sensitivity**: v1 (i10/20) vs v2 (i50) median |Δ| 0.3–1.3% over
  ~780 shared cells (maxes 9–18% are all launch-bound corners). The
  iteration count never mattered; 50 is uniform for presentation.
- **Math-arm integrity**: throughput.py now re-pins math sdpa AFTER model
  load (LLaDA's remote `__init__` re-enables flash sdp,
  `modeling_llada.py:1056` — the run.py trap). Verified: every math-arm
  record shows `flash_sdp: false`. Patch + `node` provenance field are the
  only driver changes; both uncommitted at write time.

## 6. flash-attn-2 spot check: INAPPLICABLE (stronger than "excluded")

flash_attn 2.8.3 is installed, but our DLM checkpoints load through
remote-code config classes that `AutoModelForCausalLM` cannot map (recorded
in `…-fa2/fa2_delta.jsonl`), and the frozen adapter does not plumb
`attn_implementation`. The flash-attn package path does not exist for these
models — torch-SDPA-flash IS the best available attention for every row, so
the §6 exclusion needs no "marginal" defense.

## 7. Hand-back + remaining

1. rsync (additive) `_runs/20260827-throughput-gh200*` to the workstation +
   md5s, per the campaign handoff's conventions.
2. Commit: throughput.py (repin + node field), this doc + index row, the
   run-dir scripts/consoles. NOT yet done — awaiting Anton's go (his call
   whether raw JSONLs ride git like the H100 prototype dir did or rsync
   only).
3. Open (cheap, unblocked): nothing. The grid + extensions are complete.
