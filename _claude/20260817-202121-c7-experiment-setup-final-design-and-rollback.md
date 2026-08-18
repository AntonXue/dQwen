# C.7 decode-trajectory experiment: FINAL design, receipts, and the rollback map

> 2026-08-17 ~20:21, workstation. Supersedes the pool in `20260817-191325`
> (the frozen 13) after two Anton rulings this session: the ×2-speed
> configs joined the all-pass filter, and the model roster dropped the
> control. Anton is away for a few hours; this doc is written BEFORE
> implementation so the whole change is auditable and revertible; runs
> launch immediately after.

## 1. The design (all rulings final, Anton this session)

**Models (4, all DLMs, one per GPU):** dqwen3.5-9b-base-v3@step50000-swa,
llada-8b-base, dream-7b-base, dream-coder-7b-base. The control was
DROPPED (cleaner roster; cost recorded: forfeits the only place the paper
could SHOW hybrid-vs-full-attention decode-order differences — one cheap
row re-run away if ever wanted). No AR row: pure AR is definitionally the
diagonal; it appears as a gray reference line, never a measurement.

**Schemes (2×3 factorial, canvas 128, greedy, τ=0.9 fixed):**

| | 1 tok/fwd | 2 tok/fwd (×2) | adaptive |
|---|---|---|---|
| standard | standard-static-s128 | standard-static-s64 | standard-tau0.9 |
| block 16 | block16-static-s16 | block16-static-s8 | block16-tau0.9 |

block32 dropped entirely (Anton; at canvas 128 it has 4 staircase treads
vs block16's 8 — the demo optimizes legibility, protocol fidelity is not
required here). Deviations from the measured protocol = exactly two
(canvas 128, block 16), each owed a caption line. τ fixed at 0.9 (the
paper's featured threshold; the ×2 static column covers matched-budget,
so the earlier measure-then-pick procedure is dead). Stop-strings stay ON
(post-stop canvas is informative and marked, not hidden).

**Problems (6 = the ENTIRE surviving pool; no discretionary pick):**
/0 has_close_elements, /3 below_zero, /8 sum_product, /31 is_prime,
/50 decode_shift, /56 correct_bracketing. Funnel with receipts:
164 → 13 (all-pass over 24 store cells: 4 models × 6 canvas-1024
analogues {block32-s32, block32-s16, block32-tau0.9, standard-s1024,
standard-s512, standard-tau0.9} — the ×2-speed analogues are the new,
binding additions) → 6 (the 30–100-token legibility band; the 7 others
are sub-30-token one-liners). Notable casualty vs the old pool: /55 fib
— Dream fails it under both ×2-commit configs. Since the figure shows
ALL 6, there is no selection step left to attack. Caveat: receipts are
canvas-1024/block-32 proxies; block16@128 truth comes from the runs,
`passed` recorded per panel.

**Volume:** 4 × 6 × 6 = 144 generations, ≤128 forwards each.

**Execution order (Anton):** problem-major — every model does /0 (all 6
schemes), then /3, … Achieved by one process per model per GPU, all
iterating the pool in the same order; at most one job per GPU.

**Figure shape (rendered manuscript-side later):** rows = models (4),
cols = schemes (6, block trio then standard trio pending screening),
~1.0in panels → two problems per page → the whole pool in 3 pages.
Per-panel: gray AR diagonal, faint block-boundary rules (block cols),
teal commit dots, pale post-stop dots, corner n_forward + pass mark.

## 2. File-touch list (= the rollback map; ONE commit, `git revert` undoes it)

| file | change |
|---|---|
| `models/samplers.py` | +~8 lines: `GenOutput.commit_step` (len gen_length, −1 = never committed) + stamping at the single `take`-commit point. Additive; no behavior change. |
| `trajectories.py` | NEW root driver (~200 lines): loads models via `models.load`, prompts/stops via `benchmark_specs.task_config("humaneval")` + lm-eval (does NOT import `run` — gate rule 3), runs the 6 schemes × pool, grades `passed` via code_eval, appends JSONL incrementally, renders screening PNGs (matplotlib, present in qwen35). |
| `tests/structure_gate.py` | +1 file in the scan list (`trajectories.py`), so the new root file is gate-covered. |
| `models/coda.py` | comment fix from the manuscript request §8: the tied-head hazard is NOT "harmless under 4.x" — measured firing on transformers 4.57.1. |

Output (not in git): `_runs/20260817-202121-c7-decode-trajectories/`
(launch-dir contract): `trajectories.jsonl` + `screen_<problem>.png` +
per-GPU logs. JSONL per request `20260817-151600` §2, plus `raw_text`,
`stop_cut_chars`, and the scheme's DecodeConfig.

## 3. Correctness gates (asserted in-driver BEFORE the fleet launches)

- standard-static-s128: commit_step is a PERMUTATION of 0..127.
- standard-static-s64: exactly 2 stamps per step, 64 distinct steps.
- block16 statics: block-monotone across the 8 treads (every stamp in
  block b < every stamp in block b+1); 1 resp. 2 stamps per step.
- τ schemes: ≥1 stamp per forward (the progress floor), total forwards
  strictly below the static twin (else τ never fired).
- Stop-cut consistency: −1 exactly on positions at/after the cut in
  fully-uncommitted trailing blocks.

## 4. Deferred / open

- Stage 2 (leftmost-fraction statistic over HE-164): still optional per
  the manuscript request §5; NOT in this launch. Trivial to add later
  (same driver, `--problems all --schemes standard-*`).
- Paper figure rendering: manuscript repo's job, from the JSONL.
- The 20260817-191325 pool doc stands as history; THIS doc's pool is the
  operative one.
