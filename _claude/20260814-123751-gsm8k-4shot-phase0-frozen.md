# GSM8K 4-shot Phase 0 executed: protocol frozen, 8-shot cells archived

> 2026-08-14, workstation. Executes §6 Phase 0 of
> `20260814-121922-PLAN-gsm8k-4shot-rerun-and-the-metric-audit-behind-it.md`.
> Phases 1–3 (validation gate, 84 headline cells, 540 acceleration cells)
> are the cluster's; everything below is what it inherits on pull.

## The change

| file | edit |
|---|---|
| `run.py` | `BENCH["gsm8k"].shots` 8 → 4 — the AUTHORITATIVE value (run.py sets `num_fewshot` on the task from BENCH; the spec's own field is informational) |
| `benchmark_specs.py` | GSM8K `num_fewshot` 8 → 4 (kept in sync), section comment updated, task `metadata.version` 3.0 → 4.0 (4.0 = the 4-shot protocol) |
| `tests/task_freeze_gate.py` | `SHOTS["gsm8k_cot"]` 8 → 4 |
| `summarize.py` | merge_groups gains a protocol guard: stripes whose `meta.bench.shots` disagree print `MIXED-PROTOCOL … refusing to merge` instead of combining |
| `_runs/README.md` | stores table documents the archive + the empty gsm8k column |

No manifest changes: manifests don't encode shot counts, so the existing
part1/2/3 gsm8k lines ARE the rerun set once the old outputs are cleared
(re-sharding 4B/9B to 12 is the cluster's call per the plan).

## The archive

`_runs/grid_v1_vista/gsm8k/` (all 632 cells: 594 accel + 30 headline-only +
8 AR) moved to `_runs/grid_v1_vista_gsm8k_8shot/gsm8k/`. Same layout, so
`summarize.py --store grid_v1_vista_gsm8k_8shot [--merge]` reads it — this
is the frozen 8-shot side of the free protocol-sensitivity result. The
cluster must do the equivalent move on its `grid_v1/` BEFORE requeueing
(deterministic filenames + the completeness sentinel would otherwise
fast-skip every cell).

## Verified

- `tests/task_freeze_gate.py` GATE PASSED — vendored gsm8k_cot at 4-shot
  renders byte-identical to pinned stock lm-eval at 4-shot.
- Functional: `run.py --list` prints `shots=4`; a constructed 4-shot
  fewshot_context contains exactly 4 "The answer is" exemplars (first 4 of
  the stock 8, per lm-eval `first_n`).
- `--store grid_v1_vista --merge`: zero gsm8k lines, zero INCOMPLETE, zero
  MIXED-PROTOCOL. `--store grid_v1_vista_gsm8k_8shot --merge`: the 8-shot
  groups merge cleanly (LLaDA 72.48, Dream 75.89, Dream-Coder 72.18, …).
- Old cells are discriminable forever: every meta records
  `bench.shots` (verified 8 in the archived cells, 4 going forward).

## Cluster pickup (unchanged from the PLAN, restated)

1. Pull. Archive cluster-side `_runs/grid_v1/gsm8k/` the same way.
2. Phase 1 FIRST: `Qwen/Qwen3-1.7B-Base` and `Qwen/Qwen3-0.6B-Base` AR
   gsm8k anchors vs published 75.44 / 59.59 (>3pp = STOP), plus the 8
   bare-repo AR cells at 4-shot. These `-Base` models are calibration
   anchors, NEVER AR counterparts.
3. Phase 2 (84 headline cells) unblocks the manuscript; Phase 3 (540) can
   trail.
