# Layout contract: `_runs` is raw-only, one stamped dir per launch, coalescing outside

> 2026-08-12, Anton's rulings after the HE+ regrade batch: (1) run outputs
> must be recoverable from names alone; (2) launches carry `yyyymmdd-hhmmss`
> stamps, applied retroactively from directory birth times; (3) `_runs`
> holds raw data only — aggregation is a separate read-only script; (4) no
> second top-level tree — a briefly-created `_launches/` split was reverted
> and everything repaired IN `_runs`.

## The contract (also in `_runs/README.md`; operator guide in the repo README)

`_runs/` contains exactly two kinds of entries:

- **Stores** — accumulate across launches, deterministic per-cell names:
  - `grid_v1/`: `<bench>/<model>@<rev>__<bench>__<decode>__sKofN.jsonl`.
    NO timestamp in the filename ON PURPOSE: identity = filename is what
    makes reruns idempotent and lets shards land from many nodes into one
    tree. Launch time is `launched_at` inside the `meta` record (new field).
  - `regrades/`: `<yyyymmdd-hhmmss>__<model>@<rev>__<bench>-plus__from-<source>.jsonl`
    (+ `_eval_results.json`); `evalplus_driver.py --from` self-names these.
- **Launch dirs** — `<yyyymmdd-hhmmss>-<description>/`: everything one
  launch produced or logged; frozen after.

Coalescing: `python summarize.py [filter]` — one printed line per cell
(grid + regrades), read-only, never writes into `_runs`.

## The rename map (2026-08-12; old name → new)

| old | new |
|---|---|
| `battery/` | `20260805-160006-battery-first-family-battery/` |
| `battery4b/` | `20260805-203611-battery4b-step-pinned-battery/` |
| `coda/` | `20260807-032607-coda-head-to-head/` |
| `ar_twin/` | `20260809-040349-ar-twin-SUPERSEDED-wrong-base-repos/` |
| `stepsweep/` | `20260809-040642-stepsweep-steps-per-block/` |
| `smoke_tau/` | `20260811-231456-tau-smoke-humaneval/` |
| `ar_overnight/` | `20260812-022548-ar-baselines-bare-qwen/` |
| `heplus/` | `regrades/` (ten cryptic pairs → full identities under batch stamp `20260812-091609`; map + gate table in `regrades/README.md`) |

`ar_overnight` held ONLY the lane script + console logs — its data always
went to `grid_v1/*/Qwen_*__ar__s0of1.jsonl`. That was the source of the
"why two dirs?" confusion.

## References fixed with the renames

- manuscript `scripts/audit_eval_provenance.py`: `SUPERSEDED_DIRS` now
  names the stamped ar-twin dir. Re-run after the moves: **PASSES**.
- manuscript `scripts/budget_trade_grid.py`: battery dir names.
- `tests/step_sweep.py`: default output path.
- `dqeval/grid.py`: meta records now carry `launched_at` (older cells
  lack it; `summarize.py` prints `-`).
- `RUNBOOK.md`: also fixed a wrong claim (MBPP+ is NOT a regrade — edited
  sanitized prompts, needs its own generations). [RUNBOOK merged into
  README.md later the same day; the traps live there now.]
