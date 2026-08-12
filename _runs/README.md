# _runs — raw evaluation artifacts, and nothing else

Contract (2026-08-12): everything in here is **raw** — the direct output of
evaluating models. Aggregation is read-only and lives outside: `python
summarize.py [filter]` prints one line per cell; no script writes derived
data back here.

Two kinds of entries:

- **Stores** (accumulate across launches; deterministic per-cell names):
  `grid_v1/` and `regrades/`.
- **Launch dirs** (`<yyyymmdd-hhmmss>-<description>/`, stamp = launch time,
  retroactive stamps taken from directory birth times): everything a single
  launch produced or logged. Frozen once the launch is over.

## Stores

| dir | what | naming |
|---|---|---|
| `grid_v1/` | generation cells from `run.py` (the canonical store) | `<benchmark>/<model>@<revision>__<benchmark>__<decode>__s<K>of<N>.jsonl` |
| `regrades/` | CPU regrades of saved generations (EvalPlus HE+) | `<yyyymmdd-hhmmss>__<model>@<revision>__<benchmark>-plus__from-<source>.jsonl` (+ `_eval_results.json`) |

`grid_v1` cell filenames are deliberately **not** timestamped: the filename
is the cell's identity, which is what makes reruns idempotent and lets
shards land from different nodes at different times into one tree. The
launch time lives *inside* each cell — `launched_at` in the `meta` record
on line 1 — and `summarize.py` prints it.

## Launch dirs

| dir | what |
|---|---|
| `20260805-160006-battery-first-family-battery/` | first full family battery; source of 0.8B v3@50k generations (repo `main` = the 50k final at the time) |
| `20260805-203611-battery4b-step-pinned-battery/` | step-pinned battery (`step50000-swa`); source of 2B/4B/9B v3@50k generations + control |
| `20260807-032607-coda-head-to-head/` | CoDA port, same-backbone head-to-head |
| `20260809-040349-ar-twin-SUPERSEDED-wrong-base-repos/` | **SUPERSEDED — never cite.** Loaded Qwen's pretrained `-Base` repos, not our bare conversion sources. Kept as the historical record; the audit script skips it. Replacement: the `__ar__` cells in `grid_v1/`. |
| `20260809-040642-stepsweep-steps-per-block/` | steps-per-block sweeps |
| `20260811-231456-tau-smoke-humaneval/` | τ-threshold smoke (2B/9B); picked the adaptive grid {0.5–0.9} |
| `20260812-022548-ar-baselines-bare-qwen/` | lane script + console logs of the overnight AR baselines; **the data is in `grid_v1/*/Qwen_*__ar__s0of1.jsonl`** |
