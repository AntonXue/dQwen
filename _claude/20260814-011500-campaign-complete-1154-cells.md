# CAMPAIGN COMPLETE: 1,154/1,154 GATE cells on cluster hardware, ~21h wall

> 2026-08-14 ~01:05, when the last mop-up cell (the bus-error CoDA
> math500 stripe, of all cells) renamed into place. Launched 2026-08-13
> ~04:00 with part 1. Every paper number now has a cluster-GATE stamp.

## The run, by the numbers

- **14 jobs total**: 13 tiled campaign jobs (1 / 6 / 6 across the three
  parts, all <=16 nodes, 8h walls) + one 5-node mop-up. Never touched
  the QOS caps (20 running / 40 submitted).
- **1,154 cells = 72 AR + 272 big-table + 810 acceleration.** All
  complete, all ending in their summary sentinel.
- **Exactly ONE cell failure all campaign**: a transient node fault (Bus
  error during a datasets Arrow map) on CoDA math500 s0of2. Recovered by
  the mop-up. The `|| continue` lane semantics and idempotent SKIP did
  their jobs; nothing was ever lost.
- **Exactly ONE wall-clip**: job 908635 TIMEOUT at 8h with 5 CoDA cells
  unfinished (see below). All 12 other campaign jobs completed inside
  their walls — several with hours to spare (part-1's 9-node job: 3h09m
  vs 3.2h estimated worst lane; ~3% cost-model error).
- Throughput followed the designed S-curve: ~1.3 cells/min overnight
  (front-loaded heavy lanes), ~3.4 cells/min in the cheap-tail flood.
- Raw store: `_runs/grid_v1/`, **381M**, on $HOME (backed up, not
  subject to $SCRATCH purge).

## The one estimation miss worth remembering: CoDA canvas-fills

The cost model priced CoDA at 2B-like rates. Its cells ran ~3x long —
but NOT because it is slow: measured 0.035 s/forward (normal for its
size). **CoDA emits no usable stop under base-style prompting, so every
document runs the full canvas: n_forward mean = p90 = max = 1024, vs
mean 61 for dqwen3.5-2B on the identical MBPP shard (17x).** Consistent
with its published protocol (fixed 768-token canvas, one token/step, no
early exit). Two consequences: (1) model-contiguous lane packing
concentrated all CoDA cells into one job, which is why exactly one job
hit its wall; (2) CoDA raw generations carry post-answer junk that the
stop-string truncation removes at grading — protocol-appendix material.
The mop-up (manifests/plan_mopup.py: one cell per lane from live
`--pending`, 8h wall) cleared all 5 in ~2.5h each.

## Verification state

Early read-only `summarize.py --merge` over 1,149 cells: **every shard
group a complete disjoint union except the two CoDA groups** whose 5
cells were still pending — i.e., zero structural surprises. Those cells
have since landed; the final full-merge readout was generating as this
doc was written (Lustre crawled right after the queue drained) and gets
recorded in the next entry with the big-table numbers.

## Part-1 validation recap (full table: 20260813-071500)

MMLU exact vs workstation (+-0.2); GSM8K noise-level except bare
Qwen3.5-2B +5.4pp — the _ARLM canvas fix landing on the documented
verbosity pathological; math500 re-measurements above full-MATH refs as
expected. Parked curiosity: Qwen3.5-4B drops 16pp on mbpp-FENCE vs
scaffold (extraction/formatting suspect; per-sample read pending).

## Handoff to the workstation (rendering/plotting)

`_runs` is untracked by design; data moves by rsync, code by git. On the
WORKSTATION (archive the PROBE record first — same filenames!):

    cd ~/foo/dQwen/_runs
    mv grid_v1 20260814-probe-era-grid-archive
    rsync -avz --info=progress2 \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/grid_v1/ \
      grid_v1/
    find grid_v1 -name '*.jsonl' | wc -l   # must print 1154

After that, workstation summarize.py + the paper scripts read pure GATE
data with zero path changes.

## Open threads

- Final merged big-table + frontier numbers -> next _claude entry.
- 4B mbpp-fence anomaly: per-sample read.
- Decontamination scan: still MANDATORY-not-started on the paper side.
- HF cache (1.8T) lives on $SCRATCH and is purge-exposed; disposable
  (re-prewarm rebuilds it).
