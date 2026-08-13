# QOS caps kill the array interface; campaign launches TILED (plan_tiles + launch.sh)

> 2026-08-13 ~03:45, cluster session, pre-greenlight. Two Vista facts
> surfaced by Anton + sacctmgr, then the launch mechanics were rebuilt
> around them. Nothing submitted yet.

## The two constraints

1. **qgh QOS: MaxJobsPU=20, MaxSubmitPU=40 — and SLURM counts every array
   TASK against both** (sacctmgr-verified). The one-cell-per-array-task
   interface cannot queue even part 1 (72 tasks). sbatch_cells.sh stays
   for single-cell debugging only.
2. **sbatch is LOGIN-NODE-ONLY on Vista** (Anton). Compute/idev nodes have
   network for prewarm but cannot submit. $HOME is one filesystem, so the
   login node sees the checkout directly — no pull step.

## The design: tiled lanes (chosen over 2 alternatives)

`manifests/plan_tiles.py` packs manifest cells into node-lanes targeting
**~3.2h** from a measured cost model (GH200 probe anchors: sec/fwd by
family, fwd/doc by benchmark, static-sK ~ K/32 of s32, tau <= 0.7x s32,
ceiling = exactly 1024 fwd/doc, mc-nelbo mmlu ~ 1 fwd/doc; AR rates are
coarse and get recalibrated from part 1's wall_s meta fields). Lanes are
model-contiguous (loads amortize), 16 lanes per job, **8h wall** (Anton:
"to be safe" — 2.5x the lane target). `launch.sh <tiles.json>` submits
every job of a plan from the login node.

Rejected: 32 independent striding workers (worse job-count ergonomics,
no load grouping); one big 24h multi-node job (waits for simultaneous
allocation on a busy queue, unbounded idle-tail SU burn).

## The packing (est totals; conservative by construction)

| plan | cells | lanes | jobs | est node-h |
|---|--:|--:|--:|--:|
| part1-ar-baselines | 72 | 9 | 1 (-N 9) | 25 |
| part2-big-table | 272 | 92 | 6 (5x16+12) | 240 |
| part3-acceleration | 810 | 92 | 6 (5x16+12) | 277 |

13 jobs, all <=16 nodes, worst lane est 3.3h vs 8h wall. Overrun/failure
economics unchanged: finished cells persist, `launch.sh` resubmission
fast-skips, `--pending` audits.

## Launch flow (part 1 first, then 2+3 after its merge validates)

    ssh <login>; cd ~/foo/dQwen
    bash launch.sh manifests/part1-ar-baselines.tiles.json

Then `summarize.py --merge`, diff vs the 08-12 workstation AR batch
(expect: MATH/MBPP AR cells ABOVE the old 256-cap numbers — the _ARLM
canvas fix — others within cross-hardware noise), recalibrate AR rates in
plan_tiles.py from measured wall_s, and launch parts 2+3.
