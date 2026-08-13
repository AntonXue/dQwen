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
    bash slurm_launch.sh manifests/part1-ar-baselines.tiles.json

Then `summarize.py --merge`, diff vs the 08-12 workstation AR batch
(expect: MATH/MBPP AR cells ABOVE the old 256-cap numbers — the _ARLM
canvas fix — others within cross-hardware noise), recalibrate AR rates in
plan_tiles.py from measured wall_s, and launch parts 2+3.

## UPDATE: dress rehearsal caught TWO launch-killers; SLURM surface = 1 file

Anton ("super super make sure with a small test") was right to be
nervous. Executing the literal job-script body on the idev node (a
SLURM allocation, so `srun` is real) found:

1. **`set -u` before `source ~/.bashrc`** — the rc references
   $SHELL_STARTUP_DEBUG unset, so nounset killed every job at line one.
   Both sbatch scripts had it (the original kit was never sbatch-run).
   Fix: strict mode only AFTER rc + conda activate.
2. **`cd "$(dirname "$0")"` in a batch script** — sbatch runs a SPOOLED
   COPY (/var/spool/...), so $0 leaves the repo. Fix: prefer
   $SLURM_SUBMIT_DIR (and this very idev session demonstrates the trap:
   its SLURM_SUBMIT_DIR is ~/foo/ADLMC, wherever idev was typed).

Then the consolidation (Anton): launch.sh + sbatch_tiles.sh +
sbatch_cells.sh -> **one dual-mode `slurm_launch.sh`** (submits plans;
IS the submitted job script; payload mode dispatches on `<tiles> <int>`
args). sbatch_cells.sh DELETED — its array premise is dead under the
caps and single-cell debugging is a bare `run.py` call. Rehearsed both
modes: submit (stubbed sbatch, correct -N/args/abs-path) and payload
(live 3-cell mini-campaign end-to-end offline + 24s idempotent re-sweep,
verified before and after the merge).
