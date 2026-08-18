#!/bin/bash
# C.8 trajectory lane: one model, both stop variants sequentially
# (HANDOFF 20260818-140400). NOT a grid campaign -- own driver, own
# output dir, no store writes.
#
#   sbatch sbatch_c8.sh llada-8b-base        # from a login node; x4 models
#SBATCH -J c8-traj
#SBATCH -p gh
#SBATCH -A ASC25023
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH --output=_slurm_out/%x_%j.out
set -o pipefail
source ~/.bashrc
conda activate qwen35
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export SCRATCH=${SCRATCH:-/scratch/11079/antonxue}
export HF_HOME=${HF_HOME:-$SCRATCH/cache/huggingface}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p _slurm_out
OUT=_runs/20260818-140400-c8-trajectories-he164-g256
MODEL=$1
echo "== c8 $MODEL early-stop ($(date +%H:%M:%S))"
python trajectories.py --model "$MODEL" --gen 256 --problems all --out $OUT
echo "== c8 $MODEL no-early-stop ($(date +%H:%M:%S))"
python trajectories.py --model "$MODEL" --gen 256 --problems all --out $OUT --no-early-stop
echo "== c8 $MODEL done ($(date +%H:%M:%S))"
