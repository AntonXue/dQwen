#!/bin/bash
# C.8 trajectory lane, 24-way shape: ONE (model, scheme) per GPU
# (HANDOFF 20260818-140400 + Anton's 24-GPU ruling). Each lane writes its
# own shard file (trajectories_<model>[_nostop]__<scheme>.jsonl); shards
# concatenate into the canonical hand-back files at drain. Block schemes
# also run the --no-early-stop variant sequentially; standard schemes
# have no nostop variant (no early exit to disable).
#
#   sbatch sbatch_c8.sh llada-8b-base block16-tau0.9   # from a login node
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
MODEL=$1; SCHEME=$2
echo "== c8 $MODEL / $SCHEME early-stop ($(date +%H:%M:%S))"
python trajectories.py --model "$MODEL" --gen 256 --problems all \
    --schemes "$SCHEME" --out $OUT
case "$SCHEME" in block16*)
  echo "== c8 $MODEL / $SCHEME no-early-stop ($(date +%H:%M:%S))"
  python trajectories.py --model "$MODEL" --gen 256 --problems all \
      --schemes "$SCHEME" --out $OUT --no-early-stop
  ;;
esac
echo "== c8 $MODEL / $SCHEME done ($(date +%H:%M:%S))"
