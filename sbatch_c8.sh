#!/bin/bash
# C.8 trajectory campaign: SIX jobs, one per decode scheme, 4 nodes each
# (rank = model). 2h walls -- worst rank ~90 min (block16-s16 llada, both
# stop variants); per-shard resume makes any clip a free requeue.
#
#   for s in block16-static-s16 block16-static-s8 block16-tau0.9 \
#            standard-static-s256 standard-static-s128 standard-tau0.9; do
#     sbatch sbatch_c8.sh $s
#   done
#SBATCH -J c8-traj
#SBATCH -p gh
#SBATCH -A ASC25023
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
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
SCHEME=$1

srun --ntasks-per-node=1 -n "$SLURM_NNODES" bash -c '
  MODELS=(llada-8b-base dream-7b-base dream-coder-7b-base dqwen3.5-9b-base-v3)
  MODEL=${MODELS[$SLURM_PROCID]}
  SCHEME="'"$SCHEME"'"
  OUT=_runs/20260818-140400-c8-trajectories-he164-g256
  echo "== rank $SLURM_PROCID: $MODEL / $SCHEME early-stop ($(date +%H:%M:%S))"
  python trajectories.py --model "$MODEL" --gen 256 --problems all \
      --schemes "$SCHEME" --out $OUT \
    || echo "== rank $SLURM_PROCID FAILED early-stop (continuing)"
  case "$SCHEME" in block16*)
    echo "== rank $SLURM_PROCID: $MODEL / $SCHEME no-early-stop ($(date +%H:%M:%S))"
    python trajectories.py --model "$MODEL" --gen 256 --problems all \
        --schemes "$SCHEME" --out $OUT --no-early-stop \
      || echo "== rank $SLURM_PROCID FAILED nostop (continuing)"
    ;;
  esac
  echo "== rank $SLURM_PROCID done ($(date +%H:%M:%S))"
'
