#!/bin/bash
# One job = one TILE PLAN job: N nodes, each node runs its lane's cell list
# sequentially (lane = plan_tiles.py's <=~3.2h packing; the 8h wall is
# belt-and-suspenders over that estimate). Submit via launch.sh from a
# LOGIN node -- sbatch is login-only on Vista.
#
# Idempotency keeps requeues free: finished cells exit in seconds, so an
# overrun lane killed at the wall loses only its in-flight cell. Mop-up:
# `run.py --pending`, then just resubmit the same jobs -- completed cells
# fast-skip.
#SBATCH -J dqeval-tile
#SBATCH -p gh
#SBATCH -A ASC25023
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00
#SBATCH --output=_slurm_out/%x_%j.out
set -o pipefail           # no -e: one failed cell must not kill the lane
source ~/.bashrc          # rc files + conda reference unset vars, so
conda activate qwen35     # nounset only AFTER env setup (smoke-caught)
set -u

export SCRATCH=${SCRATCH:-/scratch/11079/antonxue}
export HF_HOME=${HF_HOME:-$SCRATCH/cache/huggingface}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
[ -n "${LOCAL_NVME:-}" ] && export TRITON_CACHE_DIR="$LOCAL_NVME/triton_cache"

cd "$(dirname "$0")"
mkdir -p _slurm_out
TILES=$1; JOB=$2

srun --ntasks-per-node=1 -n "$SLURM_NNODES" bash -c '
  MANIFEST=$(python -c "import json,sys; print(json.load(open(\"'"$TILES"'\"))[\"manifest\"])")
  LANE=$(python -c "import json; print(\" \".join(map(str, json.load(open(\"'"$TILES"'\"))[\"jobs\"]['"$JOB"'][$SLURM_PROCID])))")
  for i in $LANE; do
    echo "== lane $SLURM_PROCID cell $i ($(date +%H:%M:%S))"
    python run.py "$MANIFEST" "$i" || echo "== cell $i FAILED (continuing)"
  done
  echo "== lane $SLURM_PROCID done ($(date +%H:%M:%S))"
'
