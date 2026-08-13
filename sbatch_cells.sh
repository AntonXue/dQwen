#!/bin/bash
# One SLURM array task = one cell. Vista (GH200, 1 GPU/node).
#
#   python run.py --pending manifest.jsonl        # -> e.g. 0,1,2,5,9
#   sbatch --array=0,1,2,5,9%32 sbatch_cells.sh manifest.jsonl
#
# Requeues are free: completed cells exit in seconds (summary sentinel),
# and --pending regenerates the remaining index list at any time.
# Prereqs, once, on a LOGIN node (network): bash setup_env.sh   (harness
# layer into the existing qwen35 env) and python run.py --prewarm
# manifest.jsonl (models + datasets + code_eval into $HF_HOME).
#SBATCH -J dqeval
#SBATCH -p gh
#SBATCH -A ASC25023
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH --output=_slurm_out/%x_%A_%a.out
set -eo pipefail
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
python run.py "$1" "$SLURM_ARRAY_TASK_ID"
