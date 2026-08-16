#!/bin/bash
# The one SLURM file: submits tile plans, and IS the job script it submits.
#
#   bash slurm_launch.sh manifests/part1-ar-baselines.tiles.json [more...]
#       submit mode (LOGIN node -- sbatch is login-only): one sbatch per
#       job in each plan, -N from the job's lane count.
#
#   bash slurm_launch.sh <plan.tiles.json> <job#>
#       payload mode (what sbatch runs; also runnable directly inside an
#       idev allocation for rehearsal): node r runs lane r sequentially.
#
# Plans come from manifests/plan_tiles.py (~3.2h lanes, 16/job; the 8h
# wall is 2.5x that). Resubmitting is always safe: completed cells exit
# in seconds (summary sentinel), so a lane killed at the wall resumes
# where it died. `run.py --pending manifest.jsonl` audits leftovers.
# Single-cell debugging is just `python run.py MODEL REV DECODE BENCH K/N`.
#SBATCH -J dqeval
#SBATCH -p gh
#SBATCH -A ASC25023
#SBATCH --ntasks-per-node=1
#SBATCH -t 16:00:00
#SBATCH --output=_slurm_out/%x_%j.out
set -o pipefail   # -u only after rc files: they reference unset vars
                  # (smoke-caught; and no -e in payload -- one failed cell
                  # must not kill its lane)

# sbatch runs a SPOOLED COPY of this script, so $0 points into
# /var/spool -- the repo is $SLURM_SUBMIT_DIR there (smoke-caught #2).
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$SELF")}"

if ! [[ "${2:-}" =~ ^[0-9]+$ ]]; then
  # ---- submit mode ----
  set -eu
  mkdir -p _slurm_out
  for t in "$@"; do
    n=$(python3 -c "import json; print(len(json.load(open('$t'))['jobs']))")
    for ((j = 0; j < n; j++)); do
      lanes=$(python3 -c "import json; print(len(json.load(open('$t'))['jobs'][$j]))")
      sbatch -N "$lanes" "$SELF" "$t" "$j"
    done
  done
  squeue -u "${USER:-$(whoami)}" -o '%.10i %.12j %.5D %.10M %.10L %.8T'
  exit 0
fi

# ---- payload mode ----
source ~/.bashrc
conda activate qwen35
set -u
export SCRATCH=${SCRATCH:-/scratch/11079/antonxue}
export HF_HOME=${HF_HOME:-$SCRATCH/cache/huggingface}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
[ -n "${LOCAL_NVME:-}" ] && export TRITON_CACHE_DIR="$LOCAL_NVME/triton_cache"
mkdir -p _slurm_out
TILES=$1; JOB=$2

srun --ntasks-per-node=1 -n "$SLURM_NNODES" bash -c '
  MANIFEST=$(python -c "import json,sys; print(json.load(open(\"'"$TILES"'\"))[\"manifest\"])")
  LANE=$(python -c "import json; print(\" \".join(map(str, json.load(open(\"'"$TILES"'\"))[\"jobs\"]['"$JOB"'][$SLURM_PROCID])))")
  for i in $LANE; do
    echo "== lane $SLURM_PROCID cell $i ($(date +%H:%M:%S))"
    # STORM-WAIT: scratch flaps kill cells in their import window and can
    # outlast a single retry (measured: the 08-16 05:56 outage burned 700+
    # cells through pre-hardened lanes; a later flap beat retry-once).
    # Probe the exact failure surface (site-packages + the stdlib module
    # that broke) and STALL until healthy -- walls absorb the wait.
    until timeout 30 python -c "import numpy, multiprocessing.popen_fork" >/dev/null 2>&1; do
      echo "== scratch storm: stalling 180s ($(date +%H:%M:%S))"; sleep 180
    done
    python run.py "$MANIFEST" "$i" \
      || { echo "== cell $i failed once, retrying in 90s (scratch weather)"; sleep 90; \
           python run.py "$MANIFEST" "$i" || echo "== cell $i FAILED (continuing)"; }
  done
  echo "== lane $SLURM_PROCID done ($(date +%H:%M:%S))"
'
