#!/bin/bash
# Submit every job of one or more tile plans. Run from a LOGIN node
# (sbatch is login-only on Vista):
#
#   bash launch.sh manifests/part1-ar-baselines.tiles.json
#
# Node counts come from each job's lane count. Resubmitting a plan is
# always safe: completed cells exit in seconds.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p _slurm_out
for t in "$@"; do
  n=$(python3 -c "import json; print(len(json.load(open('$t'))['jobs']))")
  for ((j = 0; j < n; j++)); do
    lanes=$(python3 -c "import json; print(len(json.load(open('$t'))['jobs'][$j]))")
    sbatch -N "$lanes" sbatch_tiles.sh "$t" "$j"
  done
done
squeue -u "$USER" -o '%.10i %.12j %.5D %.10M %.10L %.8T'
