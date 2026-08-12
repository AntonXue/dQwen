#!/bin/bash
# AR-counterpart overnight lane. Usage: lane.sh GPU MODEL1 [MODEL2 ...]
# Benchmark-major order: if morning comes early, complete COLUMNS exist for
# every model rather than complete models with missing columns.
cd /home/ayx98/foo/dQwen
PY=/ssd1/ayx98/miniconda3/envs/qwen35/bin/python
GPU=$1; shift
for bench in gsm8k mmlu humaneval mbpp math; do
  for model in "$@"; do
    tag="${model//\//_}_${bench}"
    echo "[$(date +%H:%M)] START $model $bench"
    CUDA_VISIBLE_DEVICES=$GPU $PY run.py "$model" - ar "$bench" \
      > "_runs/ar_overnight/${tag}.log" 2>&1 \
      && echo "[$(date +%H:%M)] DONE  $model $bench" \
      || echo "[$(date +%H:%M)] FAIL  $model $bench (see ${tag}.log)"
  done
done
echo "[$(date +%H:%M)] === lane on GPU $GPU complete ==="
