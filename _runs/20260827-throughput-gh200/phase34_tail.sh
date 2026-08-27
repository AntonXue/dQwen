#!/bin/bash
# Resume tail after node restart (c639-072): llada naive + all 9B arms.
cd /home1/11079/antonxue/foo/dQwen
OUT=_runs/20260827-throughput-gh200
SLICE="--batches 1,16 --lengths 1024,4096"
VENV=$WORK/tp-venv/bin/python
STOCK=/scratch/11079/antonxue/miniconda3/envs/qwen35/bin/python
run() { echo "=== $1 $(date -Is) ==="; shift; CUDA_VISIBLE_DEVICES=0 "$@" || echo "!!! FAILED: $*"; }
run "llada naive"         $STOCK throughput.py --model llada-8b-base --out $OUT --sdpa math --iters 10 $SLICE
run "9b deployment"       $VENV  throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa fast --iters 10
run "9b fla+math"         $VENV  throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa math --iters 10 $SLICE
run "9b fallback+flash"   $STOCK throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa fast --iters 10 $SLICE
run "9b naive"            $STOCK throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa math --iters 10 $SLICE
echo "=== tail done $(date -Is) ==="
