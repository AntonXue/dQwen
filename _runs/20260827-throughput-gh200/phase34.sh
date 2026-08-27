#!/bin/bash
# Phase 3+4: remaining 7 models, deployment + slices, cheapest first.
cd /home1/11079/antonxue/foo/dQwen
OUT=_runs/20260827-throughput-gh200
SLICE="--batches 1,16 --lengths 1024,4096"
VENV=$WORK/tp-venv/bin/python
STOCK=/scratch/11079/antonxue/miniconda3/envs/qwen35/bin/python
run() { echo "=== $1 $(date -Is) ==="; shift; CUDA_VISIBLE_DEVICES=0 "$@" || echo "!!! FAILED: $*"; }

# --- 0.8B hybrid: deployment + all three cross slices
run "0.8b deployment"      $VENV  throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa fast
run "0.8b fla+math"        $VENV  throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa math $SLICE
run "0.8b fallback+flash"  $STOCK throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa fast $SLICE
run "0.8b naive"           $STOCK throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa math $SLICE

# --- CoDA (pure attention): deployment + naive
run "coda deployment"      $VENV  throughput.py --model coda-1.7b-base --out $OUT --sdpa fast
run "coda naive"           $STOCK throughput.py --model coda-1.7b-base --out $OUT --sdpa math $SLICE

# --- 4B hybrid: deployment + three slices
run "4b deployment"        $VENV  throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa fast
run "4b fla+math"          $VENV  throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa math $SLICE
run "4b fallback+flash"    $STOCK throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa fast $SLICE
run "4b naive"             $STOCK throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa math $SLICE

# --- 7-9B tier: iters 10 (recorded per-record), deployment + naive
run "dream deployment"     $VENV  throughput.py --model dream-7b-base --out $OUT --sdpa fast --iters 10
run "dream naive"          $STOCK throughput.py --model dream-7b-base --out $OUT --sdpa math --iters 10 $SLICE
run "dream-coder deployment" $VENV  throughput.py --model dream-coder-7b-base --out $OUT --sdpa fast --iters 10
run "dream-coder naive"    $STOCK throughput.py --model dream-coder-7b-base --out $OUT --sdpa math --iters 10 $SLICE
run "llada deployment"     $VENV  throughput.py --model llada-8b-base --out $OUT --sdpa fast --iters 10
run "llada naive"          $STOCK throughput.py --model llada-8b-base --out $OUT --sdpa math --iters 10 $SLICE
run "9b deployment"        $VENV  throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa fast --iters 10
run "9b fla+math"          $VENV  throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa math --iters 10 $SLICE
run "9b fallback+flash"    $STOCK throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa fast --iters 10 $SLICE
run "9b naive"             $STOCK throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa math --iters 10 $SLICE

echo "=== phase 3+4 done $(date -Is) ==="
