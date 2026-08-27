#!/bin/bash
# v2 release battery: uniform --iters 50 (5 warmup + 50 timed), priority order.
# Tier 1 first, then 9B (scale anchor), then family, then comparators, then
# envelope extensions. Stability repeats (A/B/C) go to their own dirs so the
# main dataset stays one-measurement-per-cell. v1 (mixed 10/20 iters) kept as
# the iters-sensitivity cross-check; its 9B partial was deleted.
cd /home1/11079/antonxue/foo/dQwen
OUT=_runs/20260827-throughput-gh200-i50
STAB=_runs/20260827-throughput-gh200-stability
SLICE="--batches 1,16 --lengths 1024,4096"
VENV=$WORK/tp-venv/bin/python
STOCK=/scratch/11079/antonxue/miniconda3/envs/qwen35/bin/python
I=50
run() { echo "=== $1 $(date -Is) ==="; shift; CUDA_VISIBLE_DEVICES=0 "$@" || echo "!!! FAILED: $*"; }

# --- Tier 1: the boss claim
run "2b deployment"        $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa fast --iters $I
run "control deployment"   $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $OUT --sdpa fast --iters $I
run "2b fla+math"          $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa math --iters $I $SLICE
run "2b fallback+flash"    $STOCK throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa fast --iters $I $SLICE
run "2b naive"             $STOCK throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa math --iters $I $SLICE
run "control naive"        $STOCK throughput.py --model dqwen3-1.7b-base-v3 --out $OUT --sdpa math --iters $I $SLICE
run "stability A 2b"       $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $STAB-a --sdpa fast --iters $I $SLICE
run "stability A control"  $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $STAB-a --sdpa fast --iters $I $SLICE

# --- 9B: the scale anchor
run "9b deployment"        $VENV  throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa fast --iters $I
run "9b fla+math"          $VENV  throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa math --iters $I $SLICE
run "9b fallback+flash"    $STOCK throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa fast --iters $I $SLICE
run "9b naive"             $STOCK throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa math --iters $I $SLICE

# --- family scaling
run "0.8b deployment"      $VENV  throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa fast --iters $I
run "0.8b fla+math"        $VENV  throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa math --iters $I $SLICE
run "0.8b fallback+flash"  $STOCK throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa fast --iters $I $SLICE
run "0.8b naive"           $STOCK throughput.py --model dqwen3.5-0.8b-base-v3 --out $OUT --sdpa math --iters $I $SLICE
run "4b deployment"        $VENV  throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa fast --iters $I
run "4b fla+math"          $VENV  throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa math --iters $I $SLICE
run "4b fallback+flash"    $STOCK throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa fast --iters $I $SLICE
run "4b naive"             $STOCK throughput.py --model dqwen3.5-4b-base-v3 --out $OUT --sdpa math --iters $I $SLICE

# --- comparators (deployment + naive; no GDN axis)
run "coda deployment"      $VENV  throughput.py --model coda-1.7b-base --out $OUT --sdpa fast --iters $I
run "coda naive"           $STOCK throughput.py --model coda-1.7b-base --out $OUT --sdpa math --iters $I $SLICE
run "stability B 2b"       $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $STAB-b --sdpa fast --iters $I $SLICE
run "stability B control"  $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $STAB-b --sdpa fast --iters $I $SLICE
run "dream deployment"     $VENV  throughput.py --model dream-7b-base --out $OUT --sdpa fast --iters $I
run "dream naive"          $STOCK throughput.py --model dream-7b-base --out $OUT --sdpa math --iters $I $SLICE
run "dream-coder deployment" $VENV throughput.py --model dream-coder-7b-base --out $OUT --sdpa fast --iters $I
run "dream-coder naive"    $STOCK throughput.py --model dream-coder-7b-base --out $OUT --sdpa math --iters $I $SLICE
run "llada deployment"     $VENV  throughput.py --model llada-8b-base --out $OUT --sdpa fast --iters $I
run "llada naive"          $STOCK throughput.py --model llada-8b-base --out $OUT --sdpa math --iters $I $SLICE

# --- envelope extensions (new cells, main dir; keys never collide with the grid)
run "longL 2b"             $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 16384,32768 --batches 1,2,4
run "longL control"        $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 16384,32768 --batches 1,2,4
run "longL 9b"             $VENV  throughput.py --model dqwen3.5-9b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 16384,32768 --batches 1,2,4
run "crossover-L 2b"       $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 1536,2560,3072,3584,5120,6144 --batches 1
run "crossover-L control"  $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 1536,2560,3072,3584,5120,6144 --batches 1
run "crossover-B 2b"       $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 2048 --batches 3,6,12,24
run "crossover-B control"  $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 2048 --batches 3,6,12,24
run "batchsat 2b"          $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 256,512,1024 --batches 64,128
run "batchsat control"     $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $OUT --sdpa fast --iters $I --lengths 256,512,1024 --batches 64,128

# --- stability C (end-of-session spread) + fa2 delta
run "stability C 2b"       $VENV  throughput.py --model dqwen3.5-2b-base-v3 --out $STAB-c --sdpa fast --iters $I $SLICE
run "stability C control"  $VENV  throughput.py --model dqwen3-1.7b-base-v3 --out $STAB-c --sdpa fast --iters $I $SLICE
run "fa2 delta"            $VENV  _runs/20260827-throughput-gh200-fa2/fa2_delta.py

echo "=== v2 battery done $(date -Is) ==="
