#!/bin/bash
# MBPP+ generation lanes (EvalPlus mbpp: 378 sanitized problems, edited
# prompts -- NOT a regrade of the MBPP cells). Family + control use the
# GATE-style standard full-canvas decode (sdpa math pinned in the driver);
# AR counterparts use greedy transformers generate. Grading (base+plus)
# runs inline on CPU after each model finishes; samples are saved either
# way, so regrading/sanitizing later never needs a GPU.
# Usage: bash lane.sh 1   |   bash lane.sh 2
cd /home/ayx98/foo/dQwen
PY=/ssd1/ayx98/miniconda3/envs/qwen35/bin/python
export PYTHONPATH=/home/ayx98/foo/dQwen
D="$(dirname "$0")"

dlm() {  # dlm <registry-name> <revision> <label>
  CUDA_VISIBLE_DEVICES=$GPU $PY evalplus_driver.py --dataset mbpp \
    --model "$1" --revision "$2" --gen-length 512 --block-length 512 \
    --steps-per-block 512 \
    --samples "$D/$3@$2__mbpp-plus__standard-static-s512.jsonl" \
    > "$D/$3@$2__mbpp-plus__standard-static-s512.log" 2>&1
}
ar() {   # ar <hf-id>
  local tag="${1//\//_}"
  CUDA_VISIBLE_DEVICES=$GPU $PY evalplus_driver.py --dataset mbpp \
    --model "$1" --ar --gen-length 512 \
    --samples "$D/$tag@main__mbpp-plus__ar.jsonl" \
    > "$D/$tag@main__mbpp-plus__ar.log" 2>&1
}

GPU=$1
if [ "$GPU" = "1" ]; then
  dlm dqwen3.5-0.8b-base-v3 step50000-swa EER6b_dQwen3.5-0.8B-Base-v3
  dlm dqwen3.5-2b-base-v3   step50000-swa EER6b_dQwen3.5-2B-Base-v3
  dlm dqwen3.5-4b-base-v3   step50000-swa EER6b_dQwen3.5-4B-Base-v3
  dlm dqwen3-1.7b-base-v3   step50000-swa EER6b_dQwen3-1.7B-Base-v3
else
  dlm dqwen3.5-9b-base-v3   step50000-swa EER6b_dQwen3.5-9B-Base-v3
  ar Qwen/Qwen3.5-0.8B
  ar Qwen/Qwen3.5-2B
  ar Qwen/Qwen3.5-4B
  ar Qwen/Qwen3.5-9B
  ar Qwen/Qwen3-0.6B
  ar Qwen/Qwen3-1.7B
fi
echo "lane $GPU done"
