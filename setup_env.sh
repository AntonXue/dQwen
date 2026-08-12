#!/usr/bin/env bash
# Finish the environment: install requirements.txt, then apply the two
# source patches that pinned lm-eval 0.4.8 needs under transformers 5.x.
# The patches touch lm-eval's OWN files, so pip cannot express them.
# Idempotent: safe to re-run.
#
#   PYTHON=/path/to/env/bin/python bash setup_env.sh
set -euo pipefail

PY=${PYTHON:-/ssd1/ayx98/miniconda3/envs/qwen35/bin/python}
PIP="$PY -m pip"

echo ">> installing requirements.txt"
$PIP install -q -r "$(dirname "$0")/requirements.txt"

SP=$($PY -c "import lm_eval, os; print(os.path.dirname(os.path.dirname(lm_eval.__file__)))" 2>/dev/null \
     || $PY -c "import site; print(site.getsitepackages()[0])")

# PATCH 1: lm_eval 0.4.8's models/hf_vlms.py binds
# transformers.AutoModelForVision2Seq at class-definition time; transformers
# 5.x renamed it to AutoModelForImageTextToText, and hf_vlms is imported at
# package init -- without this, `from lm_eval import ...` raises before
# anything runs. Same class, pure rename; we never evaluate VLMs.
HF_VLMS="$SP/lm_eval/models/hf_vlms.py"
if [ ! -f "$HF_VLMS" ]; then
  echo "ERROR: $HF_VLMS not found; lm_eval install may be broken." >&2
  echo "       try: $PIP install --force-reinstall --no-deps lm_eval==0.4.8" >&2
  exit 1
fi
if grep -q "AutoModelForVision2Seq" "$HF_VLMS"; then
  echo ">> patching hf_vlms.py: AutoModelForVision2Seq -> AutoModelForImageTextToText"
  sed -i 's/AutoModelForVision2Seq/AutoModelForImageTextToText/g' "$HF_VLMS"
else
  echo ">> hf_vlms.py already patched"
fi

# PATCH 2: stock gsm8k task yamls reference the dataset as bare `gsm8k`;
# newer huggingface_hub requires `namespace/name` (the dataset lives at
# openai/gsm8k). Our own specs carry the right id already -- this patch
# keeps the STOCK side loadable, which tests/task_freeze_gate.py compares
# against. Data-only, no behaviour change.
GSM_DIR="$SP/lm_eval/tasks/gsm8k"
if [ -d "$GSM_DIR" ] && grep -rql "dataset_path: gsm8k$" "$GSM_DIR"/*.yaml 2>/dev/null; then
  echo ">> patching gsm8k task yamls: dataset_path gsm8k -> openai/gsm8k"
  sed -i 's|dataset_path: gsm8k$|dataset_path: openai/gsm8k|' "$GSM_DIR"/*.yaml
else
  echo ">> gsm8k task yamls already patched (or absent)"
fi

echo ">> verifying import under transformers $($PY -c 'import transformers; print(transformers.__version__)')"
$PY -c "from lm_eval import simple_evaluate; from lm_eval.api.registry import register_model; print('lm_eval import OK')"
