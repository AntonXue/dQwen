#!/usr/bin/env bash
# Install lm-evaluation-harness 0.4.8 into the qwen35 env and apply the one patch
# it needs under transformers 5.x. Idempotent: safe to re-run.
#
# WHY 0.4.8 (pinned, not latest): LLaDA and Dream both report through lm_eval 0.4.8.
# The harness version controls task configs (few-shot handling, prompt templates,
# extraction filters), so matching theirs is part of reproducing their numbers. A
# newer lm_eval would dodge the patch below but risks silent task-config drift in
# exactly the comparison we are trying to validate.
#
# THE PATCH: lm_eval 0.4.8's models/hf_vlms.py binds
#   AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq
# at class-definition time. transformers 5.x renamed that class to
# AutoModelForImageTextToText, and hf_vlms is imported at package init, so without
# the fix `from lm_eval import ...` raises AttributeError before anything runs.
# We only evaluate text models, so the VLM class is never used -- this just lets the
# package import. Verified: the two classes are the same rename, not a behaviour change.
set -euo pipefail

PY=${PYTHON:-/ssd1/ayx98/miniconda3/envs/qwen35/bin/python}
PIP="$PY -m pip"

echo ">> installing lm_eval==0.4.8"
$PIP install -q "lm_eval==0.4.8"

SP=$($PY -c "import lm_eval, os; print(os.path.dirname(os.path.dirname(lm_eval.__file__)))" 2>/dev/null \
     || $PY -c "import site; print(site.getsitepackages()[0])")
HF_VLMS="$SP/lm_eval/models/hf_vlms.py"

if [ ! -f "$HF_VLMS" ]; then
  echo "ERROR: $HF_VLMS not found; lm_eval install may be broken (orphaned dist-info?)." >&2
  echo "       try: $PIP install --force-reinstall --no-deps lm_eval==0.4.8" >&2
  exit 1
fi

if grep -q "AutoModelForVision2Seq" "$HF_VLMS"; then
  echo ">> patching hf_vlms.py: AutoModelForVision2Seq -> AutoModelForImageTextToText"
  sed -i 's/AutoModelForVision2Seq/AutoModelForImageTextToText/g' "$HF_VLMS"
else
  echo ">> hf_vlms.py already patched"
fi

# PATCH 2: the gsm8k task yamls reference the dataset as bare `gsm8k`, but newer
# huggingface_hub requires `namespace/name` and the dataset moved to `openai/gsm8k`.
# Without this, gsm8k* tasks die with HfUriError. Data-only, no behaviour change.
GSM_DIR="$SP/lm_eval/tasks/gsm8k"
if [ -d "$GSM_DIR" ] && grep -rql "dataset_path: gsm8k$" "$GSM_DIR"/*.yaml 2>/dev/null; then
  echo ">> patching gsm8k task yamls: dataset_path gsm8k -> openai/gsm8k"
  sed -i 's|dataset_path: gsm8k$|dataset_path: openai/gsm8k|' "$GSM_DIR"/*.yaml
else
  echo ">> gsm8k task yamls already patched (or absent)"
fi

# MATH answer verification: minerva_math needs these (same as LLaDA's eval script).
# NOTE antlr4-python3-runtime==4.11 conflicts with omegaconf/hydra's 4.9 pin -- benign
# here (we don't use hydra), and it's what math_verify requires.
echo ">> installing MATH verification deps (sympy, math_verify, antlr4 4.11)"
$PIP install -q "antlr4-python3-runtime==4.11" math_verify sympy || true

echo ">> verifying import under transformers $($PY -c 'import transformers; print(transformers.__version__)')"
$PY -c "from lm_eval import simple_evaluate; from lm_eval.api.registry import register_model; print('lm_eval import OK')"
echo ">> done"
