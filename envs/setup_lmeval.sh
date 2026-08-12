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

# PATCH 4: bare dataset ids. Newer huggingface_hub requires `namespace/name`, and
# newer `datasets` refuses script-based loaders. Same class as the gsm8k patch above.
#   hellaswag -> Rowan/hellaswag      (parquet, no script)
#   winogrande -> allenai/winogrande  (parquet, no script)
# NOT patched: piqa. Its canonical repo (ybisk/piqa) is script-based with no official
# parquet mirror; third-party mirrors exist but are unvetted, and silently swapping a
# benchmark's data source is exactly the kind of change that invalidates a comparison.
# Left failing on purpose -- decide provenance before enabling.
for pair in "hellaswag:Rowan/hellaswag" "winogrande:allenai/winogrande"; do
  name="${pair%%:*}"; repo="${pair##*:}"
  d="$SP/lm_eval/tasks/$name"
  if [ -d "$d" ] && grep -rql "^dataset_path: $name$" "$d"/*.yaml 2>/dev/null; then
    echo ">> patching $name dataset_path -> $repo"
    sed -i "s|^dataset_path: $name$|dataset_path: $repo|" "$d"/*.yaml
  fi
done

# MATH answer verification: minerva_math needs these (same as LLaDA's eval script).
# NOTE antlr4-python3-runtime==4.11 conflicts with omegaconf/hydra's 4.9 pin -- benign
# here (we don't use hydra), and it's what math_verify requires.
echo ">> installing MATH verification deps (sympy, math_verify, antlr4 4.11)"
$PIP install -q "antlr4-python3-runtime==4.11" math_verify sympy || true

# PATCH 3: HF `evaluate` code_eval defaults to num_workers=4 AND a flat 3s timeout
# for a problem's ENTIRE test script (~3ms/case on 1000-case plus suites; verdicts
# become machine-load-sensitive). lm_eval calls .compute() with neither knob, so we
# inject env-overridable defaults: LM_EVAL_CODE_WORKERS (32) + LM_EVAL_CODE_TIMEOUT
# (10s). Idempotent: keyed on the _CODE_TIMEOUT marker (upgrades workers-only envs).
echo ">> patching lm_eval code_eval workers+timeout (LM_EVAL_CODE_WORKERS/LM_EVAL_CODE_TIMEOUT)"
$PY - "$SP" <<'PYEOF'
import os, sys
sp = sys.argv[1]
imp = 'import evaluate as hf_evaluate\n'
block = ('import os\n\nimport evaluate as hf_evaluate\n\n'
         '# dqeval patch: HF code_eval defaults to num_workers=4; bump to a sane,\n'
         '# env-overridable default so grading uses the cores available.\n'
         '_CODE_WORKERS = int(os.environ.get("LM_EVAL_CODE_WORKERS") or min(32, os.cpu_count() or 8))\n'
         '# dqeval patch: code_eval defaults to a flat 3s for a problem\'s ENTIRE test\n'
         '# script; 10s keeps verdicts deterministic under load. Env-overridable.\n'
         '_CODE_TIMEOUT = float(os.environ.get("LM_EVAL_CODE_TIMEOUT") or 10.0)\n')
kw = '        num_workers=_CODE_WORKERS,\n        timeout=_CODE_TIMEOUT,\n'
targets = {
    os.path.join(sp, "lm_eval/tasks/mbpp/utils.py"):
        ('        k=[1],\n    )[0]["pass@1"]',
         '        k=[1],\n' + kw + '    )[0]["pass@1"]'),
    os.path.join(sp, "lm_eval/tasks/humaneval/utils.py"):
        ('        k=k,\n    )',
         '        k=k,\n' + kw + '    )'),
}
for path, (old, new) in targets.items():
    if not os.path.exists(path):
        print("   skip (absent):", path); continue
    src = open(path).read()
    if "_CODE_TIMEOUT" in src:
        print("   already patched:", os.path.basename(os.path.dirname(path))); continue
    if "_CODE_WORKERS" in src:   # workers-only env: upgrade in place
        src = src.replace(
            '_CODE_WORKERS = int(os.environ.get("LM_EVAL_CODE_WORKERS") or min(32, os.cpu_count() or 8))\n',
            '_CODE_WORKERS = int(os.environ.get("LM_EVAL_CODE_WORKERS") or min(32, os.cpu_count() or 8))\n'
            '_CODE_TIMEOUT = float(os.environ.get("LM_EVAL_CODE_TIMEOUT") or 10.0)\n', 1)
        src = src.replace('        num_workers=_CODE_WORKERS,\n',
                          kw, 1)
    else:
        src = src.replace(imp, block, 1).replace(old, new, 1)
    open(path, "w").write(src)
    print("   patched:", path)
PYEOF

echo ">> verifying import under transformers $($PY -c 'import transformers; print(transformers.__version__)')"
$PY -c "from lm_eval import simple_evaluate; from lm_eval.api.registry import register_model; print('lm_eval import OK')"
echo ">> done"
