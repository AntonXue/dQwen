# Benchmarks are Python files now: dqeval/tasks/*.py, yaml layer deleted

> 2026-08-12, Anton's ruling: "we need this file structure to be easily
> skimmable by people — even if we have to redundantly have a bunch of
> (usefully named!) files." lm-eval stays the execution engine; its yaml
> format does not (get_task_dict accepts config dicts directly).

## The layout

One benchmark = one file exporting a complete config (`TASK`) or group
(`GROUP`): `humaneval.py`, `humaneval_plus.py`, `mbpp.py`, `mbpp_plus.py`,
`mbpp_fence.py`, `gsm8k.py`, `minerva_math.py`, `mmlu.py`,
`hellaswag_sub.py`, `race_sub.py`; `_grading.py` (code-exec pass@1 with the
workers/timeout knobs + the plus-comparator soundness patch);
`__init__.py` = the registry (`task_config(name)` returns a DEEP COPY —
lm-eval's loader mutates what it's given).

Frozen mechanically, not transcribed: fewshot literals (mbpp 3-shot, gsm8k
8 CoT) and the 57-subject MMLU table were extracted from the pinned
install; minerva_math freezes CONFIG but imports the Minerva
extraction/math_verify LOGIC from the pin (single source for the 200 lines
where a transcription slip silently changes numbers).

## The gate (permanent): tests/task_freeze_gate.py

Vendored == pinned stock, checked as doc fingerprints + byte-identical
rendered prompts/targets/fewshot-contexts. PASSED 2026-08-12 across all 70
leaf tasks (57 mmlu + 7 minerva + humaneval/mbpp/gsm8k_cot); local variants
construct clean. Re-run whenever tasks/ changes or the lm_eval pin bumps —
at a bump, the frozen copies PROTECT the protocol.

## What died

- All task yamls + tasks/utils.py re-export shim + TaskManager
  include_path injection.
- setup_lmeval.sh PATCH 3 (site-packages grading patch): workers/timeout
  are ordinary code in _grading.py now (LM_EVAL_CODE_WORKERS /
  LM_EVAL_CODE_TIMEOUT, defaults 32 / 10s, recorded per-cell in
  provenance).

## Traps hit (recorded for the next reader)

1. lm-eval's dict loader POPS "task" from the config and never writes the
   name back onto unregistered tasks → they run as [Task: None] and result
   aliasing crashes with a NoneType concat. grid._restore_names() is the
   repair (lm-eval's python-class branch does the same fixup itself,
   self-described in their source as "very scuffed"). Registered names
   (humaneval, mbpp, gsm8k_cot, minerva/mmlu leaves) never hit it because
   the loader merges our complete dict over the stock base — which the
   freeze gate proves is a no-op merge.
2. mbpp fewshot docs carry `text`, test docs carry `prompt` — mbpp_plus's
   doc_to_text must keep the `prompt if prompt is defined else text`
   conditional (bitten once via yaml already, kept in the .py).

Task names are UNCHANGED (humaneval_plus_sound, mbpp_plus_full, ...), so
the grid_v1 store and summarize.py needed no migration. Smoked end-to-end
via run.py: all 8 BENCH benchmarks ran stripe cells on the vendored
configs (gsm8k/math/mmlu/humaneval-plus/mbpp-plus/mbpp-fence today;
humaneval/mbpp are byte-identical registered-merge no-ops).

## UPDATE: the single-entry reshape (same day)

Anton's target: "call a single file and say run this model, this sampler
config, this dataset, this shard" + descriptive names everywhere. Final:

| was | is |
|---|---|
| grid.py + harness.py | **cell_runner.py** — THE entry: `run_cell(Cell(...))`; imports the full stack (root run.py is the login-node-safe CLI) |
| adapter.py | **models.py** (registry + contract + loading + pins) |
| sampler.py | **samplers.py** (decode + MC-NELBO; now owns GenOutput) |
| families/ | **model_families/** |
| tasks/ | **benchmarks/** ("benchmark" = our file, "task" = lm-eval's name string — vocabulary collision gone) |

Orthogonality rule enforced and AST-checked: models.py and samplers.py do
not import each other (GenOutput moved to samplers — it is the sampler's
return type; generate() duck-types the adapter). model_families/* import
both; cell_runner sits on top; benchmarks/ touch only _grading.

Gates: freeze gate PASSED post-rename; smoke matrix covered all four
execution paths (AR generate, AR gsm8k, DLM block-decode via RecordingLM,
DLM MC-NELBO via DQEvalLM.loglikelihood). hellaswag_sub/race_sub deleted
earlier the same day (retention-probe orphans; striping obsoleted the
subsample trick).
