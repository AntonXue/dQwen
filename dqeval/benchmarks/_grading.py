"""Code-execution grading for the code benchmarks (HF code_eval underneath).

Knobs, recorded per-cell in the meta record's provenance:
  LM_EVAL_CODE_WORKERS  parallel grading processes     (default: cores, <=32)
  LM_EVAL_CODE_TIMEOUT  seconds per problem's WHOLE test script (default 10
                        -- the plus suites run ~1000 cases per problem, and
                        a tight budget makes verdicts machine-load-sensitive)

Grader lineage and the EvalPlus agreement study:
_claude/20260812-113000-lmeval-only-plus-columns-evalplus-removed.md
"""

import os
import evaluate as hf_evaluate


_CODE_WORKERS = int(os.environ.get("LM_EVAL_CODE_WORKERS") or min(32, os.cpu_count() or 8))
_CODE_TIMEOUT = float(os.environ.get("LM_EVAL_CODE_TIMEOUT") or 10.0)

_code_eval = None


def _metric():
    """HF code_eval, loaded lazily (its import-time canary requires
    HF_ALLOW_CODE_EVAL=1, which run.py sets for unsafe benchmarks)."""
    global _code_eval
    if _code_eval is None:
        _code_eval = hf_evaluate.load("code_eval")
    return _code_eval


def pass_at_k(references: list, predictions: list, k: list = None):
    """HumanEval-style: predictions are full programs (prompt + completion,
    assembled by `build_predictions`). Behavior-identical to stock lm-eval."""
    assert k is not None
    if isinstance(k, int):
        k = [k]
    return _metric().compute(references=references, predictions=predictions,
                             k=k, num_workers=_CODE_WORKERS,
                             timeout=_CODE_TIMEOUT)[0]


def pass_at_1(references, predictions):
    """MBPP-style: the completion IS the program. Behavior-identical to stock."""
    return _metric().compute(references=references, predictions=[predictions],
                             k=[1], num_workers=_CODE_WORKERS,
                             timeout=_CODE_TIMEOUT)[0]["pass@1"]


def build_predictions(resps: list, docs: list) -> list:
    """HumanEval filter: completions grade as prompt + completion."""
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


# plus-suite comparator soundness
# The rendered test scripts in evalplus/humanevalplus + evalplus/mbppplus carry
# an inline comparator whose `is_floats([])` is vacuously True (all() on
# empty), routing empty-expected checks into np.allclose -- and numpy
# broadcasts shape (1,) against (0,) to an empty result, so
# np.allclose([wrong], []) == True: any WRONG non-empty output passes whenever
# the expected value is an empty list/tuple. Caught 2026-08-12 (HumanEval/62 +
# /96 grader A/B vs the evalplus package, which compares these correctly).

_UNSOUND = "        return all(isinstance(i, float) for i in x)"
_SOUND = "        return len(x) > 0 and all(isinstance(i, float) for i in x)"


def _sound_test(doc):
    # a few problems render a bespoke comparator instead (e.g. HumanEval/32
    # checks a residual tolerance); those don't carry this bug -- skip them
    return {**doc, "test": doc["test"].replace(_UNSOUND, _SOUND, 1)}


def plus_process_docs(dataset):
    n = sum(_UNSOUND in t for t in dataset["test"])
    if not n or n < 0.8 * len(dataset):
        raise ValueError(
            f"comparator soundness patch matched {n}/{len(dataset)} docs -- "
            "the rendered format drifted; re-derive the patch."
        )
    return dataset.map(_sound_test)
