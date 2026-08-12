"""Shared helpers for dqeval's local task variants.

The variants differ from their stock lm-eval counterparts in exactly one place
each, so everything else -- metric, and in particular the exact few-shot problems
-- is IMPORTED from the stock task rather than copied. A copied few-shot block
that drifted would turn a one-variable probe into a two-variable one.
"""

from lm_eval.tasks.mbpp.utils import list_fewshot_samples, pass_at_1  # noqa: F401
# STOCK humaneval grading path, re-exported for humaneval_plus_sound.yaml
from lm_eval.tasks.humaneval.utils import (  # noqa: F401
    build_predictions,
    pass_at_k,
)

# Re-exported so the RACE variant's yaml can reference the STOCK preprocessing by
# `!function utils.*` instead of copying it (copies drift; imports cannot).
from lm_eval.tasks.race.preprocess_race import (  # noqa: F401
    doc_to_choice as race_doc_to_choice,
    doc_to_target as race_doc_to_target,
    doc_to_text as race_doc_to_text,
)

# --- seeded subsampling ----------------------------------------------------
# WHY NOT `--limit N`: lm-eval's limit takes the FIRST N docs. Benchmark splits are
# frequently grouped (RACE by article; many sets by source/category), so first-N is a
# biased slice, not a sample. This program has already been bitten by exactly that --
# the naive first-50 HumanEval subset ran ~10pp high and had to be rebuilt as HE-50R.
# Shuffle-then-take with a PINNED seed gives a uniform sample that is identical across
# models and reruns, which is what makes cross-arm comparisons valid.
SUBSAMPLE_SEED = 1234
SUBSAMPLE_N = 1000


def _subsample(docs, n=SUBSAMPLE_N, seed=SUBSAMPLE_SEED):
    if len(docs) <= n:
        return docs
    return docs.shuffle(seed=seed).select(range(n))


def hellaswag_sub_process_docs(dataset):
    """Stock hellaswag preprocessing, then a seeded 1000-doc sample (of 10,042)."""
    from lm_eval.tasks.hellaswag.utils import process_docs as stock
    return _subsample(stock(dataset))


def race_sub_process_docs(dataset):
    """Seeded 1000-doc sample of RACE-high test (1,045) -- near-full, but sampled
    rather than truncated so the article grouping cannot skew it."""
    return _subsample(dataset)


# --- plus-suite comparator soundness -----------------------------------------
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
