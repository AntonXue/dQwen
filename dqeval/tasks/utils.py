"""Shared helpers for dqeval's local task variants.

The variants differ from their stock lm-eval counterparts in exactly one place
each, so everything else -- metric, and in particular the exact few-shot problems
-- is IMPORTED from the stock task rather than copied. A copied few-shot block
that drifted would turn a one-variable probe into a two-variable one.
"""

from lm_eval.tasks.mbpp.utils import list_fewshot_samples, pass_at_1  # noqa: F401

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
