"""Shared helpers for dqeval's local task variants.

The variants differ from their stock lm-eval counterparts in exactly one place
each, so everything else -- metric, and in particular the exact few-shot problems
-- is IMPORTED from the stock task rather than copied. A copied few-shot block
that drifted would turn a one-variable probe into a two-variable one.
"""

from lm_eval.tasks.mbpp.utils import list_fewshot_samples, pass_at_1  # noqa: F401
