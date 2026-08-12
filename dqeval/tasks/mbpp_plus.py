"""MBPP+ — EvalPlus's MBPP-sanitized set (378 problems, edited prompts, ~34x
denser tests) under OUR MBPP protocol: the same [BEGIN]/[DONE] scaffold and
the same frozen 3-shot examples (imported from mbpp.py — a protocol
invariant, not a convenience).

Task name stays `mbpp_plus_full` for store continuity, and as a reminder of
WHY this file exists: lm-eval's stock `mbpp_plus` grades against
test_list[0..2] — the three ORIGINAL asserts — and never executes the plus
suite in the dataset's `test` field. This config targets the real suite.

COMPARABILITY: published MBPP+ numbers (EvalPlus's own pipeline) use a
0-shot docstring prompt, not this scaffold — this column is a re-measurement
under the house protocol, not a reproduction (Dream-tripwire-style
disclosure in the manuscript ledger)."""

from dqeval.tasks._grading import pass_at_1, plus_process_docs
from dqeval.tasks.mbpp import FEWSHOT_SAMPLES

TASK = {
    "task": "mbpp_plus_full",
    "dataset_path": "evalplus/mbppplus",
    "test_split": "test",
    "process_docs": plus_process_docs,
    "doc_to_text": "You are an expert Python programmer, and here is your task: "
                   "{{prompt if prompt is defined else text}} Your code should "
                   "pass these tests:\n\n"
                   "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n[BEGIN]\n",
    "doc_to_target": "{% if is_fewshot is defined %}{{code}}\n[DONE]"
                     "{% else %}{{test}}{% endif %}",
    "unsafe_code": True,
    "target_delimiter": "",
    "fewshot_config": {"sampler": "first_n", "samples": FEWSHOT_SAMPLES},
    "num_fewshot": 3,
    "metric_list": [{"metric": pass_at_1, "aggregation": "mean",
                     "higher_is_better": True}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["[DONE]"], "do_sample": False},
    "repeats": 1,
    "metadata": {"version": 1.0},
}
