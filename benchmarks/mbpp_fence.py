"""MBPP with a markdown code fence instead of the [BEGIN]/[DONE] scaffold
(task name `mbpp_ticks`). A LABELLED format-sensitivity variant, never the
Table-4 MBPP column: the fence's sign tracks code-instruction exposure
(dQwen +4/+6pp, Dream-Coder +0.20, LLaDA -2.40) because the task->fenced-
solution pattern is corpus-dependent. Same frozen 3-shot examples as mbpp.py
(protocol invariant)."""

from ._grading import pass_at_1
from .mbpp import FEWSHOT_SAMPLES

TASK = {
    "task": "mbpp_ticks",
    "dataset_path": "google-research-datasets/mbpp",
    "dataset_name": "full",
    "test_split": "test",
    "doc_to_text": "You are an expert Python programmer, and here is your task: "
                   "{{text}} Your code should pass these tests:\n\n"
                   "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n```python\n",
    "doc_to_target": "{% if is_fewshot is defined %}{{code}}\n```"
                     "{% else %}{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}{% endif %}",
    "unsafe_code": True,
    "target_delimiter": "",
    "fewshot_config": {"sampler": "first_n", "samples": FEWSHOT_SAMPLES},
    "num_fewshot": 3,
    "metric_list": [{"metric": pass_at_1, "aggregation": "mean",
                     "higher_is_better": True}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["```"], "do_sample": False},
    "repeats": 1,
    "metadata": {"version": 1.0},
}
