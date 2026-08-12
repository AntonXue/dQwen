"""MBPP+ with the markdown-fence scaffold (task name `mbpp_plus_ticks`) —
the fourth quadrant of the prompt-format probe:

                 [BEGIN]/[DONE]      ```python fence
    MBPP-500     mbpp.py             mbpp_fence.py
    MBPP+-378    mbpp_plus.py        THIS FILE

Same sanitized problems and real plus suite as mbpp_plus.py, same frozen
3-shot examples (protocol invariant), only the scaffold differs — so the
fence effect measured on MBPP (corpus-dependent: dQwen +4/+6pp, LLaDA
-2.40) can be re-read under ~34x denser tests. A LABELLED
format-sensitivity variant, never the Table-4 MBPP+ column."""

from ._grading import pass_at_1, plus_process_docs
from .mbpp import FEWSHOT_SAMPLES

TASK = {
    "task": "mbpp_plus_ticks",
    "dataset_path": "evalplus/mbppplus",
    "test_split": "test",
    "process_docs": plus_process_docs,
    "doc_to_text": "You are an expert Python programmer, and here is your task: "
                   "{{prompt if prompt is defined else text}} Your code should "
                   "pass these tests:\n\n"
                   "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n```python\n",
    "doc_to_target": "{% if is_fewshot is defined %}{{code}}\n```"
                     "{% else %}{{test}}{% endif %}",
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
