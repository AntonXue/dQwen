"""HumanEval+ — same 164 problems and prompts as humaneval, EvalPlus's ~78x
denser test suite (dataset field `test` carries the full suite). Task name
stays `humaneval_plus_sound` for store continuity: "sound" = the rendered
comparator's empty-expected bug is patched at load time (see
_grading.plus_process_docs)."""

from ._grading import (build_predictions, pass_at_k,
                                   plus_process_docs)

TASK = {
    "task": "humaneval_plus_sound",
    "dataset_path": "evalplus/humanevalplus",
    "test_split": "test",
    "process_docs": plus_process_docs,
    "doc_to_text": "{{prompt}}",
    "doc_to_target": "{{test}}\ncheck({{entry_point}})",
    "unsafe_code": True,
    "target_delimiter": " ",
    "num_fewshot": 0,
    "metric_list": [{"metric": pass_at_k, "aggregation": "mean",
                     "higher_is_better": True, "k": [1]}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                          "max_gen_toks": 1024, "do_sample": False},
    "repeats": 1,
    "filter_list": [{"name": "create_test",
                     "filter": [{"function": "custom",
                                 "filter_fn": build_predictions}]}],
    "metadata": {"version": 1.0},
}
