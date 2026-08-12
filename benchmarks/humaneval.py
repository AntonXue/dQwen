"""HumanEval (164 problems; completion-style: the prompt holds the def, the
model writes the body). Frozen from pinned lm-eval 0.4.8's `humaneval` task;
fidelity-gated byte-identical 2026-08-12. Grading via _grading (10s/problem,
parallel)."""

from ._grading import build_predictions, pass_at_k

TASK = {
    "task": "humaneval",
    "dataset_path": "openai/openai_humaneval",
    "test_split": "test",
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
