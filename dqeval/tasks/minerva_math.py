"""MATH (Hendrycks), Minerva 4-shot protocol (`minerva_math`, 7 subject
subtasks + size-weighted group). Config frozen from pinned lm-eval 0.4.8;
fidelity-gated 2026-08-12.

CONFIG is frozen here; the EXTRACTION/GRADING LOGIC is imported from the
pinned install (Minerva answer normalization is ~200 lines of regex, and
math_verify's symbolic equivalence — the ruled MATH column metric — lives in
process_results). Copying that logic would create a second source of truth
for exactly the code where a silent transcription slip changes numbers;
importing keeps it single-sourced while the pin holds.

Metrics per doc: exact_match (strict Minerva) AND math_verify — the
manuscript column is math_verify (Anton's 2026-08-12 ruling; strict
collapses to a format test on big post-trained AR models)."""

from lm_eval.tasks.minerva_math.utils import (
    doc_to_text,
    list_fewshot_samples,
    process_docs,
    process_results,
)

SUBJECTS = [
    ("minerva_math_algebra", "algebra"),
    ("minerva_math_counting_and_prob", "counting_and_probability"),
    ("minerva_math_geometry", "geometry"),
    ("minerva_math_intermediate_algebra", "intermediate_algebra"),
    ("minerva_math_num_theory", "number_theory"),
    ("minerva_math_prealgebra", "prealgebra"),
    ("minerva_math_precalc", "precalculus"),
]


def _leaf(task_name, dataset_name):
    return {
        "task": task_name,
        "tag": ["math_word_problems"],
        "dataset_path": "EleutherAI/hendrycks_math",
        "dataset_name": dataset_name,
        "dataset_kwargs": {"trust_remote_code": True},
        "training_split": "train",
        "test_split": "test",
        "process_docs": process_docs,
        "doc_to_text": doc_to_text,
        "doc_to_target": "{{answer if few_shot is undefined else solution}}",
        "process_results": process_results,
        "target_delimiter": " ",
        "fewshot_config": {"sampler": "first_n", "samples": list_fewshot_samples},
        "num_fewshot": 4,
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True},
            {"metric": "math_verify", "aggregation": "mean", "higher_is_better": True},
        ],
        "output_type": "generate_until",
        "generation_kwargs": {"until": ["Problem:"], "do_sample": False,
                              "temperature": 0.0},
        "repeats": 1,
        "metadata": {"version": 2.0},
    }


GROUP = {
    "group": "minerva_math",
    "task": [_leaf(*row) for row in SUBJECTS],
    "aggregate_metric_list": [
        {"metric": "exact_match", "weight_by_size": True},
        {"metric": "math_verify", "weight_by_size": True},
    ],
    "metadata": {"version": 1.0},
}
