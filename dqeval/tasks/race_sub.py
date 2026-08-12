"""RACE-high on a SEEDED 1000-doc sample of the 1,045-doc test split
(`race_sub`). Near-full (96%), so this is about ORDERING, not cost: RACE is
grouped by article, so a truncated slice over-weights whichever articles
come first. doc_to_* are IMPORTED from lm-eval's own preprocess_race rather
than copied, so upstream preprocessing cannot drift away from ours."""

from lm_eval.tasks.race.preprocess_race import (
    doc_to_choice,
    doc_to_target,
    doc_to_text,
)

from dqeval.tasks.hellaswag_sub import subsample

TASK = {
    "task": "race_sub",
    "dataset_path": "EleutherAI/race",
    "dataset_name": "high",
    "output_type": "multiple_choice",
    "test_split": "test",
    "process_docs": subsample,
    "doc_to_text": doc_to_text,
    "doc_to_target": doc_to_target,
    "doc_to_choice": doc_to_choice,
    "metric_list": [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
    ],
    "metadata": {"version": 2.0},
}
