"""HellaSwag on a SEEDED 1000-doc sample of the 10,042-doc validation split
(`hellaswag_sub`). Full-set cost is ~10h per 0.8B arm through multi-token
MC-NELBO; n=1000 gives stderr ~1.5pp, well inside what a retention curve
needs. Preprocessing is stock lm-eval's, then a pinned-seed shuffle-sample —
NEVER first-N (lm-eval's `limit` is a biased slice; this program rebuilt
HE-50R after a naive first-50 ran ~10pp high)."""

from lm_eval.tasks.hellaswag.utils import process_docs as _stock_process

SUBSAMPLE_SEED = 1234
SUBSAMPLE_N = 1000


def subsample(docs, n=SUBSAMPLE_N, seed=SUBSAMPLE_SEED):
    """Pinned-seed uniform sample: identical across every model and rerun."""
    if len(docs) <= n:
        return docs
    return docs.shuffle(seed=seed).select(range(n))


def process_docs(dataset):
    return subsample(_stock_process(dataset))


TASK = {
    "task": "hellaswag_sub",
    "tag": ["multiple_choice"],
    "dataset_path": "Rowan/hellaswag",
    "output_type": "multiple_choice",
    "training_split": "train",
    "validation_split": "validation",
    "process_docs": process_docs,
    "doc_to_text": "{{query}}",
    "doc_to_target": "{{label}}",
    "doc_to_choice": "choices",
    "metric_list": [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
        {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
    ],
    "metadata": {"version": 1.0},
}
