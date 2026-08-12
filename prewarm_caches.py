"""Prewarm every cache a manifest needs, on a login node WITH network.

    python prewarm_caches.py manifest.jsonl

TACC compute nodes are offline: models, datasets and the HF code_eval
metric module must all be in $HF_HOME before the array starts, and cells
then run with HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1 (sbatch_cells.sh
sets both). This script downloads exactly what the manifest references:
model repos at their pinned revisions (registry names resolved through
models.MODELS; bare HF ids taken as-is) and each benchmark's datasets via
the same task construction run.py uses.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

from huggingface_hub import snapshot_download

from run import BENCH, load_manifest, _build_task_dict, Cell  # noqa: E402
from models import MODELS  # noqa: E402


def main(path):
    cells = load_manifest(path)

    repos = {}
    for c in cells:
        if c.model in MODELS:
            repos[(MODELS[c.model].repo, c.revision or "main")] = True
        else:
            repos[(c.model, c.revision or "main")] = True
    for repo, rev in sorted(repos):
        print(f">> model {repo} @ {rev}")
        snapshot_download(repo, revision=rev)

    for bench in sorted({c.benchmark for c in cells}):
        print(f">> datasets for {bench}")
        _build_task_dict(Cell("x", None, "ar", bench, (0, 1)))

    if any(BENCH[c.benchmark]["unsafe"] for c in cells):
        print(">> HF code_eval metric module")
        import evaluate
        evaluate.load("code_eval")

    print("prewarm complete -- the array can run offline")


if __name__ == "__main__":
    main(sys.argv[1])
