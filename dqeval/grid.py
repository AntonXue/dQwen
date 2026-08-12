"""The grid layer: one Cell = one SLURM task = one output file.

Design settled with Anton 2026-08-11/12 (spec discussion in the manuscript
repo's chat; the VistaCoder _claude/20260811-190000 eval request holds the
grid itself). Principles:

  - CELL GRAMMAR: (model, revision, decode, benchmark, shard) is the primary
    key everywhere — filenames, meta records, the master table.
  - LM-EVAL-FIRST: prompts, few-shot, extraction, metrics all come from the
    PINNED lm-eval 0.4.8 (the reproduces-published-numbers claim). This module
    only doctors the task's docs (striped shards) and records a sidecar.
  - GENERATE THEN GRADE: every cell dumps raw generations + n_forward per
    sample. Regrades (HumanEval+/MBPP+ etc.) are CPU re-passes over the
    sidecar, never new GPU runs.
  - SHARDS ARE STRIPES, docs[k::n] — contiguous chunks are biased samples
    (the n=80 HumanEval episode: first-80 is ~19pp easier than the full set).
    Per-shard aggregate metrics are therefore MEANINGLESS; the merge step
    recomputes from per-sample records and refuses partial grids.
  - RERUN EVERYTHING: no legacy numbers are merged in; old results become a
    regression check against the new grid, not part of it.

Usage (library):
    from dqeval.grid import Cell, run_cell
    run_cell(Cell("dqwen3.5-2b-base-v3", "step50000-swa",
                  "block32-tau0.8", "gsm8k", (2, 8)))

Usage (SLURM shim; the cluster side owns sbatch specifics):
    python -m dqeval.grid manifest.jsonl $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# benchmarks: generation-defining constants. mbpp vs mbpp_ticks are DISTINCT
# benchmarks (different prompts -> different generations); HumanEval+ / MBPP+
# are GRADERS over saved generations and deliberately absent here.
# Shot counts follow run_eval._PUBLISHED_SHOTS (Dream's base-model table).
# NB stock mbpp is the cross-model headline ([BEGIN]/[DONE]); the fence
# variant exists as its own benchmark for the format-sensitivity study.
# --------------------------------------------------------------------------
BENCH = {
    "humaneval":  dict(task="humaneval",    gen=512,  shots=0, shards=1, unsafe=True),
    "mbpp":       dict(task="mbpp",         gen=512,  shots=3, shards=1, unsafe=True),
    "mbpp-fence": dict(task="mbpp_ticks",   gen=512,  shots=3, shards=1, unsafe=True),
    "gsm8k":      dict(task="gsm8k_cot",    gen=1024, shots=8, shards=8, unsafe=False),
    "math":       dict(task="minerva_math", gen=1024, shots=4, shards=16, unsafe=False),
    "mmlu":       dict(task="mmlu",         gen=None, shots=5, shards=1, unsafe=False),
}

TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)          # Anton 2026-08-12: simple, uniform
BLOCK_STATIC = (32, 16, 8, 4, 2)              # steps per 32-block
STANDARD_STATIC = (512, 256, 128, 64, 32, 16) # total steps over the whole canvas


@dataclass(frozen=True)
class Cell:
    model: str            # dqeval registry name; or a bare HF id for decode="ar"
    revision: str | None
    decode: str           # "ar" | "mc-nelbo" | "<canvas>-static-s<K>" | "<canvas>-tau<T>"
    benchmark: str        # key into BENCH
    shard: tuple = (0, 1)

    def tag(self) -> str:
        k, n = self.shard
        return (f"{self.model.replace('/', '_')}@{self.revision or 'main'}"
                f"__{self.benchmark}__{self.decode}__s{k}of{n}")


def parse_decode(decode: str, gen_length: int) -> dict:
    """Scheme string -> DQEvalLM kwargs. 'standard' = one block spanning the
    canvas, which is exactly unconstrained any-order decoding."""
    canvas, _, spec = decode.partition("-")
    block = {"block32": 32, "standard": gen_length}[canvas]
    if spec.startswith("tau"):
        return dict(block_length=block, commit="dynamic",
                    confidence_threshold=float(spec[3:]),
                    steps_per_block=block)   # tau decides commits; budget = ceiling
    if spec.startswith("static-s"):
        steps = int(spec[len("static-s"):])
        if steps > block:
            raise ValueError(f"{decode}: steps {steps} > block {block}")
        return dict(block_length=block, commit="static", steps_per_block=steps)
    raise ValueError(f"unparseable decode scheme: {decode!r}")


# --------------------------------------------------------------------------
# lm-eval plumbing
# --------------------------------------------------------------------------

def _tasks_dir():
    return str(Path(__file__).parent / "tasks")


def _build_task_dict(cell: Cell):
    from lm_eval.tasks import TaskManager, get_task_dict
    bench = BENCH[cell.benchmark]
    td = get_task_dict([bench["task"]], TaskManager(include_path=_tasks_dir()))
    _stripe(td, *cell.shard)
    _set_shots(td, bench["shots"])
    return td


def _leaf_tasks(td):
    """mmlu is a group: its value is a nested dict of subtasks. Flatten."""
    for v in td.values():
        if isinstance(v, dict):
            yield from _leaf_tasks(v)
        else:
            yield v


def _stripe(td, k, n):
    """docs[k::n] on every leaf task's test split. Content hashes go into the
    meta record so the merge step can verify union = full set, no overlap."""
    if (k, n) == (0, 1):
        return
    for task in _leaf_tasks(td):
        split = task.config.test_split
        ds = task.dataset[split]
        task.dataset[split] = ds.select(range(k, len(ds), n))


def _set_shots(td, shots):
    # simple_evaluate applies num_fewshot via set_config; since we hand-build
    # the task_dict for striping, we must do the same (task DEFAULTS may be
    # 0-shot — the exact slip that cost a round on 2026-08-05).
    for task in _leaf_tasks(td):
        if shots is not None:
            task.set_config(key="num_fewshot", value=shots)


def _doc_fingerprint(td):
    h = hashlib.sha256()
    n_docs = 0
    for task in _leaf_tasks(td):
        for doc in task.test_docs():
            h.update(json.dumps(doc, sort_keys=True, default=str).encode())
            n_docs += 1
    return {"n_docs": n_docs, "sha256": h.hexdigest()}


# --------------------------------------------------------------------------
# model construction (imports deferred: manifest work on a login node must
# not require torch/lm_eval)
# --------------------------------------------------------------------------

def _build_lm(cell: Cell):
    bench = BENCH[cell.benchmark]
    if cell.decode == "ar":
        from lm_eval.models.huggingface import HFLM
        return HFLM(pretrained=cell.model, revision=cell.revision or "main",
                    dtype="bfloat16", batch_size=16, trust_remote_code=True)
    from dqeval.harness import RecordingLM
    if cell.decode == "mc-nelbo":
        if bench["gen"] is not None:
            raise ValueError("mc-nelbo decode is only for MC benchmarks (mmlu)")
        return RecordingLM(cell.model, revision=cell.revision)
    if bench["gen"] is None:
        raise ValueError(f"{cell.benchmark} is MC; use decode='mc-nelbo'")
    return RecordingLM(cell.model, revision=cell.revision,
                       gen_length=bench["gen"],
                       **parse_decode(cell.decode, bench["gen"]))


# --------------------------------------------------------------------------
# the one entrypoint
# --------------------------------------------------------------------------

def out_path(cell: Cell, out_root) -> Path:
    return Path(out_root) / cell.benchmark / (cell.tag() + ".jsonl")


def is_complete(path: Path) -> bool:
    """Complete iff the last record is the summary sentinel."""
    if not path.exists():
        return False
    try:
        last = path.read_text().rstrip("\n").rsplit("\n", 1)[-1]
        return json.loads(last).get("kind") == "summary"
    except Exception:
        return False


def run_cell(cell: Cell, out_root="_runs/grid_v1", lm=None):
    """Run one cell; idempotent. Pass a pre-built `lm` to amortize one model
    load over many cells (sweeps) — caller must ensure it matches the cell."""
    import lm_eval

    out = out_path(cell, out_root)
    if is_complete(out):
        print(f"[grid] SKIP complete: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    bench = BENCH[cell.benchmark]
    if bench["unsafe"]:
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    td = _build_task_dict(cell)
    fingerprint = _doc_fingerprint(td)
    own_lm = lm is None
    if own_lm:
        lm = _build_lm(cell)

    # NB low-level evaluate() has no confirm_run_unsafe_code gate (that check
    # lives in simple_evaluate); the code-exec metric itself only needs
    # HF_ALLOW_CODE_EVAL=1, set above.
    t0 = time.time()
    res = lm_eval.evaluate(lm=lm, task_dict=td, log_samples=True,
                           bootstrap_iters=0)
    wall = time.time() - t0

    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(dict(
            kind="meta", cell=asdict(cell), bench=bench, docs=fingerprint,
            decode_config=_decode_config_dict(lm),
            provenance=_provenance(cell, lm), wall_clock_s=round(wall, 1),
        )) + "\n")
        for rec in getattr(lm, "records", []):
            f.write(json.dumps(dict(kind="sample", **rec)) + "\n")
        for task_name, samples in (res.get("samples") or {}).items():
            for s in samples:
                f.write(json.dumps(dict(
                    kind="lm_eval_sample", task=task_name,
                    doc_id=s.get("doc_id"),
                    metrics={k: v for k, v in s.items()
                             if isinstance(v, (int, float, bool))},
                ), default=str) + "\n")
        f.write(json.dumps(dict(
            kind="summary", cell=asdict(cell),
            results=res["results"], wall_clock_s=round(wall, 1),
        ), default=str) + "\n")
    tmp.rename(out)
    print(f"[grid] DONE {out}  ({wall:.0f}s)")
    return out


def _decode_config_dict(lm):
    d = getattr(lm, "decode", None)
    return {k: getattr(d, k) for k in vars(d)} if d is not None else {"decode": "ar"}


def _provenance(cell: Cell, lm):
    """Structured provenance — audit_eval_provenance.py reads fields, not logs."""
    p = dict(model=cell.model, revision=cell.revision)
    adapter = getattr(lm, "adapter", None)
    if adapter is not None:
        p["hf_repo"] = getattr(getattr(adapter, "spec", None), "repo", None)
        cfg = getattr(getattr(adapter, "model", None), "config", None)
        p["hf_name_or_path"] = getattr(cfg, "_name_or_path", None)
    else:  # AR path: the HFLM holds the model directly
        p["hf_repo"] = cell.model
    import torch, transformers, lm_eval as le
    from importlib.metadata import version
    p["versions"] = dict(torch=torch.__version__,
                         transformers=transformers.__version__,
                         lm_eval=version("lm_eval"))
    return p


# --------------------------------------------------------------------------
# SLURM shim: python -m dqeval.grid manifest.jsonl <index>
# --------------------------------------------------------------------------

def load_manifest(path):
    cells = []
    for line in open(path):
        d = json.loads(line)
        d["shard"] = tuple(d.get("shard", (0, 1)))
        cells.append(Cell(**d))
    return cells


if __name__ == "__main__":
    import sys
    cells = load_manifest(sys.argv[1])
    idx = int(sys.argv[2])
    run_cell(cells[idx])
