"""The cell runner — dqeval's single entry point.

    from dqeval.cell_runner import Cell, run_cell
    run_cell(Cell("dqwen3.5-2b-base-v3", "step50000-swa",
                  "block32-tau0.8", "gsm8k", (2, 8)))

One Cell = (model, revision, decode, benchmark, shard) = one SLURM task =
one provenance-stamped output file, keyed by that tuple everywhere. Root
run.py is the CLI shim (and the only file a login node needs); this module
imports the full ML stack.

Two halves, top to bottom:

  THE LM PLUGIN — DQEvalLM registers with lm-eval as model type "dqeval":
  loglikelihood = MC-NELBO through `adapter.logits` (canonical surface; the
  shared-forward fast path for single-token options is bitwise-identical to
  per-option scoring), generate_until = the bs=1 block-diffusion sampler
  with stop strings. RecordingLM adds the per-doc sidecar (raw text,
  n_forward) that makes regrades CPU re-passes. MASK is NOT suppressed in
  loglikelihood (parity with LLaDA's reference); cfg_scale stays 0.

  THE GRID — benchmark configs come from dqeval/benchmarks/*.py (one python
  file per benchmark, frozen from pinned lm-eval, held to stock by
  tests/task_freeze_gate.py); shards are stripes (docs[k::n]), NEVER
  contiguous chunks (first-80 HumanEval is ~19pp easier than the full set);
  every cell dumps raw generations so regrades never need a GPU; a cell is
  complete iff its file ends with the summary record, which is what makes
  reruns idempotent and requeues free.

SLURM form: python -m dqeval.cell_runner manifest.jsonl $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from dqeval import samplers
from dqeval.models import load
from dqeval.samplers import DecodeConfig


# Model-end stop strings, shared by every generation driver — truncation
# rules move scores, so there is exactly one copy. The ``` fence catches
# models (Dream) that wrap completions in markdown, which task-level `until`
# strings miss; python source never contains ``` so it is safe to stop on.
EOT_BASE = ("```", "<|im_end|>", "<|endoftext|>")


def eot_stops(adapter):
    """EOT_BASE plus the model's own pad token, if it has one."""
    pad = adapter.tokenizer.pad_token
    return list(EOT_BASE) + ([pad] if pad else [])


def truncate_at(text, stops):
    """Cut `text` at the first occurrence of any stop string."""
    for st in stops:
        if st and st in text:
            text = text.split(st)[0]
    return text


@register_model("dqeval")
class DQEvalLM(LM):
    def __init__(self, pretrained, revision=None, *, batch_size=1,
                 max_length=2048, gen_length=1024, block_length=32,
                 steps_per_block=32, order="low_confidence",
                 top_p=1.0, top_k=0, mc_num=128, mc_bs=16,
                 temperature=0.0, cfg_scale=0.0,
                 commit="static", confidence_threshold=0.9, **kw):
        super().__init__()
        self.adapter = load(pretrained, revision=revision)
        self.max_length = int(max_length)
        self.mc_num = int(mc_num)
        self.mc_bs = int(mc_bs)
        # decode config for generate_until; loglikelihood tasks ignore it
        self.decode = DecodeConfig(
            gen_length=int(gen_length), block_length=int(block_length),
            steps_per_block=int(steps_per_block), order=order,
            top_p=float(top_p), top_k=int(top_k),
            temperature=float(temperature), cfg_scale=float(cfg_scale),
            commit=commit, confidence_threshold=float(confidence_threshold),
        )

    # -- required by lm-eval for request chunking --------------------------
    @property
    def eot_token_id(self):
        return self.adapter.pad_id

    @property
    def max_gen_toks(self):
        return self.decode.gen_length

    def tok_encode(self, s):
        return self.adapter.tokenizer(s, add_special_tokens=False)["input_ids"]

    def tok_decode(self, ids):
        return self.adapter.decode(torch.as_tensor(ids))

    def _prep(self, r):
        """(context_ids, continuation_ids) with the context length-budgeted."""
        ctx = self.tok_encode(r.args[0])
        cont = self.tok_encode(r.args[1])
        budget = self.max_length - len(cont)
        if budget < len(ctx):
            ctx = ctx[-budget:] if budget > 0 else ctx[:1]
        return ctx, cont

    # -- multiple choice: mmlu, arc, hellaswag, piqa, winogrande, ... ------
    @torch.inference_mode()
    def loglikelihood(self, requests):
        dev = self.adapter.device
        mask_id = self.adapter.mask_id
        out = [None] * len(requests)

        # FAST PATH. Multiple-choice tasks issue one request per (question, option)
        # with the SAME context and single-token continuations (mmlu/arc letters).
        # The masked canvas [ctx, MASK] is then identical across a question's options,
        # so one forward serves them all -- we just read a different vocab index per
        # option. This is the exact mc_num=1 MASK-slot marginal, provably identical to
        # scoring each option with its own forward, and still bs=1 (no batch-invariance
        # exposure). It removes the ~4x redundant forwards multiple-choice would spend.
        groups: dict[tuple, list[tuple[int, int]]] = {}
        for i, r in enumerate(requests):
            ctx, cont = self._prep(r)
            if len(cont) == 1:
                groups.setdefault(tuple(ctx), []).append((i, cont[0]))
            else:
                # multi-token continuation -> full MC-NELBO (no shared-forward reuse)
                p = torch.tensor(ctx, device=dev)
                a = torch.tensor(cont, device=dev)
                ll = samplers.mc_nelbo_loglikelihood(
                    self.adapter.logits, p, a,
                    mc_num=self.mc_num, batch_size=self.mc_bs, mask_id=mask_id,
                )
                out[i] = (ll, False)

        for ctx, items in groups.items():
            canvas = torch.tensor([*ctx, mask_id], device=dev)[None, :]
            # position-aligned logit at the single masked (answer) slot
            logits = self.adapter.logits(canvas)[0, len(ctx)]
            # Use the SAME op nelbo uses (F.cross_entropy on the native-dtype logits),
            # not log_softmax(logits.float()), so the shared-forward result is BITWISE
            # identical to scoring each option through mc_nelbo. ll = -CE.
            toks = torch.tensor([t for _, t in items], device=dev)
            ce = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0).expand(len(items), -1), toks, reduction="none")
            for (i, _), c in zip(items, ce):
                out[i] = (-c.item(), False)   # is_greedy=False, LLaDA convention

        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "rolling loglikelihood (perplexity tasks) is not implemented; "
            "not needed for mmlu/arc/gsm8k/humaneval"
        )

    # -- generative: gsm8k, minerva_math, humaneval, mbpp, bbh -------------
    @torch.inference_mode()
    def generate_until(self, requests):
        return [self._generate_one(r)[0] for r in requests]

    def _generate_one(self, r):
        """(final_text, GenOutput) for one request — factored out so
        RecordingLM below can record raw text + n_forward per request."""
        ctx = r.args[0]
        kw = r.args[1] if len(r.args) > 1 and isinstance(r.args[1], dict) else {}
        # stops = the task's own `until` strings + the model-end set (masked
        # DLMs don't self-terminate the way an AR model emits EOS).
        stops = list(kw.get("until", []) or []) + eot_stops(self.adapter)
        ids = self.adapter.encode(ctx)
        # stop_strings early-stops the decode itself (saves the forwards that
        # would fill the rest of the canvas); truncate_at is the post-hoc
        # backstop covering full/window mode, which decodes the whole canvas.
        gen = samplers.generate(self.adapter, ids, self.decode,
                               stop_strings=stops)
        return truncate_at(gen.text, stops), gen


class RecordingLM(DQEvalLM):
    """DQEvalLM that also keeps one record per generated request:
    (task, doc_id, raw text, truncated text, n_forward). run_cell below dumps
    these as the per-sample sidecar, which is what makes regrading
    (HumanEval+/MBPP+, format studies) a CPU re-pass instead of a GPU rerun.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def generate_until(self, requests):
        out = []
        for r in requests:
            text, gen = self._generate_one(r)
            self.records.append(dict(task=r.task_name, doc_id=r.doc_id,
                                     raw=gen.text, final=text,
                                     n_forward=gen.n_forward))
            out.append(text)
        return out


# --------------------------------------------------------------------------
# Benchmarks are generation-defining: mbpp vs mbpp-fence are distinct rows
# (different prompts -> different generations). The "+" rows are ordinary
# cells since the 2026-08-12 lm-eval-only ruling: humaneval-plus shares
# humaneval's prompts (denser tests; regenerating 164 docs is cheaper than
# a second grading framework -- grader A/B: 9/1640 verdicts differed,
# family cells exact), and mbpp-plus runs EvalPlus's sanitized problems
# under OUR scaffold via the local mbpp_plus_full task (stock mbpp_plus
# never executes the plus suite -- see dqeval/benchmarks/mbpp_plus.py).
# Shot counts follow Dream's base-model table.
# --------------------------------------------------------------------------
BENCH = {
    "humaneval":  dict(task="humaneval",    gen=512,  shots=0, shards=1, unsafe=True),
    "humaneval-plus": dict(task="humaneval_plus_sound", gen=512, shots=0, shards=1, unsafe=True),
    "mbpp":       dict(task="mbpp",         gen=512,  shots=3, shards=1, unsafe=True),
    "mbpp-plus":  dict(task="mbpp_plus_full", gen=512, shots=3, shards=1, unsafe=True),
    "mbpp-fence": dict(task="mbpp_ticks",   gen=512,  shots=3, shards=1, unsafe=True),
    "mbpp-plus-fence": dict(task="mbpp_plus_ticks", gen=512, shots=3, shards=1, unsafe=True),
    "gsm8k":      dict(task="gsm8k_cot",    gen=1024, shots=8, shards=8, unsafe=False),
    "math":       dict(task="minerva_math", gen=1024, shots=4, shards=16, unsafe=False),
    "mmlu":       dict(task="mmlu",         gen=None, shots=5, shards=1, unsafe=False),
}

TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)          # adaptive-commit thresholds
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

def _build_task_dict(cell: Cell):
    # configs come from dqeval.benchmarks (one file per benchmark, frozen
    # from pinned lm-eval; tests/task_freeze_gate.py holds them to stock)
    from lm_eval.tasks import TaskManager, get_task_dict
    from dqeval.benchmarks import task_config
    bench = BENCH[cell.benchmark]
    td = get_task_dict([task_config(bench["task"])], TaskManager())
    _restore_names(td)
    _stripe(td, *cell.shard)
    _set_shots(td, bench["shots"])
    return td


def _restore_names(td):
    # lm-eval's loader pops "task" out of a dict config and never writes it
    # back onto the task object (its python-class branch does the same repair
    # itself, self-described as "very scuffed"). Without this, unregistered
    # tasks run as [Task: None] and result aliasing crashes.
    for key, v in td.items():
        if isinstance(v, dict):
            _restore_names(v)
        elif v.config.task is None:
            v.config.task = key


def _leaf_tasks(td):
    """mmlu is a group: its value is a nested dict of subtasks. Flatten."""
    for v in td.values():
        if isinstance(v, dict):
            yield from _leaf_tasks(v)
        else:
            yield v


def _stripe(td, k, n):
    """docs[k::n] on every leaf task's test split. The meta record carries a
    doc-set fingerprint so the merge step can verify the shards' union."""
    if (k, n) == (0, 1):
        return
    for task in _leaf_tasks(td):
        split = task.config.test_split
        ds = task.dataset[split]
        task.dataset[split] = ds.select(range(k, len(ds), n))


def _set_shots(td, shots):
    # simple_evaluate applies num_fewshot via set_config; since we hand-build
    # the task_dict for striping, we must do the same (task defaults may be
    # 0-shot, which produces normal-looking but incomparable numbers).
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


def run_cell(cell: Cell, out_root="_runs/grid_v1"):
    """Run one cell; idempotent (a completed cell returns in seconds)."""
    import lm_eval

    out = out_path(cell, out_root)
    if is_complete(out):
        print(f"[grid] SKIP complete: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    bench = BENCH[cell.benchmark]
    if bench["unsafe"]:
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    if cell.decode != "ar":
        _pin_math_sdpa()   # deterministic attention backend, recorded in meta

    td = _build_task_dict(cell)
    fingerprint = _doc_fingerprint(td)
    lm = _build_lm(cell)

    # cell filenames are deterministic (identity = the tag, for idempotent
    # resume), so the launch time is recorded here instead
    launched_at = time.strftime("%Y%m%d-%H%M%S")
    t0 = time.time()
    res = lm_eval.evaluate(lm=lm, task_dict=td, log_samples=True,
                           bootstrap_iters=0, confirm_run_unsafe_code=True)
    wall = time.time() - t0

    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(dict(
            kind="meta", launched_at=launched_at,
            cell=asdict(cell), bench=bench, docs=fingerprint,
            decode_config=_decode_config_dict(lm),
            provenance=_provenance(cell, lm), wall_clock_s=round(wall, 1),
        )) + "\n")
        for rec in getattr(lm, "records", []):
            f.write(json.dumps(dict(kind="sample", **rec)) + "\n")
        for task_name, samples in (res.get("samples") or {}).items():
            for s in samples:
                resp = s.get("filtered_resps") or s.get("resps")
                while isinstance(resp, list):
                    resp = resp[0] if resp else None
                f.write(json.dumps(dict(
                    kind="lm_eval_sample", task=task_name,
                    doc_id=s.get("doc_id"),
                    # response text kept so regrades (HumanEval+/MBPP+) work
                    # for AR cells too, which have no sidecar
                    resp=resp,
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


def _pin_math_sdpa():
    """Force the math SDPA backend: flash/mem-efficient kernels are
    nondeterministic across shapes, and publication cells must be exactly
    reproducible. Recorded in the meta record's provenance."""
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


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
    import torch, transformers
    from importlib.metadata import version
    p["versions"] = dict(torch=torch.__version__,
                         transformers=transformers.__version__,
                         lm_eval=version("lm_eval"))
    # code-grading knobs live in dqeval/tasks/_grading.py; recorded so any
    # env-var override is visible from the cell's records
    from dqeval.benchmarks._grading import _CODE_TIMEOUT, _CODE_WORKERS
    p["code_eval"] = dict(timeout_s=_CODE_TIMEOUT, workers=_CODE_WORKERS)
    if cell.decode != "ar":
        p["sdpa_math_only"] = (torch.backends.cuda.math_sdp_enabled()
                               and not torch.backends.cuda.flash_sdp_enabled())
    return p


# --------------------------------------------------------------------------
# CLI: two forms.
#   manifest (SLURM):  python -m dqeval.cell_runner manifest.jsonl $SLURM_ARRAY_TASK_ID
#   direct (one cell): python -m dqeval.cell_runner MODEL REVISION DECODE BENCHMARK [K/N]
#     e.g. python -m dqeval.cell_runner dqwen3.5-2b-base-v3 step50000-swa \
#              block32-tau0.8 gsm8k 2/8
#     REVISION "main" or "-" means the default branch.
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
    if sys.argv[1].endswith(".jsonl"):
        run_cell(load_manifest(sys.argv[1])[int(sys.argv[2])])
    else:
        model, rev, decode, bench = sys.argv[1:5]
        shard = (tuple(int(x) for x in sys.argv[5].split("/"))
                 if len(sys.argv) > 5 else (0, 1))
        run_cell(Cell(model, None if rev in ("main", "-") else rev,
                      decode, bench, shard))
