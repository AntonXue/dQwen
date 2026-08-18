"""C.7 decode-trajectory driver -- commit-order records for the appendix demo.

Design + receipts: _claude/20260817-202121-c7-experiment-setup-final-design.
Manuscript request: VistaCoder_Technical_Report/_claude/20260817-151600.

Runs OUTSIDE the cell grid on purpose: records are per-generation commit
trajectories (canvas 128, block 16), not store cells, and the JSONL format is
the manuscript's. Prompts/stops/grading reuse the HumanEval spec verbatim.

    # generation (one process per GPU; problem-major within each process)
    CUDA_VISIBLE_DEVICES=1 python trajectories.py --model dream-7b-base
    # screening pages (CPU, after all models finish)
    python trajectories.py --render

Idempotent: (model, scheme, task) triples already in the model's JSONL are
skipped, so requeues resume.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

import datasets
import torch

from benchmark_specs import HUMANEVAL, pass_at_k

OUT = Path(__file__).parent / "_runs" / "20260817-202121-c7-decode-trajectories"

GEN = 128                      # demo canvas (93.3% of canonical solutions fit)
TAU = 0.9                      # the paper's featured threshold
POOL = [0, 3, 8, 31, 50, 56]   # the ENTIRE all-pass pool; no discretionary pick
STOPS = HUMANEVAL["generation_kwargs"]["until"]

MODELS = {                     # row order of the figure
    "llada-8b-base": None,
    "dream-7b-base": None,
    "dream-coder-7b-base": None,
    "dqwen3.5-9b-base-v3": "step50000-swa",
}
SCHEMES = {                    # column order: block trio, then standard trio
    "block16-static-s16":   dict(block_length=16, steps_per_block=16),
    "block16-static-s8":    dict(block_length=16, steps_per_block=8),
    "block16-tau0.9":       dict(block_length=16, steps_per_block=16,
                                 commit="dynamic", confidence_threshold=TAU),
    "standard-static-s128": dict(block_length=GEN, steps_per_block=GEN),
    "standard-static-s64":  dict(block_length=GEN, steps_per_block=64),
    "standard-tau0.9":      dict(block_length=GEN, steps_per_block=GEN,
                                 commit="dynamic", confidence_threshold=TAU),
}


def check_invariants(scheme: str, out) -> None:
    """The design doc's correctness gates -- fail loudly, per generation."""
    cs = out.commit_step.tolist()
    stamped = [s for s in cs if s >= 0]
    # every executed forward commits >= 1 position (the progress floor)
    assert set(stamped) == set(range(out.n_forward)), \
        f"{scheme}: stamped steps != executed forwards"
    if scheme == "standard-static-s128":
        assert sorted(cs) == list(range(GEN)), f"{scheme}: not a permutation"
    if scheme == "standard-static-s64":
        assert all(stamped.count(t) == 2 for t in range(out.n_forward)), \
            f"{scheme}: expected exactly 2 commits per forward"
    if scheme.startswith("block16"):
        per_block = [[s for s in cs[b * 16:(b + 1) * 16] if s >= 0]
                     for b in range(GEN // 16)]
        done = [b for b in per_block if b]
        assert all(max(a) < min(b) for a, b in zip(done, done[1:])), \
            f"{scheme}: staircase violated (blocks not step-monotone)"
    if "tau" in scheme:
        cap = {"block16-tau0.9": 128, "standard-tau0.9": 128}[scheme]
        assert out.n_forward <= cap, f"{scheme}: forwards exceed static cap"


def run_model(name: str, out_dir: Path) -> None:
    import models
    from models.samplers import DecodeConfig, generate

    ds = datasets.load_dataset(HUMANEVAL["dataset_path"], split="test")
    path = out_dir / f"trajectories_{name}.jsonl"
    done = set()
    if path.exists():
        done = {(r["scheme"], r["task_id"])
                for r in map(json.loads, open(path))}
    m = models.load(name, revision=MODELS[name])
    with open(path, "a") as f:
        for doc_id in POOL:                       # problem-major (Anton)
            doc = ds[doc_id]
            prompt_ids = m.encode(doc["prompt"])
            for scheme, kw in SCHEMES.items():
                if (scheme, doc["task_id"]) in done:
                    continue
                t0 = time.time()
                cfg = DecodeConfig(gen_length=GEN, temperature=0.0, **kw)
                out = generate(m, prompt_ids, cfg, stop_strings=STOPS)
                check_invariants(scheme, out)
                # Grading-time sanitization ONLY (trajectories untouched):
                # at the 128 canvas, models close the solution with a
                # markdown fence + prose that the protocol's stop set
                # (tuned on the 1024 canvas) never sees; a fence is never
                # part of a valid completion, so grade up to it. The full
                # text is stored regardless.
                cut = out.text.split("\n```")[0].split("```")[0]
                ref = doc["test"] + f"\ncheck({doc['entry_point']})"
                passed = bool(pass_at_k([ref], [[doc["prompt"] + cut]],
                                        k=[1])["pass@1"])
                f.write(json.dumps(dict(
                    model=name, revision=MODELS[name], scheme=scheme,
                    task_id=doc["task_id"], gen_length=GEN,
                    commit_step=out.commit_step.tolist(),
                    forwards=out.n_forward, passed=passed,
                    fence_cut=(cut != out.text),
                    generation=out.text,
                    raw_generation=m.decode(out.gen_ids),
                    decode_config=kw,
                )) + "\n")
                f.flush()
                print(f"[{name}] {doc['task_id']} {scheme}: "
                      f"{out.n_forward} fwd, passed={passed} "
                      f"({time.time() - t0:.1f}s)", flush=True)


def render(out_dir: Path) -> None:
    """Screening pages: one PNG per problem, rows=models, cols=schemes.
    The PAPER figure is rendered manuscript-side; these are for eyeballs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(line) for p in out_dir.glob("trajectories_*.jsonl")
            for line in open(p)]
    by = {(r["model"], r["scheme"], r["task_id"]): r for r in recs}
    ds = datasets.load_dataset(HUMANEVAL["dataset_path"], split="test")
    for doc_id in POOL:
        tid = ds[doc_id]["task_id"]
        fig, axes = plt.subplots(len(MODELS), len(SCHEMES),
                                 figsize=(15, 10), squeeze=False)
        for i, model in enumerate(MODELS):
            for j, scheme in enumerate(SCHEMES):
                ax = axes[i][j]
                r = by.get((model, scheme, tid))
                if r is None:
                    ax.set_axis_off()
                    continue
                cs = r["commit_step"]
                xs = [s for s in cs if s >= 0]
                ys = [p for p, s in enumerate(cs) if s >= 0]
                ax.plot([0, GEN], [0, GEN], color="0.85", lw=1, zorder=0)
                if scheme.startswith("block16"):
                    for y in range(16, GEN, 16):
                        ax.axhline(y, color="0.92", lw=0.5, zorder=0)
                ax.scatter(xs, ys, s=4, color="#1b7a72", zorder=2)
                ax.set_xlim(0, GEN); ax.set_ylim(0, GEN)
                ax.set_xticks([]); ax.set_yticks([])
                ax.annotate(f"{r['forwards']} fwd"
                            + ("" if r["passed"] else "  ✗"),
                            (0.97, 0.03), xycoords="axes fraction",
                            ha="right", fontsize=7,
                            color="0.4" if r["passed"] else "#a33")
                if i == 0:
                    ax.set_title(scheme, fontsize=8)
                if j == 0:
                    ax.set_ylabel(model.replace("-base", ""), fontsize=8)
        fig.suptitle(f"{tid} -- commit step (x) vs position (y)", fontsize=10)
        fig.tight_layout()
        png = out_dir / f"screen_{tid.replace('/', '_')}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"wrote {png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.render:
        render(a.out)
    elif a.model:
        run_model(a.model, a.out)
    else:
        ap.error("need --model or --render")


if __name__ == "__main__":
    main()
