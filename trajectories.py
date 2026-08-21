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

DEFAULT_GEN = 128              # the C.8 demo canvas; the HE-164 statistic
                               # campaign runs --gen 256 (boss ask, 2026-08-18)
TAU = 0.9                      # the paper's featured threshold
POOL = [0, 3, 8, 31, 50, 56]   # the ENTIRE all-pass pool; no discretionary pick
STOPS = HUMANEVAL["generation_kwargs"]["until"]

MODELS = {                     # row order of the figure (comparators, control,
                               # then the family ascending -- ours-last)
    "llada-8b-base": None,
    "dream-7b-base": None,
    "dream-coder-7b-base": None,
    "coda-1.7b-base": None,                      # boss extension 2026-08-20
    "dqwen3-1.7b-base-v3": "step50000-swa",      # "
    "dqwen3.5-0.8b-base-v3": "step50000-swa",    # "
    "dqwen3.5-2b-base-v3": "step50000-swa",      # "
    "dqwen3.5-4b-base-v3": "step50000-swa",      # "
    "dqwen3.5-9b-base-v3": "step50000-swa",
}


def schemes_for(gen: int) -> dict:
    """The 2x3 factorial at a given canvas: block trio, then standard trio.
    The standard statics scale with the canvas (1 and 2 tokens/forward);
    the block trio is canvas-independent (gen/16 staircase treads)."""
    return {
        "block16-static-s16": dict(block_length=16, steps_per_block=16),
        "block16-static-s8":  dict(block_length=16, steps_per_block=8),
        "block16-tau0.9":     dict(block_length=16, steps_per_block=16,
                                   commit="dynamic", confidence_threshold=TAU),
        f"standard-static-s{gen}": dict(block_length=gen, steps_per_block=gen),
        f"standard-static-s{gen // 2}": dict(block_length=gen,
                                             steps_per_block=gen // 2),
        "standard-tau0.9":    dict(block_length=gen, steps_per_block=gen,
                                   commit="dynamic", confidence_threshold=TAU),
    }


def check_invariants(scheme: str, out, gen: int) -> None:
    """The design doc's correctness gates -- fail loudly, per generation."""
    cs = out.commit_step.tolist()
    stamped = [s for s in cs if s >= 0]
    # every executed forward commits >= 1 position (the progress floor)
    assert set(stamped) == set(range(out.n_forward)), \
        f"{scheme}: stamped steps != executed forwards"
    if scheme == f"standard-static-s{gen}":
        assert sorted(cs) == list(range(gen)), f"{scheme}: not a permutation"
    if scheme == f"standard-static-s{gen // 2}":
        assert all(stamped.count(t) == 2 for t in range(out.n_forward)), \
            f"{scheme}: expected exactly 2 commits per forward"
    if scheme.startswith("block16"):
        per_block = [[s for s in cs[b * 16:(b + 1) * 16] if s >= 0]
                     for b in range(gen // 16)]
        done = [b for b in per_block if b]
        assert all(max(a) < min(b) for a, b in zip(done, done[1:])), \
            f"{scheme}: staircase violated (blocks not step-monotone)"
    if "tau" in scheme:
        assert out.n_forward <= gen, f"{scheme}: forwards exceed static cap"


def _graded_prefix(text: str) -> str:
    """What gets graded: text cut at the earliest protocol stop (post-hoc;
    identical to early-stop truncation, since later blocks never alter
    earlier commitments), then at the first markdown fence -- at the 128
    canvas, models close correct solutions with fences + prose that the
    1024-tuned stop set never sees. Full text is always stored."""
    hits = [text.find(s) for s in STOPS if s and text.find(s) >= 0]
    if hits:
        text = text[:min(hits)]
    return text.split("\n```")[0].split("```")[0]


def run_model(name: str, out_dir: Path, no_stop: bool = False,
              gen: int = DEFAULT_GEN, problems=None,
              schemes_subset=None) -> None:
    import models
    from models.samplers import DecodeConfig, generate

    ds = datasets.load_dataset(HUMANEVAL["dataset_path"], split="test")
    problems = POOL if problems is None else problems
    # no_stop: block-mode only -- standard mode has no early exit, so its
    # trajectories are identical with or without stop strings.
    all_schemes = schemes_for(gen)
    schemes = ({k: v for k, v in all_schemes.items() if k.startswith("block16")}
               if no_stop else all_schemes)
    # --schemes subset (C.8 24-way parallelism, 2026-08-18): each subset
    # writes its OWN shard file so per-scheme jobs never share a writer;
    # shards concatenate into the canonical file at hand-back. Default
    # (None) keeps the original single-file behavior byte-identical.
    if schemes_subset is not None:
        unknown = set(schemes_subset) - set(all_schemes)
        if unknown:
            raise SystemExit(f"unknown scheme(s): {sorted(unknown)}")
        schemes = {k: v for k, v in schemes.items() if k in schemes_subset}
        if not schemes:
            print(f"[{name}] no applicable schemes for this variant; no-op")
            return
    shard = ("" if schemes_subset is None
             else "__" + "+".join(sorted(schemes_subset)))
    path = out_dir / (f"trajectories_{name}"
                      f"{'_nostop' if no_stop else ''}{shard}.jsonl")
    done = set()
    if path.exists():
        done = {(r["scheme"], r["task_id"])
                for r in map(json.loads, open(path))}
    m = models.load(name, revision=MODELS[name])
    with open(path, "a") as f:
        for doc_id in problems:                   # problem-major (Anton)
            doc = ds[doc_id]
            prompt_ids = m.encode(doc["prompt"])
            for scheme, kw in schemes.items():
                if (scheme, doc["task_id"]) in done:
                    continue
                t0 = time.time()
                cfg = DecodeConfig(gen_length=gen, temperature=0.0, **kw)
                out = generate(m, prompt_ids, cfg,
                               stop_strings=None if no_stop else STOPS)
                check_invariants(scheme, out, gen)
                cut = _graded_prefix(out.text)
                ref = doc["test"] + f"\ncheck({doc['entry_point']})"
                passed = bool(pass_at_k([ref], [[doc["prompt"] + cut]],
                                        k=[1])["pass@1"])
                f.write(json.dumps(dict(
                    model=name, revision=MODELS[name], scheme=scheme,
                    task_id=doc["task_id"], gen_length=gen,
                    commit_step=out.commit_step.tolist(),
                    forwards=out.n_forward, passed=passed,
                    early_stop=not no_stop,
                    fence_cut=(cut != out.text),
                    graded_prefix_chars=len(cut),
                    generation=out.text,
                    raw_generation=m.decode(out.gen_ids),
                    decode_config=kw,
                )) + "\n")
                f.flush()
                print(f"[{name}] {doc['task_id']} {scheme}: "
                      f"{out.n_forward} fwd, passed={passed} "
                      f"({time.time() - t0:.1f}s)", flush=True)


def render(out_dir: Path, no_stop: bool = False,
           gen: int = DEFAULT_GEN, problems=None) -> None:
    """Screening pages: one PNG per problem, rows=models, cols=schemes.
    no_stop pages union the *_nostop block records with the ORIGINAL
    standard records (whose trajectories are no-stop-identical).
    The PAPER figure is rendered manuscript-side; these are for eyeballs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(line) for p in out_dir.glob("trajectories_*.jsonl")
            for line in open(p)]
    by = {}
    for r in recs:
        es = r.get("early_stop", True)
        want = ((not es) if (no_stop and r["scheme"].startswith("block16"))
                else es)
        if want:
            by[(r["model"], r["scheme"], r["task_id"])] = r
    ds = datasets.load_dataset(HUMANEVAL["dataset_path"], split="test")
    for doc_id in (problems if problems is not None else POOL):
        tid = ds[doc_id]["task_id"]
        schemes = list(schemes_for(gen))
        fig, axes = plt.subplots(len(MODELS), len(schemes),
                                 figsize=(15, 10), squeeze=False)
        for i, model in enumerate(MODELS):
            for j, scheme in enumerate(schemes):
                ax = axes[i][j]
                r = by.get((model, scheme, tid))
                if r is None:
                    ax.set_axis_off()
                    continue
                cs = r["commit_step"]
                xs = [s for s in cs if s >= 0]
                ys = [p for p, s in enumerate(cs) if s >= 0]
                ax.plot([0, gen], [0, gen], color="0.85", lw=1, zorder=0)
                if scheme.startswith("block16"):
                    for y in range(16, gen, 16):
                        ax.axhline(y, color="0.92", lw=0.5, zorder=0)
                ax.scatter(xs, ys, s=4, color="#1b7a72", zorder=2)
                ax.set_xlim(0, gen); ax.set_ylim(0, gen)
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
        fig.suptitle(f"{tid} -- commit step (x) vs position (y)"
                     + (" [no early stop]" if no_stop else ""), fontsize=10)
        fig.tight_layout()
        png = out_dir / (f"screen_{tid.replace('/', '_')}"
                         + ("_nostop" if no_stop else "") + ".png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"wrote {png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="block-mode cells decode the whole canvas")
    ap.add_argument("--gen", type=int, default=DEFAULT_GEN,
                    help="canvas length (schemes scale with it)")
    ap.add_argument("--problems", default="pool",
                    help="'pool' (the 6), 'all' (HE-164), or comma doc ids")
    ap.add_argument("--schemes", default=None,
                    help="comma subset of scheme names (parallel lanes write "
                         "per-subset shard files); default = all, one file")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    probs = (POOL if a.problems == "pool" else
             list(range(164)) if a.problems == "all" else
             [int(x) for x in a.problems.split(",")])
    if a.render:
        render(a.out, no_stop=a.no_early_stop, gen=a.gen, problems=probs)
    elif a.model:
        run_model(a.model, a.out, no_stop=a.no_early_stop, gen=a.gen,
                  problems=probs,
                  schemes_subset=(a.schemes.split(",") if a.schemes else None))
    else:
        ap.error("need --model or --render")


if __name__ == "__main__":
    main()
