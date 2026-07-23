"""EvalPlus driver — HumanEval+ / MBPP+ (denser-test code eval).

lm-eval's `humaneval`/`mbpp` grade against the thin original test suites; EvalPlus
regenerates ~80x/~35x more tests (the "+" sets), and the plus pass@1 is what code
papers report (Dream-Coder's 60.4 HE+, 61.6 MBPP+). This driver reuses the SAME
generation path as our lm-eval `generate_until` (block-diffusion decode, early-stop,
code-fence stop) so the base numbers stay consistent -- only the GRADING is denser.

    python -m dqeval.evalplus_driver --model dream-coder-7b-base --dataset humaneval
    python -m dqeval.evalplus_driver --model canonical --dataset humaneval   # grade-path self-test (no GPU)

`--model canonical` writes EvalPlus's own reference solutions as the samples, which
must score ~100% -- a way to validate the grading integration with no model/GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

# stop strings for code generation -- same set the harness uses, plus __main__ guard
CODE_STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(",
              "```", "<|endoftext|>", "<|im_end|>"]


def _truncate(text: str) -> str:
    cut = [text.find(s) for s in CODE_STOPS if s in text]
    return text[:min(cut)] if cut else text


def generate_completion(adapter, prompt: str, cfg) -> str:
    import torch  # noqa: F401
    from dqeval.samplers import unified
    ids = adapter.encode(prompt)
    out = unified.generate(adapter, ids, cfg, stop_strings=CODE_STOPS)
    return _truncate(out.text)


def load_problems(dataset: str):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def read_passk(results_path: str) -> dict:
    """Extract base and plus pass@1 from EvalPlus's eval_results.json."""
    with open(results_path) as f:
        res = json.load(f)
    evals = res.get("eval", res)
    base_pass = plus_pass = n = 0
    for tid, entries in evals.items():
        e = entries[0] if isinstance(entries, list) else entries
        n += 1
        base_pass += int(e.get("base_status", e.get("base", ["", ""])[0] if isinstance(e.get("base"), list) else "") == "pass"
                         or (isinstance(e.get("base"), list) and e["base"][0] == "pass"))
        plus_pass += int(e.get("plus_status", "") == "pass"
                         or (isinstance(e.get("plus"), list) and e["plus"][0] == "pass"))
    return {"n": n, "base": base_pass / n if n else 0, "plus": plus_pass / n if n else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dqeval registry name, or 'canonical' for the grade self-test")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--gen-length", type=int, default=512)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps-per-block", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", default=None, help="where to write samples.jsonl")
    a = ap.parse_args()

    problems = load_problems(a.dataset)
    items = list(problems.items())
    if a.limit:
        items = items[:a.limit]

    samples = []
    if a.model == "canonical":
        # grade-path self-test: EvalPlus's own reference solutions (no model/GPU)
        for tid, prob in items:
            samples.append({"task_id": tid,
                            "solution": prob["prompt"] + prob["canonical_solution"]})
    else:
        from dqeval.adapter import load
        from dqeval.config import DecodeConfig
        adapter = load(a.model, revision=a.revision)
        cfg = DecodeConfig(gen_length=a.gen_length, block_length=a.block_length,
                           steps_per_block=a.steps_per_block, mode="append", temperature=0.0)
        for i, (tid, prob) in enumerate(items):
            comp = generate_completion(adapter, prob["prompt"], cfg)
            samples.append({"task_id": tid, "solution": prob["prompt"] + comp})
            if (i + 1) % 20 == 0:
                print(f"  generated {i + 1}/{len(items)}", flush=True)

    spath = a.samples or os.path.join(tempfile.mkdtemp(), f"{a.model.replace('/', '_')}_{a.dataset}.jsonl")
    with open(spath, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} samples -> {spath}", flush=True)

    from evalplus.evaluate import evaluate
    evaluate(dataset=a.dataset, samples=spath, i_just_wanna_run=True)

    rpath = spath.replace(".jsonl", "_eval_results.json")
    if os.path.exists(rpath):
        pk = read_passk(rpath)
        print("\n===== EVALPLUS RESULTS =====")
        print(f"  {a.model} {a.dataset}: base pass@1 = {pk['base']*100:.2f}%  "
              f"plus pass@1 = {pk['plus']*100:.2f}%  (n={pk['n']})")
    else:
        print(f"NOTE: eval results not at {rpath}; check EvalPlus stdout above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
