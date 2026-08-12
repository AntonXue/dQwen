"""EvalPlus driver — HumanEval+ / MBPP+ (denser-test code eval).

lm-eval's `humaneval`/`mbpp` grade against the thin original test suites; EvalPlus
regenerates ~80x/~35x more tests (the "+" sets), and the plus pass@1 is what code
papers report (Dream-Coder's 60.4 HE+, 61.6 MBPP+). This driver reuses the SAME
generation path as our lm-eval `generate_until` (block-diffusion decode, early-stop,
code-fence stop) so the base numbers stay consistent -- only the GRADING is denser.

    python evalplus_driver.py --model dqwen3.5-2b-base-v3 --revision step50000-swa \
        --dataset mbpp --gen-length 512 --block-length 512 --steps-per-block 512
    python evalplus_driver.py --model Qwen/Qwen3.5-2B --ar --dataset mbpp --gen-length 512
    python evalplus_driver.py --model canonical --dataset humaneval   # grade-path self-test (no GPU)

Generation mode mirrors run.py's idempotency contract: output goes to the
_runs/evalplus/ store under a DETERMINISTIC name, completions append
incrementally (a killed run loses at most one doc), re-running the same
command resumes, and grading runs only once the problem set is complete --
so these queue on SLURM exactly like grid cells.

`--model canonical` writes EvalPlus's own reference solutions as the samples, which
must score ~100% -- a way to validate the grading integration with no model/GPU.

`--from FILE` regrades SAVED generations instead of running a model (CPU only):
accepts an lm-eval samples jsonl (doc_id + filtered_resps) or a grid_v1 cell
jsonl (kind-tagged records). HumanEval only: HE+ keeps the original prompts, so
saved completions regrade cleanly. MBPP+ does NOT regrade -- it is built on
MBPP-sanitized (378 problems, partially edited prompts) under EvalPlus's own
format, so it needs its own generation runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import os
import tempfile
import time

# HumanEval completions are a function BODY (the prompt holds the def), so any
# top-level construct ends them -- same set the harness uses, plus __main__ guard.
CODE_STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(",
              "```", "<|endoftext|>", "<|im_end|>"]
# MBPP+ completions are WHOLE functions (helpers + imports are legitimate), so
# the stops are EvalPlus's own base-model EOS set, not the HumanEval one.
MBPP_STOPS = ["<|endoftext|>", "<|endofmask|>", "</s>", "<|im_end|>",
              "\nif __name__", "\ndef main(", "\nprint(", '\n"""', "```"]


def stops_for(dataset: str) -> list:
    return CODE_STOPS if dataset == "humaneval" else MBPP_STOPS


def _truncate(text: str, stops) -> str:
    cut = [text.find(s) for s in stops if s in text]
    return text[:min(cut)] if cut else text


def generate_completion(adapter, prompt: str, cfg, stops) -> str:
    import torch  # noqa: F401
    from dqeval import sampler
    ids = adapter.encode(prompt)
    out = sampler.generate(adapter, ids, cfg, stop_strings=stops)
    return _truncate(out.text, stops)


def generate_completion_ar(model, tok, prompt: str, gen_length: int, stops) -> str:
    import torch
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=gen_length, do_sample=False,
                             stop_strings=stops, tokenizer=tok,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    return _truncate(text, stops)


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
    ap.add_argument("--model", default="from-file",
                    help="dqeval registry name, or 'canonical' for the grade self-test; unused with --from")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--gen-length", type=int, default=1024)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps-per-block", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", default=None, help="where to write samples.jsonl")
    ap.add_argument("--parallel", type=int, default=os.cpu_count(),
                    help="grading worker processes (default: all cores)")
    # Regrades are CPU-only and independent: for a batch of files, launch
    # one process per file concurrently (e.g. `... --from A & ... --from B &`
    # or xargs -P) — each run is ~25s; sequential batching is the slow way.
    ap.add_argument("--from", dest="from_file", default=None,
                    help="regrade saved generations (lm-eval samples jsonl or "
                         "grid_v1 cell jsonl) instead of running a model")
    ap.add_argument("--ar", action="store_true",
                    help="--model is a bare HF causal-LM id (AR counterpart "
                         "rows): greedy transformers generate, not the adapter")
    a = ap.parse_args()

    problems = load_problems(a.dataset)
    items = list(problems.items())
    if a.limit:
        items = items[:a.limit]

    root = os.path.dirname(os.path.abspath(__file__))

    # ---- output path first: generation mode uses a deterministic store name
    # (resume + grade-when-complete); regrades are stamped events.
    if a.from_file:
        # identity from the source: grid cells carry it in their meta record
        with open(a.from_file) as f:
            first = json.loads(f.readline())
        if first.get("kind") == "meta":
            c = first["cell"]
            tag = f"{c['model'].replace('/', '_')}@{c['revision'] or 'main'}"
            src = "grid_v1-" + c["decode"]
        else:
            tag = os.path.basename(a.from_file)
            for suf in (".samples.jsonl", ".jsonl", "." + a.dataset):
                tag = tag.removesuffix(suf)
            src = os.path.basename(os.path.dirname(a.from_file)) or "file"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        default = os.path.join(root, "_runs", "regrades",
                               f"{stamp}__{tag}__{a.dataset}-plus__from-{src}.jsonl")
    elif a.model == "canonical":
        tag = "canonical"
        default = os.path.join(tempfile.mkdtemp(), f"canonical_{a.dataset}.jsonl")
    else:
        tag = f"{a.model.replace('/', '_')}@{a.revision or 'main'}"
        decode = ("ar" if a.ar else
                  ("standard" if a.block_length >= a.gen_length
                   else f"block{a.block_length}") + f"-static-s{a.steps_per_block}")
        default = os.path.join(root, "_runs", "evalplus",
                               f"{tag}__{a.dataset}-plus__{decode}.jsonl")
    spath = a.samples or default
    if os.path.dirname(spath):
        os.makedirs(os.path.dirname(spath), exist_ok=True)

    # ---- produce samples
    samples = []
    if a.from_file:
        if a.dataset != "humaneval":
            raise SystemExit("--from is HumanEval-only: MBPP+ uses edited "
                             "sanitized prompts and needs its own generations.")
        comps = {}
        for line in open(a.from_file):
            d = json.loads(line)
            if "kind" in d:                      # grid_v1 cell file
                if d["kind"] == "sample":        # DLM sidecar (truncated text)
                    comps[d["doc_id"]] = d["final"]
                elif d["kind"] == "lm_eval_sample" and d.get("resp") is not None:
                    comps.setdefault(d["doc_id"], d["resp"])
            elif "filtered_resps" in d:          # lm-eval samples file
                r = d["filtered_resps"]
                while isinstance(r, list):
                    r = r[0] if r else ""
                comps[d["doc_id"]] = r
        for i, (tid, prob) in enumerate(items):
            if i in comps:
                samples.append({"task_id": tid,
                                "solution": prob["prompt"] + comps[i]})
        print(f"regrading {len(samples)}/{len(items)} saved completions "
              f"from {a.from_file}")
    elif a.model == "canonical":
        # grade-path self-test: EvalPlus's own reference solutions (no model/GPU)
        for tid, prob in items:
            samples.append({"task_id": tid,
                            "solution": prob["prompt"] + prob["canonical_solution"]})
    else:
        done = set()
        if os.path.exists(spath):
            with open(spath) as f:
                done = {json.loads(l)["task_id"] for l in f if l.strip()}
            if done:
                print(f"resuming: {len(done)}/{len(items)} already in {spath}")
        todo = [(t, p) for t, p in items if t not in done]
        if not todo:
            print(f"all {len(items)} generations already present in {spath}")
        else:
            # provenance sidecar: one line per launch (the samples file itself
            # must stay pure task_id/solution jsonl for EvalPlus)
            with open(spath + ".launches.jsonl", "a") as lf:
                lf.write(json.dumps(dict(
                    launched_at=time.strftime("%Y%m%d-%H%M%S"), model=a.model,
                    revision=a.revision, dataset=a.dataset, ar=a.ar,
                    gen_length=a.gen_length, block_length=a.block_length,
                    steps_per_block=a.steps_per_block, todo=len(todo))) + "\n")
            stops = stops_for(a.dataset)
            if a.ar:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from dqeval.grid import _pin_math_sdpa
                _pin_math_sdpa()   # same deterministic backend as GATE cells
                tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
                model = AutoModelForCausalLM.from_pretrained(
                    a.model, revision=a.revision, torch_dtype=torch.bfloat16,
                    device_map="cuda").eval()
                gen = lambda prob: generate_completion_ar(   # noqa: E731
                    model, tok, prob["prompt"], a.gen_length, stops)
            else:
                from dqeval.adapter import load
                from dqeval.sampler import DecodeConfig
                from dqeval.grid import _pin_math_sdpa
                _pin_math_sdpa()   # same deterministic backend as GATE cells
                adapter = load(a.model, revision=a.revision)
                cfg = DecodeConfig(gen_length=a.gen_length,
                                   block_length=a.block_length,
                                   steps_per_block=a.steps_per_block,
                                   temperature=0.0)
                gen = lambda prob: generate_completion(      # noqa: E731
                    adapter, prob["prompt"], cfg, stops)
            with open(spath, "a") as f:
                for i, (tid, prob) in enumerate(todo):
                    f.write(json.dumps({"task_id": tid,
                                        "solution": prob["prompt"] + gen(prob)})
                            + "\n")
                    f.flush()
                    if (i + 1) % 20 == 0:
                        print(f"  generated {i + 1}/{len(todo)} this launch "
                              f"({len(done) + i + 1}/{len(items)} total)",
                              flush=True)

    if samples:                       # from_file / canonical: whole-file write
        with open(spath, "w") as f:
            for smp in samples:
                f.write(json.dumps(smp) + "\n")
        print(f"wrote {len(samples)} samples -> {spath}", flush=True)

    if a.limit:
        print("--limit set: skipping grading (EvalPlus requires the full "
              "problem set); samples are saved above.")
        return 0
    with open(spath) as f:
        have = len({json.loads(l)["task_id"] for l in f if l.strip()})
    if have < len(problems):
        print(f"partial: {have}/{len(problems)} generations in {spath}; "
              "re-run the same command to resume. Grading deferred.")
        return 0

    from evalplus.evaluate import evaluate
    # evalplus defaults to cpu_count()//2 workers; this box has plenty
    evaluate(dataset=a.dataset, samples=spath, i_just_wanna_run=True,
             parallel=a.parallel)

    rpath = spath.replace(".jsonl", "_eval_results.json")
    if os.path.exists(rpath):
        pk = read_passk(rpath)
        print("\n===== EVALPLUS RESULTS =====")
        print(f"  {tag} {a.dataset}: base pass@1 = {pk['base']*100:.2f}%  "
              f"plus pass@1 = {pk['plus']*100:.2f}%  (n={pk['n']})")
    else:
        print(f"NOTE: eval results not at {rpath}; check EvalPlus stdout above.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
