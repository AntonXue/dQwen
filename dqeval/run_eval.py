"""Drive lm-eval on a dqeval model.

    python -m dqeval.run_eval --model llada-8b-base --tasks mmlu --limit 50
    python -m dqeval.run_eval --model dqwen3.5-2b-base --revision step30000-swa \
        --tasks arc_challenge --limit 100 --mc-num 128

Prereq: envs/setup_lmeval.sh (lm_eval 0.4.8, patched for transformers 5.x).
"""

from __future__ import annotations

import argparse
import json

import dqeval.harness  # noqa: F401  -- registers "dqeval" with lm-eval
from lm_eval import simple_evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dqeval registry name")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", required=True, help="comma-separated lm-eval tasks")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--mc-num", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--gen-length", type=int, default=1024)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps-per-block", type=int, default=32)
    ap.add_argument("--order", default="low_confidence",
                    choices=["low_confidence", "entropy", "topk_margin", "random", "sequential"],
                    help="unmasking order (Dream uses 'entropy')")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--allow-code", action="store_true",
                    help="permit humaneval/mbpp to execute generated code")
    ap.add_argument("--include-path", default=None,
                    help="extra task-yaml directory (dqeval/tasks holds our variants)")
    ap.add_argument("--out", default=None, help="write full results json here")
    ap.add_argument("--no-log-samples", action="store_false", dest="log_samples",
                    help="disable per-sample dumping (ON by default: prompts + raw and "
                         "filtered generations land in <out>.<task>.samples.jsonl, so a "
                         "formatting/extraction bug can be told from a real miss without "
                         "regenerating)")
    ap.set_defaults(log_samples=True)
    a = ap.parse_args()

    # --- foot-gun guard ---------------------------------------------------
    # MMLU (and friends) publish AT a shot count; lm-eval's task default may be
    # 0, so omitting --num-fewshot silently produces a 0-shot run that looks
    # normal but is not comparable to anything published. Shot counts follow
    # Dream's base-model table (arXiv:2508.15487). 0-shot entries are listed
    # explicitly so "not set" stays distinguishable from "deliberately zero".
    _PUBLISHED_SHOTS = {"mmlu": 5, "gsm8k_cot": 8, "mbpp": 3, "mbpp_ticks": 3, "humaneval": 0,
                        "arc_easy": 0, "arc_challenge": 0, "hellaswag": 0, "piqa": 0,
                        "race": 0, "race_sub": 0, "hellaswag_sub": 0, "winogrande": 5, "bbh_fewshot": 3, "minerva_math": 4}
    if a.num_fewshot is None:
        for t in a.tasks.split(","):
            want = _PUBLISHED_SHOTS.get(t.strip())
            if want:
                print(f"!! WARNING: --num-fewshot not set and {t!r} publishes at {want}-shot.\n"
                      f"   lm-eval will use the TASK default, which may be 0. Pass "
                      f"--num-fewshot {want} to match the published protocol.")

    print(f">> decode: gen_length={a.gen_length} block={a.block_length} "
          f"steps={a.steps_per_block} order={a.order} temp={a.temperature} "
          f"| tasks={a.tasks} num_fewshot={a.num_fewshot} limit={a.limit}")

    model_args = (
        f"pretrained={a.model},mc_num={a.mc_num},max_length={a.max_length},"
        f"gen_length={a.gen_length},block_length={a.block_length},"
        f"steps_per_block={a.steps_per_block},order={a.order},"
        f"temperature={a.temperature},top_p={a.top_p}"
    )
    if a.revision:
        model_args += f",revision={a.revision}"

    task_manager = None
    if a.include_path:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager(include_path=a.include_path)

    res = simple_evaluate(
        model="dqeval",
        model_args=model_args,
        task_manager=task_manager,
        tasks=a.tasks.split(","),
        limit=a.limit,
        num_fewshot=a.num_fewshot,
        bootstrap_iters=0,
        # keep every prompt + generation for post-hoc debugging (dumped below);
        # the aggregate json alone cannot distinguish a formatting/extraction
        # bug from a genuine miss.
        log_samples=a.log_samples,
        # code tasks (humaneval/mbpp) EXECUTE model-generated code; lm-eval refuses
        # unless this is set. Requires HF_ALLOW_CODE_EVAL=1 in the environment too.
        confirm_run_unsafe_code=a.allow_code,
    )

    print("\n===== RESULTS =====")
    print(f"model={a.model}" + (f" @{a.revision}" if a.revision else "")
          + f"  limit={a.limit}")
    for task, metrics in res["results"].items():
        nums = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
        print(f"  {task}: {nums}")
        for k, v in metrics.items():
            if k.endswith("_stderr") or "alias" in k:
                continue
        vers = res.get("versions", {}).get(task)
        cfg = res.get("configs", {}).get(task, {})
        print(f"    task_version={vers}  num_fewshot={cfg.get('num_fewshot')}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"results": res["results"], "versions": res.get("versions"),
                       "configs": res.get("configs"), "model": a.model,
                       "revision": a.revision, "limit": a.limit,
                       # decode provenance: a result json must be attributable
                       # to its exact decode settings.
                       "decode": {"gen_length": a.gen_length,
                                  "block_length": a.block_length,
                                  "steps_per_block": a.steps_per_block,
                                  "order": a.order, "temperature": a.temperature,
                                  "top_p": a.top_p, "mc_num": a.mc_num,
                                  "max_length": a.max_length},
                       "model_args": model_args}, f, indent=2, default=str)
        print(f"\nwrote {a.out}")

        # per-sample dump: one jsonl per task next to the results json. Holds the
        # prompt, the raw resp, the filtered/graded resp, the gold target, the source
        # doc, and the per-doc metric -- everything needed to see WHAT the model
        # emitted and HOW it was extracted/graded. This is the debugging goldmine.
        if a.log_samples and res.get("samples"):
            stem = a.out[:-5] if a.out.endswith(".json") else a.out
            for task, samples in res["samples"].items():
                spath = f"{stem}.{task}.samples.jsonl"
                with open(spath, "w") as sf:
                    for s in samples:
                        args = s.get("arguments") or []
                        a0 = args[0] if args else None
                        prompt = a0[0] if isinstance(a0, (list, tuple)) else a0
                        rec = {
                            "doc_id": s.get("doc_id"),
                            "task": task,
                            "prompt": prompt,
                            "resps": s.get("resps"),
                            "filtered_resps": s.get("filtered_resps"),
                            "target": s.get("target"),
                            "doc": s.get("doc"),
                            "metrics": {k: v for k, v in s.items()
                                        if isinstance(v, (int, float, bool))},
                        }
                        sf.write(json.dumps(rec, default=str) + "\n")
                print(f"wrote {len(samples)} samples -> {spath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
