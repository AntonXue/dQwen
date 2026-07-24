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
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps-per-block", type=int, default=32)
    ap.add_argument("--mode", choices=["append", "full", "window"], default="full",
                    help="full = LLaDA/Dream native (whole canvas visible, reveal "
                         "block-by-block); append = growing canvas (dQwen champion)")
    ap.add_argument("--allow-code", action="store_true",
                    help="permit humaneval/mbpp to execute generated code")
    ap.add_argument("--out", default=None, help="write full results json here")
    a = ap.parse_args()

    model_args = (
        f"pretrained={a.model},mc_num={a.mc_num},max_length={a.max_length},"
        f"gen_length={a.gen_length},block_length={a.block_length},"
        f"steps_per_block={a.steps_per_block},mode={a.mode}"
    )
    if a.revision:
        model_args += f",revision={a.revision}"

    res = simple_evaluate(
        model="dqeval",
        model_args=model_args,
        tasks=a.tasks.split(","),
        limit=a.limit,
        num_fewshot=a.num_fewshot,
        bootstrap_iters=0,
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
                       "revision": a.revision, "limit": a.limit}, f, indent=2, default=str)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
