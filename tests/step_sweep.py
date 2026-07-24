"""Step-budget sweep: accuracy vs compute (forwards) on HumanEval.

Tests the hypothesis that dQwen -- trained with standard DLM masking -- holds
accuracy as the step budget drops (few steps / high parallelism), while
decode-tuned models (Dream) fall off faster. Whoever has the flattest curve wins
the EFFICIENT frontier, which is the actual point of a diffusion LM.

Loads the model ONCE and sweeps steps_per_block; inline base-test grading (subset
ok, unlike evalplus.evaluate which needs the full set). Records pass@1 and the mean
#forwards per problem (the compute axis) at each step budget.

    python tests/step_sweep.py --model dqwen3.5-9b-base --steps 32,8,4,2 --limit 80
"""

from __future__ import annotations

import argparse
import contextlib
import io
import signal
import sys

import torch


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def grade(prompt, completion, test, entry_point, timeout=8):
    # model-generated code can contain infinite loops -> exec WITHOUT a timeout
    # hangs the whole sweep (this is why evalplus sandboxes). SIGALRM bounds it
    # (single-threaded main-thread script, so alarm is valid).
    prog = prompt + completion + "\n" + test + f"\ncheck({entry_point})"
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(prog, {})
        return True
    except BaseException:          # includes _Timeout
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


CODE_STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(",
              "```", "<|endoftext|>", "<|im_end|>"]


def _truncate(t):
    cut = [t.find(s) for s in CODE_STOPS if s in t]
    return t[:min(cut)] if cut else t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--steps", default="32,8,4,2", help="comma-separated steps_per_block")
    ap.add_argument("--gen-length", type=int, default=512)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--order", default="low_confidence")
    ap.add_argument("--limit", type=int, default=80)
    a = ap.parse_args()

    import logging
    logging.disable(logging.WARNING)
    from evalplus.data import get_human_eval_plus
    from dqeval.adapter import load
    from dqeval.config import DecodeConfig
    from dqeval.samplers import unified

    adapter = load(a.model, revision=a.revision)      # loaded ONCE
    probs = list(get_human_eval_plus().items())[:a.limit]
    step_vals = [int(s) for s in a.steps.split(",")]

    print(f"\nmodel={a.model}  HumanEval(base, n={len(probs)})  "
          f"gen={a.gen_length} block={a.block_length} mode={a.mode} order={a.order}")
    print(f"{'steps/blk':>9}{'pass@1':>9}{'mean_fwd':>10}")
    print("-" * 28)
    rows = []
    for spb in step_vals:
        cfg = DecodeConfig(gen_length=a.gen_length, block_length=a.block_length,
                           steps_per_block=spb, mode=a.mode, order=a.order, temperature=0.0)
        npass, fwds = 0, 0
        for tid, prob in probs:
            ids = adapter.encode(prob["prompt"])
            out = unified.generate(adapter, ids, cfg, stop_strings=CODE_STOPS)
            fwds += out.n_forward
            if grade(prob["prompt"], _truncate(out.text), prob["test"], prob["entry_point"]):
                npass += 1
        acc = npass / len(probs) * 100
        mf = fwds / len(probs)
        rows.append((spb, acc, mf))
        print(f"{spb:>9}{acc:>8.1f}%{mf:>10.1f}", flush=True)

    print("\nfrontier (pass@1 @ mean_fwd):",
          "  ".join(f"{a:.0f}%@{f:.0f}" for _, a, f in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
