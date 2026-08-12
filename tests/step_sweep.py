"""Step-budget sweep: accuracy vs compute (forwards) on HumanEval.

Tests the hypothesis that dQwen -- trained with standard DLM masking -- holds
accuracy as the step budget drops (few steps / high parallelism), while
decode-tuned models (Dream) fall off faster. Whoever has the flattest curve wins
the EFFICIENT frontier, which is the actual point of a diffusion LM.

Loads the model ONCE and sweeps steps_per_block; inline base-test grading (subset
ok, unlike evalplus.evaluate which needs the full set). Records pass@1 and the mean
#forwards per problem (the compute axis) at each step budget.

    python tests/step_sweep.py --model dqwen3.5-9b-base --steps 32,8,4,2 --limit 164

⚠ --limit defaults to 80 for quick probes, but first-80 HumanEval is ~19pp EASIER
than the full set — never quote a subset number. Pass --limit 164 for real cells.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
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


# HumanEval's task-level stops; the model-end set comes from dqeval.harness
# so this probe truncates under the same rules as the real harness.
HE_UNTIL = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--steps", default="32,8,4,2", help="comma-separated steps_per_block")
    ap.add_argument("--gen-length", type=int, default=512)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--order", default="low_confidence")
    ap.add_argument("--limit", type=int, default=80)
    # Parallelism mechanism. "static" = LLaDA's even split: commit exactly
    # gen_length/steps_per_block tokens per step (FIXED k, k = block/steps_per_block).
    # "dynamic" = commit every position whose confidence clears --thresholds, so k
    # varies per step and the model decides its own parallelism. The two answer
    # different questions: static asks "how well does it tolerate a forced k",
    # dynamic asks "can it TELL when parallel commits are safe".
    ap.add_argument("--commit", default="static", choices=["static", "dynamic"])
    ap.add_argument("--thresholds", default="0.9",
                    help="dynamic only: comma-separated confidence thresholds")
    ap.add_argument("--out", default=None, help="generations jsonl (default: _runs/)")
    a = ap.parse_args()

    import logging
    logging.disable(logging.WARNING)
    from evalplus.data import get_human_eval_plus
    from dqeval.adapter import load
    from dqeval.harness import eot_stops, truncate_at
    from dqeval.config import DecodeConfig
    from dqeval.samplers import unified

    adapter = load(a.model, revision=a.revision)      # loaded ONCE
    stops = HE_UNTIL + eot_stops(adapter)
    probs = list(get_human_eval_plus().items())[:a.limit]
    step_vals = [int(s) for s in a.steps.split(",")]
    tag = f"{a.model.replace('/', '_')}_{a.revision or 'main'}_{a.commit}"
    save_path = a.out or f"/home/ayx98/foo/dQwen/_runs/stepsweep/he_stepsweep_{tag}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sf = open(save_path, "w")

    print(f"\nmodel={a.model}  HumanEval(base, n={len(probs)})  "
          f"gen={a.gen_length} block={a.block_length} order={a.order}")
    print(f"{'steps/blk':>9}{'pass@1':>9}{'mean_fwd':>10}")
    print("-" * 28)
    rows = []
    # static -> sweep steps_per_block (forced k). dynamic -> sweep threshold at the
    # max step budget, letting the model choose k itself.
    if a.commit == "dynamic":
        cells = [(a.block_length, float(t)) for t in a.thresholds.split(",")]
    else:
        cells = [(spb, None) for spb in step_vals]
    for spb, thr in cells:
        cfg = DecodeConfig(gen_length=a.gen_length, block_length=a.block_length,
                           steps_per_block=spb, order=a.order, temperature=0.0,
                           commit=a.commit,
                           **({"confidence_threshold": thr} if thr is not None else {}))
        npass, fwds = 0, 0
        for tid, prob in probs:
            ids = adapter.encode(prob["prompt"])
            out = unified.generate(adapter, ids, cfg, stop_strings=stops)
            fwds += out.n_forward
            ext = truncate_at(out.text, stops)
            passed = grade(prob["prompt"], ext, prob["test"], prob["entry_point"])
            npass += passed
            sf.write(json.dumps({"steps_per_block": spb, "threshold": thr, "task_id": tid,
                                 "raw": out.text[:1500], "extracted": ext[:800],
                                 "passed": bool(passed), "n_forward": out.n_forward}) + "\n")
        sf.flush()
        acc = npass / len(probs) * 100
        mf = fwds / len(probs)
        rows.append((thr if thr is not None else spb, acc, mf))
        label = f"tau={thr:g}" if thr is not None else str(spb)
        print(f"{label:>9}{acc:>8.1f}%{mf:>10.1f}", flush=True)

    print("\nfrontier (pass@1 @ mean_fwd):",
          "  ".join(f"{a:.0f}%@{f:.0f}" for _, a, f in rows))
    sf.close(); print(f"saved generations -> {save_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
