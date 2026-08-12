"""Read-only coalescer for _runs: prints one line per raw cell.

    python summarize.py            # everything
    python summarize.py gsm8k      # substring filter on any column

_runs holds only raw per-cell artifacts (see _runs/README.md); this script
is the aggregation layer, and it never writes anything back.
"""

import json
import sys
from pathlib import Path

RUNS = Path(__file__).parent / "_runs"

# headline task per benchmark (grid cell meta names the benchmark;
# lm-eval results are keyed by task)
HEAD_TASK = {"humaneval": "humaneval", "humaneval-plus": "humaneval_plus",
             "mbpp": "mbpp", "mbpp-plus": "mbpp_plus_full",
             "mbpp-fence": "mbpp_ticks",
             "gsm8k": "gsm8k_cot", "math": "minerva_math", "mmlu": "mmlu"}
# metric preference within the headline task's results dict
METRICS = ("math_verify,none", "exact_match,strict-match", "pass@1,create_test",
           "pass_at_1,none", "acc,none", "exact_match,none")


def headline(bench, results):
    r = results.get(HEAD_TASK.get(bench, bench))
    if not r:
        return "?", None
    for m in METRICS:
        if isinstance(r.get(m), (int, float)):
            return m.split(",")[0], r[m]
    for m, v in r.items():
        if isinstance(v, (int, float)) and "stderr" not in m:
            return m.split(",")[0], v
    return "?", None


def grid_rows():
    for jl in sorted(RUNS.glob("grid_v1/*/*.jsonl")):
        meta = summary = None
        with open(jl) as f:
            for line in f:
                d = json.loads(line)
                if d.get("kind") == "meta":
                    meta = d
                elif d.get("kind") == "summary":
                    summary = d
        if not meta:
            continue
        c = meta["cell"]
        model = f"{c['model'].replace('/', '_')}@{c['revision'] or 'main'}"
        shard = "s{}of{}".format(*c["shard"])
        if summary is None:
            yield (meta.get("launched_at", "-"), model, c["benchmark"],
                   c["decode"], shard,
                   "INCOMPLETE", "", str(jl.relative_to(RUNS)))
            continue
        metric, val = headline(c["benchmark"], summary["results"])
        yield (meta.get("launched_at", "-"), model, c["benchmark"],
               c["decode"], shard,
               metric, f"{100 * val:.2f}" if val is not None else "?",
               str(jl.relative_to(RUNS)))


def _passk(rj):
    ev = json.loads(rj.read_text())["eval"]
    n = len(ev)
    base = sum(r[0]["base_status"] == "pass" for r in ev.values())
    # official "+" semantics: base AND plus (plus_status alone overcounts
    # solutions that fail a base input but pass the extra tests)
    plus = sum(r[0]["base_status"] == "pass" and r[0]["plus_status"] == "pass"
               for r in ev.values())
    return n, f"{100 * base / n:.2f}/{100 * plus / n:.2f}"


def regrade_rows():
    # regrades/ is the FROZEN 2026-08-12 EvalPlus-graded record:
    # <stamp>__<model>__<bench>__from-<src>. Since the lm-eval-only ruling,
    # new "+" numbers come from grid cells, not regrades.
    for rj in sorted(RUNS.glob("regrades/*_eval_results.json")):
        n, v = _passk(rj)
        parts = rj.name.removesuffix("_eval_results.json").split("__")
        yield (parts[0], parts[1], parts[2], "regrade", f"n={n}",
               "pass@1 base/plus", v, str(rj.relative_to(RUNS)))


def main():
    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = [r for r in list(grid_rows()) + list(regrade_rows())
            if needle in "\t".join(r)]
    rows.sort(key=lambda r: (r[2], r[1], r[3], r[0]))
    if not rows:
        print(f"no cells match {needle!r} under {RUNS}")
        return
    hdr = ("launched", "model@revision", "benchmark", "decode", "shard",
           "metric", "value", "file")
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    for r in [hdr] + rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))


if __name__ == "__main__":
    main()
