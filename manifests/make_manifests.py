"""The campaign row lists -- three manifests, one sbatch launch each.

Composition is fixed by the paper-side docs (VistaCoder_Technical_Report/
_claude/) plus Anton's 2026-08-13 rulings; this file is the executable form:

  part1  AR-twin baselines, rerun on cluster hardware (GATE restamp of the
         2026-08-12 workstation batch). Twins-only rule from EVAL-REQUEST
         20260807-170744 SB; bare repos, NEVER -Base (FINDING 20260811-171500:
         bare = post-trained = the conversion source). AR cells run UNSHARDED
         (batch-composition policy) and take bare HF ids.
  part2  The big table: rows from HANDOFF 20260811-134555 S4.1 + the
         control@50k obligation (20260811-203500). One decode column,
         block32-static-s32 (the champion protocol); mmlu via mc-nelbo;
         plus the 6-row standard-static HumanEval ceiling -- full-canvas
         stays a stretch tier, per Anton 2026-08-13.
  part3  The acceleration grid, EVAL-REQUEST 20260811-190000 T1-T3 columns
         (static steps 32..2, tau 0.5..0.95), on HumanEval AND GSM8K
         (Anton 2026-08-13 extended the HE-only spec). Cells part2 already
         carries are dropped here so the parts can queue concurrently.

Fence benchmarks stay DLM-only: the MBPP format study is about our models'
training data, and no doc requests AR fence cells.

Shard rule (Anton 2026-08-13): no DLM cell generates more than ~200 docs.
mmlu is exempt (mc-nelbo generates nothing -- single-token MC scores via the
shared-forward path, ~1 forward/doc); AR cells stay UNSHARDED regardless
(bs=16 batch composition moves numbers ~0.23pp between striped and whole).

Run: python manifests/make_manifests.py   (rewrites the .jsonl siblings)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# benchmark -> stripe count for DLM generation cells (docs/shard stays <=200:
# he 164 whole, mbpp 499/4~125, mbpp+ 378/2=189, gsm8k 1319/8~165, math
# 5000/32~156)
DLM_SHARDS = {"humaneval": 1, "humaneval-plus": 1, "mbpp": 4, "mbpp-plus": 2,
              "mbpp-fence": 4, "mbpp-plus-fence": 2, "gsm8k": 8, "math": 32}

CODE_BENCHES = ["humaneval", "humaneval-plus", "mbpp", "mbpp-plus",
                "mbpp-fence", "mbpp-plus-fence"]
AR_BENCHES = ["humaneval", "humaneval-plus", "mbpp", "mbpp-plus",
              "gsm8k", "math", "mmlu"]

AR_TWINS = [
    "Qwen/Qwen3.5-0.8B",      # dqwen3.5-0.8b twin
    "Qwen/Qwen3.5-2B",        # dqwen3.5-2b twin
    "Qwen/Qwen3.5-4B",        # dqwen3.5-4b twin
    "Qwen/Qwen3.5-9B",        # dqwen3.5-9b twin
    "Qwen/Qwen3-1.7B",        # control twin AND CoDA twin (bare, eos 151645)
    "Qwen/Qwen3-0.6B",        # dqwen3-0.6b twin
    "Qwen/Qwen2.5-7B",        # Dream-v0 twin
    "Qwen/Qwen2.5-Coder-7B",  # Dream-Coder twin
]

BIG_ROWS = [
    ("dqwen3.5-0.8b-base-v3", "step50000-swa"),
    ("dqwen3.5-2b-base-v3",   "step50000-swa"),
    ("dqwen3.5-4b-base-v3",   "step50000-swa"),
    ("dqwen3.5-9b-base-v3",   "step50000-swa"),
    ("dqwen3.5-2b-base-v3",   "step25000-swa"),  # width-matched pair, hybrid leg
    ("dqwen3-1.7b-base-v3",   "step25000-swa"),  # width-matched pair, control leg
    ("dqwen3-1.7b-base-v3",   "step50000-swa"),  # control@50k (20260811-203500)
    ("llada-8b-base",         None),
    ("dream-7b-base",         None),
    ("dream-coder-7b-base",   None),
    ("coda-1.7b-base",        None),
]

# T1's six rows: the only ones that get the full-canvas sequential ceiling.
CEILING_ROWS = [
    ("dqwen3.5-0.8b-base-v3", "step50000-swa"),
    ("dqwen3.5-2b-base-v3",   "step50000-swa"),
    ("dqwen3.5-4b-base-v3",   "step50000-swa"),
    ("dqwen3.5-9b-base-v3",   "step50000-swa"),
    ("dqwen3.5-2b-base-v3",   "step25000-swa"),
    ("dqwen3-1.7b-base-v3",   "step25000-swa"),
]

ACCEL_ROWS = [
    ("dqwen3.5-0.8b-base-v3", "step50000-swa"),
    ("dqwen3.5-2b-base-v3",   "step50000-swa"),
    ("dqwen3.5-4b-base-v3",   "step50000-swa"),
    ("dqwen3.5-9b-base-v3",   "step50000-swa"),
    ("dqwen3.5-0.8b-base-v3", "step25000-swa"),  # T3: the budget axis
    ("dqwen3.5-2b-base-v3",   "step25000-swa"),
    ("dqwen3.5-4b-base-v3",   "step25000-swa"),
    ("dqwen3.5-9b-base-v3",   "step25000-swa"),
    ("dqwen3-1.7b-base-v3",   "step25000-swa"),
]

ACCEL_DECODES = (["block32-static-s%d" % s for s in (32, 16, 8, 4, 2)]
                 + ["block32-tau%s" % t for t in
                    ("0.5", "0.6", "0.7", "0.8", "0.9", "0.95")])


def cell(model, revision, decode, benchmark, shard=(0, 1)):
    d = {"model": model, "revision": revision, "decode": decode,
         "benchmark": benchmark}
    if tuple(shard) != (0, 1):
        d["shard"] = list(shard)
    return d


def sharded(model, revision, decode, benchmark, n):
    return [cell(model, revision, decode, benchmark, (k, n))
            for k in range(n)]


def tag(d):
    k, n = d.get("shard", (0, 1))
    return (f"{d['model']}@{d['revision'] or 'main'}"
            f"__{d['benchmark']}__{d['decode']}__s{k}of{n}")


def part1():
    return [cell(m, None, "ar", b) for m in AR_TWINS for b in AR_BENCHES]


def part2():
    cells = []
    for model, rev in BIG_ROWS:
        for b in CODE_BENCHES + ["gsm8k", "math"]:
            cells += sharded(model, rev, "block32-static-s32", b,
                             DLM_SHARDS[b])
        cells.append(cell(model, rev, "mc-nelbo", "mmlu"))
    for model, rev in CEILING_ROWS:
        cells.append(cell(model, rev, "standard-static-s512", "humaneval"))
    return cells


def part3(taken):
    cells = []
    for model, rev in ACCEL_ROWS:
        for decode in ACCEL_DECODES:
            cells.append(cell(model, rev, decode, "humaneval"))
            cells += sharded(model, rev, decode, "gsm8k", DLM_SHARDS["gsm8k"])
    return [c for c in cells if tag(c) not in taken]


def main():
    p1, p2 = part1(), part2()
    p3 = part3({tag(c) for c in p1 + p2})
    for name, cells in [("part1-ar-baselines", p1),
                        ("part2-big-table", p2),
                        ("part3-acceleration", p3)]:
        path = os.path.join(HERE, name + ".jsonl")
        with open(path, "w") as f:
            for c in cells:
                f.write(json.dumps(c) + "\n")
        print(f"{name}.jsonl  {len(cells)} cells")
    repos = sorted({(c["model"], c["revision"]) for c in p1 + p2 + p3})
    print(f"{len(repos)} distinct model@revision to prewarm")


if __name__ == "__main__":
    main()
