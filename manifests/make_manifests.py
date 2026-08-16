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
  part4  The decode-axis extension (manuscript EVAL-REQUEST 20260815-023130,
         2026-08-15): tau smoke gate + full 11-decode sweeps for the
         control@50k and the four comparators. Additive-only; section below.

Evening rulings (Anton 2026-08-13, second session):
  * MATH500 replaces full MATH-5000 -- the column's reproduction anchor was
    already spent by the math_verify metric ruling, and 5k was the single
    biggest compute line (~150 GPU-h -> ~15). benchmark key: math500.
  * Acceleration ALSO runs mbpp (500 docs, of2) -- HE+GSM8K+MBPP frontier.
  * part2 gains full @25k rows for 0.8B/4B/9B (the S4.2 budget-trade
    symmetry; 2B already had one). NO v1-mixture arm (S4.6 stays
    workstation-stamped).
  * AR cells get fence benchmarks too (over-run: lets the format table
    carry AR rows if ever narrated).
  * AR canvas: run.py now wires BENCH gen into HFLM (stock hardcodes 256).

Shard rule (Anton 2026-08-13 evening): ~250 docs per generation cell.
mmlu is exempt (mc-nelbo generates nothing -- single-token MC scores via the
shared-forward path, ~1 forward/doc); AR cells stay UNSHARDED regardless
(bs=16 batch composition moves numbers ~0.23pp between striped and whole).

Canvas rule (Anton 2026-08-13): gen=1024 on EVERY generation benchmark
(BENCH in run.py). Matches LLaDA-Base's published protocol on all four of
its benchmarks, covers CoDA's 768, and clears MATH's gold-solution tail
(8.5% of golds exceed 512 tokens; anatomy in _claude/20260813-005207).
The full-canvas ceiling decode is therefore standard-static-s1024.

Run: python manifests/make_manifests.py   (rewrites the .jsonl siblings)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# benchmark -> stripe count for DLM generation cells (~250 docs/shard:
# he 164 whole, mbpp 500/2=250, mbpp+ 378/2=189, gsm8k 1319/6~220,
# math500 500/2=250)
DLM_SHARDS = {"humaneval": 1, "humaneval-plus": 1, "mbpp": 2, "mbpp-plus": 2,
              "mbpp-fence": 2, "mbpp-plus-fence": 2, "gsm8k": 6,
              "math500": 2}

CODE_BENCHES = ["humaneval", "humaneval-plus", "mbpp", "mbpp-plus",
                "mbpp-fence", "mbpp-plus-fence"]
AR_BENCHES = ["humaneval", "humaneval-plus", "mbpp", "mbpp-plus",
              "mbpp-fence", "mbpp-plus-fence", "gsm8k", "math500", "mmlu"]

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
    ("dqwen3.5-0.8b-base-v3", "step25000-swa"),  # S4.2 budget-trade rows: every
    ("dqwen3.5-4b-base-v3",   "step25000-swa"),  #   size at both budgets, all
    ("dqwen3.5-9b-base-v3",   "step25000-swa"),  #   columns (over-run ruling)
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
        for b in CODE_BENCHES + ["gsm8k", "math500"]:
            cells += sharded(model, rev, "block32-static-s32", b,
                             DLM_SHARDS[b])
        cells.append(cell(model, rev, "mc-nelbo", "mmlu"))
    for model, rev in CEILING_ROWS:
        cells.append(cell(model, rev, "standard-static-s1024", "humaneval"))
    return cells


def part3(taken):
    cells = []
    for model, rev in ACCEL_ROWS:
        for decode in ACCEL_DECODES:
            cells.append(cell(model, rev, decode, "humaneval"))
            cells += sharded(model, rev, decode, "gsm8k", DLM_SHARDS["gsm8k"])
            cells += sharded(model, rev, decode, "mbpp", DLM_SHARDS["mbpp"])
    return [c for c in cells if tag(c) not in taken]


# ---- part4: the decode-axis extension (manuscript EVAL REQUEST 2026-08-15,
# VistaCoder_Technical_Report/_claude/20260815-023130). Purely ADDITIVE --
# part1/2/3 must keep regenerating byte-identical. Two figures drive it:
# fig:decode-retention wants the control's ~100B leg swept (it is in
# BIG_ROWS but was never in ACCEL_ROWS), and fig:decode-frontier cannot be
# drawn at all until the four comparators have more than one decode config.
# 4a is a GATE: the tau commit rule has never run outside dQwen3.5 -- do
# not queue the big sweeps until the smoke cells pass (criteria in
# _claude/20260815-*-HANDOFF). Smoke cells are real phase 4b/4c cells and
# fast-skip there.

SMOKE_ROWS = [("coda-1.7b-base", None), ("llada-8b-base", None),
              ("dream-7b-base", None), ("dream-coder-7b-base", None)]

ACCEL4_CHEAP = [("dqwen3-1.7b-base-v3", "step50000-swa"),  # retention @100B
                ("coda-1.7b-base", None)]                  # frontier small panel
ACCEL4_BIG   = [("llada-8b-base", None),                   # frontier 7-9B panel
                ("dream-7b-base", None),
                ("dream-coder-7b-base", None)]

# Over-run extension kept (Anton 2026-08-15): 4f, the sequential ceiling
# (standard-static-s1024, HE) for the four comparators + control@50k --
# completes the one-at-a-time anchor for every frontier panel, and for
# LLaDA it IS their published decode (gen=steps=block=1024). A tau TAIL
# (0.98/0.99 for all frontier rows, 252 cells) was cut the same night as
# too ambitious: the tau grid ENDS AT 0.95, and the fast-end saturation
# (tau0.95 = 64.9 forwards vs tau0.9's 57.0 vs s32's 114.9 on 2B@50k HE,
# accuracy flat) is narrated as a finding instead of extended.
CEILING4_ROWS = ACCEL4_CHEAP + ACCEL4_BIG


def part4(taken):
    smoke = [cell(m, r, "block32-tau0.9", "humaneval") for m, r in SMOKE_ROWS]
    cheap, big, mbpp = [], [], []
    for model, rev in ACCEL4_CHEAP:
        for decode in ACCEL_DECODES:
            cheap.append(cell(model, rev, decode, "humaneval"))
            cheap += sharded(model, rev, decode, "gsm8k", DLM_SHARDS["gsm8k"])
            cheap += sharded(model, rev, decode, "mbpp", DLM_SHARDS["mbpp"])
    for model, rev in ACCEL4_BIG:
        for decode in ACCEL_DECODES:
            big.append(cell(model, rev, decode, "humaneval"))
            big += sharded(model, rev, decode, "gsm8k", DLM_SHARDS["gsm8k"])
            mbpp += sharded(model, rev, decode, "mbpp", DLM_SHARDS["mbpp"])
    ceil = [cell(m, r, "standard-static-s1024", "humaneval")
            for m, r in CEILING4_ROWS]
    seen, out = set(taken), []
    for name, cells in [("part4a-tau-smoke", smoke),
                        ("part4b-accel-cheap", cheap),
                        ("part4c-accel-big", big),
                        ("part4d-accel-big-mbpp", mbpp),
                        ("part4f-ceilings", ceil)]:
        keep = [c for c in cells if tag(c) not in seen]
        seen.update(tag(c) for c in keep)
        out.append((name, keep))
    return out


# ---- part5: the FULL-CANVAS protocol campaign (EVAL-REQUEST 20260815-211016
# + Anton's compromise ruling 2026-08-16). Every comparator publishes
# full-canvas base numbers (LLaDA steps=len=1024, no blocks; Dream's sampler
# has no block concept) while our whole grid is block32 -- full-canvas is
# expected to become the main-table protocol. The compromise: MATH500
# carries the decode axes (consensus 4-shot, and our family ALIGNS there --
# 9B leads all comparators at s32 where on gsm8k it trails all of them);
# GSM8K keeps one full-canvas headline config so the flipped table keeps
# its column. Gates first: standard-static below s1024 has never run;
# standard-tau has never run at all.
#
# RULING 2026-08-16 (Anton, reversal #2 on the 4-shot question): GSM8K
# stays 8-SHOT, PERIOD -- the s1024 8-shot cells are "heavy and acceptable
# anyways"; gsm8k-4shot is DROPPED ENTIRELY (part5d deleted, 176 cells).
# The 08-14 cancellation stands reinforced; do not re-propose 4-shot.
# BENCH["gsm8k-4shot"] remains in run.py as inert code (no manifest
# references it).

STANDARD_LADDER = ["standard-static-s%d" % s for s in (32, 64, 128, 256, 512, 1024)]
STANDARD_TAUS = ["standard-tau%s" % t for t in ("0.5", "0.6", "0.7", "0.8", "0.9", "0.95")]
LADDER_BENCHES = ["humaneval", "mbpp", "math500"]   # gsm8k deliberately absent
GATE_ROWS = [("dqwen3.5-2b-base-v3", "step50000-swa"), ("llada-8b-base", None)]


def part5(taken):
    gates = [cell(m, r, d, "humaneval") for m, r in GATE_ROWS
             for d in ("standard-static-s32", "standard-static-s128",
                       "standard-tau0.9")]
    ladder, gsm_fc, backfill, tau = [], [], [], []
    for model, rev in BIG_ROWS:
        for decode in STANDARD_LADDER:
            for b in LADDER_BENCHES:
                ladder += sharded(model, rev, decode, b, DLM_SHARDS[b])
        gsm_fc += sharded(model, rev, "standard-static-s1024", "gsm8k",
                          DLM_SHARDS["gsm8k"])
        for decode in ACCEL_DECODES:
            if decode != "block32-static-s32":
                backfill += sharded(model, rev, decode, "math500",
                                    DLM_SHARDS["math500"])
        for decode in STANDARD_TAUS:
            for b in LADDER_BENCHES:
                tau += sharded(model, rev, decode, b, DLM_SHARDS[b])
    # part5g (manuscript amendment 2026-08-15 21:54 + the regrade-item
    # resolution): the flipped tables' MBPP-variant columns need REAL
    # generations at the headline (mbpp-plus = evalplus's EDITED prompts;
    # fences = a different template -- only HumanEval+ shares its base
    # prompts). humaneval-plus rides along as grid cells rather than a CPU
    # regrade: EvalPlus is removed and the standing ruling is "new '+'
    # numbers are grid cells"; greedy + shared prompts regenerate identical
    # text, so the cells ARE the regrade, with provenance. Headline decode
    # only -- no figure wants a variant ladder.
    variants = []
    for model, rev in BIG_ROWS:
        for b in ("mbpp-plus", "mbpp-fence", "mbpp-plus-fence",
                  "humaneval-plus"):
            variants += sharded(model, rev, "standard-static-s1024", b,
                                DLM_SHARDS[b])
    seen, out = set(taken), []
    for name, cells in [("part5a-fullcanvas-gates", gates),
                        ("part5b-fullcanvas-ladder", ladder),
                        ("part5c-fullcanvas-gsm8k", gsm_fc),
                        ("part5e-math500-block-backfill", backfill),
                        ("part5g-headline-variants", variants),
                        ("part5f-fullcanvas-tau", tau)]:
        keep = [c for c in cells if tag(c) not in seen]
        seen.update(tag(c) for c in keep)
        out.append((name, keep))
    return out


def main():
    p1, p2 = part1(), part2()
    p3 = part3({tag(c) for c in p1 + p2})
    p4 = part4({tag(c) for c in p1 + p2 + p3})
    p4_cells = [c for _, cells in p4 for c in cells]
    p5 = part5({tag(c) for c in p1 + p2 + p3 + p4_cells})
    all_cells = p1 + p2 + p3 + p4_cells
    for name, cells in ([("part1-ar-baselines", p1),
                         ("part2-big-table", p2),
                         ("part3-acceleration", p3)] + p4 + p5):
        path = os.path.join(HERE, name + ".jsonl")
        with open(path, "w") as f:
            for c in cells:
                f.write(json.dumps(c) + "\n")
        print(f"{name}.jsonl  {len(cells)} cells")
    all_cells += [c for _, cells in p5 for c in cells]
    repos = sorted({(c["model"], c["revision"]) for c in all_cells})
    print(f"{len(repos)} distinct model@revision to prewarm")


if __name__ == "__main__":
    main()
