"""Tile a manifest into ~3.2h node-lanes, 16 lanes per job (8h wall).

The qgh QOS caps jobs per user (20 running / 40 submitted, array TASKS
count), so cells are packed into lanes: one lane = one node's sequential
cell list, one job = up to 16 lanes. Short 16-node jobs backfill a busy
queue far better than one long wide job, and the SU waste is bounded by
each lane's slack instead of a straggler tail. The 8h wall is 2.5x the
lane target -- estimate error would have to be huge to hit it.

Cost model: GH200 probe anchors (_claude/20260813-005207 UPDATE 2) plus
structure -- static-sK forwards scale ~K/32 of the s32 anchor, tau cells
are bounded by ~0.7x s32, the s1024 ceiling is exactly 1024 fwd/doc,
mc-nelbo mmlu is ~1 fwd/doc via the shared-forward path. AR rates are
coarse guesses recalibrated by part 1's measured wall clocks (every cell's
meta record carries wall_s). Estimates being wrong is CHEAP: an overrun
lane dies at the wall, its finished cells persist, and a mop-up
`--pending` -> re-tile -> resubmit round converges.

Usage: python manifests/plan_tiles.py manifests/part1-ar-baselines.jsonl
Writes <manifest>.tiles.json and prints the packing + sbatch lines.
"""

import json
import sys

LANE_BUDGET_H = 3.2      # pack target inside the 4h wall (~20% margin)
LANES_PER_JOB = 16

DOCS = {"humaneval": 164, "humaneval-plus": 164, "mbpp": 250,
        "mbpp-plus": 189, "mbpp-fence": 250, "mbpp-plus-fence": 189,
        "gsm8k": 220, "math500": 250, "mmlu": 14042}
# mean forwards/doc at block32-static-s32, canvas 1024 (probe-measured for
# gsm8k/math; code extrapolated from committed lengths)
FWD_S32 = {"humaneval": 200, "humaneval-plus": 200, "mbpp": 260,
           "mbpp-plus": 260, "mbpp-fence": 260, "mbpp-plus-fence": 260,
           "gsm8k": 160, "math500": 300}
SEC_PER_FWD = [("0.8b", 0.035), ("0.6b", 0.030), ("1.7b", 0.045),
               ("2b", 0.045), ("4b", 0.055), ("9b", 0.070),
               ("dream", 0.100), ("llada", 0.110), ("coda", 0.050)]
# AR cells are HFLM bs=16 whole-benchmark runs: coarse minutes, by bench
AR_MINUTES = {"humaneval": 8, "humaneval-plus": 8, "mbpp": 12,
              "mbpp-plus": 10, "mbpp-fence": 12, "mbpp-plus-fence": 10,
              "gsm8k": 30, "math500": 20, "mmlu": 25}
AR_SIZE = [("0.6B", 0.7), ("0.8B", 0.7), ("1.7B", 1.0), ("2B", 1.0),
           ("4B", 1.3), ("7B", 1.8), ("9B", 1.8)]


def sec_per_fwd(model):
    for key, s in SEC_PER_FWD:
        if key in model:
            return s
    raise KeyError(f"no fwd rate for {model!r}")


def est_seconds(c):
    """Estimated lane time for one cell, including load overhead."""
    bench, decode, model = c["benchmark"], c["decode"], c["model"]
    if decode == "ar":
        size = next(f for key, f in AR_SIZE if key in model)
        return 90 + AR_MINUTES[bench] * 60 * size
    load = 240 if ("llada" in model or "dream" in model) else 150
    if decode == "mc-nelbo":
        return load + DOCS["mmlu"] * sec_per_fwd(model) * 1.25
    k, n = c.get("shard", (0, 1))
    docs = DOCS[bench] if n == 1 else DOCS[bench]  # DOCS already per-shard
    if decode.startswith("standard-static"):
        fwd = 1024.0
    elif "tau" in decode:
        fwd = FWD_S32[bench] * 0.7
    else:
        steps = int(decode.rsplit("-s", 1)[1])
        fwd = FWD_S32[bench] * steps / 32
    return load + docs * fwd * sec_per_fwd(model)


def pack(cells):
    """Model-contiguous next-fit into lanes of <= LANE_BUDGET_H."""
    order = sorted(range(len(cells)),
                   key=lambda i: (cells[i]["model"],
                                  cells[i]["revision"] or "",
                                  -est_seconds(cells[i])))
    budget = LANE_BUDGET_H * 3600
    lanes, cur, cur_s = [], [], 0.0
    for i in order:
        s = est_seconds(cells[i])
        if cur and cur_s + s > budget:
            lanes.append((cur, cur_s))
            cur, cur_s = [], 0.0
        cur.append(i)
        cur_s += s
    if cur:
        lanes.append((cur, cur_s))
    return lanes


def main(path):
    cells = [json.loads(l) for l in open(path)]
    lanes = pack(cells)
    jobs = [lanes[j:j + LANES_PER_JOB]
            for j in range(0, len(lanes), LANES_PER_JOB)]
    out = path.replace(".jsonl", ".tiles.json")
    json.dump({"manifest": path, "lane_budget_h": LANE_BUDGET_H,
               "jobs": [[idx for idx, _ in job] for job in jobs]},
              open(out, "w"), indent=None)
    total = sum(s for _, s in lanes)
    print(f"{path}: {len(cells)} cells -> {len(lanes)} lanes -> "
          f"{len(jobs)} jobs   (est total {total / 3600:.0f} node-h)")
    for j, job in enumerate(jobs):
        hrs = [s / 3600 for _, s in job]
        print(f"  job {j}: {len(job):>2} lanes, est/lane "
              f"min {min(hrs):.1f}h mean {sum(hrs) / len(hrs):.1f}h "
              f"max {max(hrs):.1f}h")
    over = [s for _, s in lanes if s > 3.6 * 3600]
    if over:
        print(f"  WARNING: {len(over)} lane(s) estimated past 3.6h")
    print("submit:")
    for j, job in enumerate(jobs):
        print(f"  sbatch -N {len(job)} sbatch_tiles.sh {out} {j}")


if __name__ == "__main__":
    main(sys.argv[1])
