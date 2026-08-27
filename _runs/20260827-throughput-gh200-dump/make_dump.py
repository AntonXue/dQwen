"""Regenerate the viz/diagnostic dump from the raw throughput JSONLs.

Run from the repo root:  python _runs/20260827-throughput-gh200-dump/make_dump.py
Idempotent: rewrites every derived file in this directory from scratch.
Raw JSONLs are never touched. See dump_README.md for the file inventory.
"""
import csv
import json
import statistics
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
ARM = {("fast", True): "deployment", ("math", True): "fla+math",
       ("fast", False): "fallback+flash", ("math", False): "naive"}
DATASETS = {
    "v2-i50": "20260827-throughput-gh200-i50",
    "v1": "20260827-throughput-gh200",
    "stability-a": "20260827-throughput-gh200-stability-a",
    "stability-b": "20260827-throughput-gh200-stability-b",
    "stability-c": "20260827-throughput-gh200-stability-c",
}
COLS = ["dataset", "model", "arm", "sdpa_arm", "fla", "causal_conv1d", "node",
        "mode", "batch", "length", "iters", "median_ms", "iqr_ms", "min_ms",
        "tokens_per_s", "peak_mem_mb", "oom", "unsupported", "d", "vocab",
        "trunk_module", "device", "torch"]

rows = []
for ds, dirname in DATASETS.items():
    for f in sorted((RUNS / dirname).glob("throughput_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            r["dataset"] = ds
            r["arm"] = ARM[(r["sdpa_arm"], r["causal_conv1d"])]
            rows.append({c: r.get(c, "") for c in COLS})
with open(OUT / "flat_all_records.csv", "w", newline="") as f:
    w = csv.DictWriter(f, COLS)
    w.writeheader()
    w.writerows(rows)

def cell(ds, model, arm, mode, B, L, field="tokens_per_s"):
    for r in rows:
        if (r["dataset"] == ds and r["model"] == model and r["arm"] == arm
                and r["mode"] == mode and r["batch"] == B and r["length"] == L
                and r[field] != ""):
            return r[field]
    return None

# tier1_ratios.csv: hybrid/control ratio over the full v2 deployment grid
HY, CT = "dqwen3.5-2b-base-v3", "dqwen3-1.7b-base-v3"
with open(OUT / "tier1_ratios.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "batch", "length", "hybrid_tok_s", "control_tok_s", "ratio"])
    seen = sorted({(r["mode"], r["batch"], r["length"]) for r in rows
                   if r["dataset"] == "v2-i50" and r["model"] == HY
                   and r["arm"] == "deployment" and r["tokens_per_s"] != ""})
    for mode, B, L in seen:
        h = cell("v2-i50", HY, "deployment", mode, B, L)
        c = cell("v2-i50", CT, "deployment", mode, B, L)
        if h and c:
            w.writerow([mode, B, L, h, c, round(h / c, 4)])

# arms_2x2.csv: every hybrid x arm x slice point (full mode)
with open(OUT / "arms_2x2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "arm", "batch", "length", "tokens_per_s"])
    for m in ["dqwen3.5-0.8b-base-v3", HY, "dqwen3.5-4b-base-v3", "dqwen3.5-9b-base-v3"]:
        for arm in ("deployment", "fla+math", "fallback+flash", "naive"):
            for B, L in [(1, 1024), (1, 4096), (16, 1024), (16, 4096)]:
                v = cell("v2-i50", m, arm, "full", B, L)
                if v:
                    w.writerow([m, arm, B, L, v])

# stability.csv: Tier-1 slice medians across the five measurement occasions
with open(OUT / "stability.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "batch", "length", "occasion", "dataset", "median_ms"])
    for m in (HY, CT):
        for B, L in [(1, 1024), (1, 4096), (16, 1024), (16, 4096)]:
            for occ, ds in [("v1-c608", "v1"), ("v2-c639", "v2-i50"),
                            ("repeat-a", "stability-a"), ("repeat-b", "stability-b"),
                            ("repeat-c", "stability-c")]:
                v = cell(ds, m, "deployment", "full", B, L, "median_ms")
                if v:
                    w.writerow([m, B, L, occ, ds, v])

# v1_v2_delta.csv: per-cell deployment median deltas (iters sensitivity)
with open(OUT / "v1_v2_delta.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "mode", "batch", "length", "v1_ms", "v2_ms", "pct_delta"])
    v1 = {(r["model"], r["mode"], r["batch"], r["length"]): r["median_ms"]
          for r in rows if r["dataset"] == "v1" and r["arm"] == "deployment"
          and r["median_ms"] != ""}
    v2 = {(r["model"], r["mode"], r["batch"], r["length"]): r["median_ms"]
          for r in rows if r["dataset"] == "v2-i50" and r["arm"] == "deployment"
          and r["median_ms"] != ""}
    for k in sorted(set(v1) & set(v2)):
        w.writerow([*k, v1[k], v2[k], round((v2[k] - v1[k]) / v1[k] * 100, 2)])

# oom_envelope.csv + identity_check.csv + model_summary.csv
with open(OUT / "oom_envelope.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "model", "arm", "mode", "batch", "length"])
    for r in rows:
        if r["oom"] is True:
            w.writerow([r["dataset"], r["model"], r["arm"], r["mode"],
                        r["batch"], r["length"]])
with open(OUT / "identity_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "batch", "length", "full_ms", "trunk_plus_head_ms", "ratio"])
    models = sorted({r["model"] for r in rows if r["dataset"] == "v2-i50"})
    for m in models:
        for B, L in [(16, 1024), (4, 4096)]:
            fu = cell("v2-i50", m, "deployment", "full", B, L, "median_ms")
            tr = cell("v2-i50", m, "deployment", "trunk", B, L, "median_ms")
            he = cell("v2-i50", m, "deployment", "head", B, L, "median_ms")
            if fu and tr and he:
                w.writerow([m, B, L, fu, round(tr + he, 3), round(fu / (tr + he), 3)])
with open(OUT / "model_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "d", "vocab", "b16_l4096_tok_s", "b1_l1024_tok_s",
                "b1_l1024_ms", "trunk_supported"])
    for m in models:
        meta = next(r for r in rows if r["dataset"] == "v2-i50" and r["model"] == m)
        w.writerow([m, meta["d"], meta["vocab"],
                    cell("v2-i50", m, "deployment", "full", 16, 4096),
                    cell("v2-i50", m, "deployment", "full", 1, 1024),
                    cell("v2-i50", m, "deployment", "full", 1, 1024, "median_ms"),
                    any(r["dataset"] == "v2-i50" and r["model"] == m
                        and r["mode"] == "trunk" and r["median_ms"] != ""
                        for r in rows)])

print(f"flat_all_records.csv: {len(rows)} records; derived tables rewritten")
