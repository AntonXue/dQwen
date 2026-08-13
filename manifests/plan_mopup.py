"""Mop-up tiles: one node-lane per still-pending cell, straight from
`run.py --pending`, submitted with the ordinary slurm_launch.sh (8h wall).

One cell per lane on purpose: leftovers are either wall-clipped monsters
(CoDA canvas-fills every doc -- ~2.5h/cell measured) or transient faults,
so maximum parallelism + a whole wall per cell removes all packing risk.

Usage:
    python manifests/plan_mopup.py manifests/part2-big-table.jsonl [more...]
    bash slurm_launch.sh <manifest>.mopup.tiles.json      # login node
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "run.py")

for m in sys.argv[1:]:
    out = subprocess.run([sys.executable, RUN, "--pending", m],
                         capture_output=True, text=True)
    idx = [int(i) for i in out.stdout.replace(",", " ").split()
           if i.strip().isdigit()]
    if not idx:
        print(f"{m}: nothing pending")
        continue
    jobs = [[[i] for i in idx[k:k + 16]] for k in range(0, len(idx), 16)]
    path = m.replace(".jsonl", ".mopup.tiles.json")
    json.dump({"manifest": m, "lane_budget_h": 8.0, "jobs": jobs},
              open(path, "w"))
    print(f"{path}: {len(idx)} cells -> "
          f"{len(jobs)} job(s) of {[len(j) for j in jobs]} lanes")
    print(f"  submit (login node):  bash slurm_launch.sh {path}")
