#!/usr/bin/env python
"""run.py — the one entrypoint for eval cells. Full guide: README.md.

  python run.py MODEL REVISION DECODE BENCHMARK [K/N]     one cell
  python run.py manifest.jsonl INDEX                      one manifest row
  python run.py --list                                    what can I ask for?

Examples:
  python run.py dqwen3.5-2b-base-v3 step50000-swa block32-tau0.8 gsm8k 2/8
  python run.py dqwen3.5-9b-base-v3 main block32-static-s8 humaneval
  python run.py Qwen/Qwen3.5-2B - ar gsm8k

Everything else (env flags, sys.path, idempotent skip, output layout) is
handled here or in dqeval.cell_runner — there is nothing to set up first.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv):
    from dqeval.cell_runner import (Cell, run_cell, load_manifest, BENCH,
                             TAU_GRID, BLOCK_STATIC, STANDARD_STATIC)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "--list":
        from dqeval.models import MODELS
        print("models (dqeval registry; AR cells take a bare HF id instead):")
        for name in MODELS:
            print(f"  {name}")
        print("\nbenchmarks (gen length / shots / suggested shards):")
        for b, cfg in BENCH.items():
            print(f"  {b:12s} gen={cfg['gen']}  shots={cfg['shots']}  shards={cfg['shards']}")
        print("\ndecode schemes:")
        print("  ar | mc-nelbo (mmlu only)")
        print(f"  block32-static-sK    K in {BLOCK_STATIC}")
        print(f"  standard-static-sK   K in {STANDARD_STATIC}")
        print(f"  block32-tauT | standard-tauT   T in {TAU_GRID}")
        return 0
    if argv[0].endswith(".jsonl"):
        if len(argv) != 2:
            print("usage: run.py manifest.jsonl INDEX"); return 2
        run_cell(load_manifest(argv[0])[int(argv[1])])
        return 0
    if len(argv) not in (4, 5):
        print("usage: run.py MODEL REVISION DECODE BENCHMARK [K/N]  "
              "(run.py --help for examples)"); return 2
    model, rev, decode, bench = argv[:4]
    shard = (tuple(int(x) for x in argv[4].split("/"))
             if len(argv) > 4 else (0, 1))
    run_cell(Cell(model, None if rev in ("main", "-") else rev,
                  decode, bench, shard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
