"""Parity gates for the LLaDA sampler.

THE claim, load-bearing, cheap to check and expensive to get wrong:

  GATE    our unified engine under the matching preset == the native sampler
          (models/llada.py generate_native, vendored verbatim from the pinned
          upstream and verified bitwise against it in July -- receipts in the
          port-evidence _claude doc).
          This is THE central architectural claim -- that a family's
          published decode is a PRESET in one parameter space, not a separate code
          path. If gate 2 fails, the cross-product design needs revisiting, not the
          test. Known candidate cause: our engine suppresses the MASK id before the
          argmax and LLaDA's does not.

Both run greedy (temperature=0) so they are deterministic and any mismatch is real
rather than an RNG artifact.

Usage:
    python tests/parity/test_llada_sampler.py
    python tests/parity/test_llada_sampler.py --gen-length 64 --block-length 32
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import sys

import torch

from models import load
from models.samplers import DecodeConfig
from models import llada
from models import samplers as unified

PROMPTS = [
    "def add(a, b):\n    ",
    "The capital of France is",
]


def show(tag: str, out) -> None:
    print(f"    {tag:<10} fwd={out.n_forward:<4} {out.text[:64]!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada-8b-base")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps-per-block", type=int, default=None)
    a = ap.parse_args()

    spb = a.steps_per_block or a.block_length
    cfg = DecodeConfig(
        gen_length=a.gen_length, block_length=a.block_length, steps_per_block=spb,
        temperature=0.0, order="low_confidence", commit="static",
    )
    print(f"cfg: gen={cfg.gen_length} block={cfg.block_length} "
          f"steps/block={cfg.steps_per_block} blocks={cfg.num_blocks} greedy={cfg.greedy}")

    adapter = load(a.model)
    g2_fail = []

    for p in PROMPTS:
        print(f"\n  prompt {p!r}")
        ids = adapter.encode(p)

        nat = llada.generate_native(adapter, ids, cfg)
        show("native", nat)

        uni = unified.generate(adapter, ids, cfg)
        show("unified", uni)
        if torch.equal(nat.gen_ids.cpu(), uni.gen_ids.cpu()):
            print("    GATE PASS  unified@preset == native (token-identical)")
        else:
            n = int((nat.gen_ids.cpu() != uni.gen_ids.cpu()).sum())
            print(f"    GATE FAIL  {n}/{nat.gen_ids.numel()} tokens differ")
            g2_fail.append(p)

    print("\n" + "=" * 70)
    print(f"  GATE (unified == native): {'PASS' if not g2_fail else 'FAIL ' + str(g2_fail)}")
    return 1 if g2_fail else 0


if __name__ == "__main__":
    sys.exit(main())
