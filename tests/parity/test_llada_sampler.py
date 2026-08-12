"""Parity gates for the LLaDA sampler.

Two claims, both load-bearing, both cheap to check and expensive to get wrong:

  GATE 1  vendored native == unmodified upstream, token-for-token.
          Makes "this is the sampler shipped with LLaDA" a tested statement rather
          than an assertion about a file we copied by hand.

  GATE 2  our unified engine under the matching preset == the native sampler.
          This is THE central architectural claim of dqeval -- that a family's
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
    g1_fail, g2_fail = [], []

    for p in PROMPTS:
        print(f"\n  prompt {p!r}")
        ids = adapter.encode(p)

        nat = llada.generate_native(adapter, ids, cfg)
        show("native", nat)

        try:
            up = llada.generate_upstream(adapter, ids, cfg)
            show("upstream", up)
            if torch.equal(nat.gen_ids.cpu(), up.gen_ids.cpu()):
                print("    GATE 1 PASS  native == upstream (token-identical)")
            else:
                n = int((nat.gen_ids.cpu() != up.gen_ids.cpu()).sum())
                print(f"    GATE 1 FAIL  {n}/{nat.gen_ids.numel()} tokens differ")
                g1_fail.append(p)
        except FileNotFoundError as e:
            print(f"    GATE 1 SKIP  {e}")

        uni = unified.generate(adapter, ids, cfg)
        show("unified", uni)
        if torch.equal(nat.gen_ids.cpu(), uni.gen_ids.cpu()):
            print("    GATE 2 PASS  unified@preset == native (token-identical)")
        else:
            n = int((nat.gen_ids.cpu() != uni.gen_ids.cpu()).sum())
            print(f"    GATE 2 FAIL  {n}/{nat.gen_ids.numel()} tokens differ")
            g2_fail.append(p)

    print("\n" + "=" * 70)
    print(f"  GATE 1 (native == upstream): {'PASS' if not g1_fail else 'FAIL ' + str(g1_fail)}")
    print(f"  GATE 2 (unified == native):  {'PASS' if not g2_fail else 'FAIL ' + str(g2_fail)}")
    return 1 if (g1_fail or g2_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
