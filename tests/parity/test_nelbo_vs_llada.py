"""Parity gate: our MC-NELBO == LLaDA's reference implementation, exactly.

This is the estimator that makes our multiple-choice numbers comparable to LLaDA's
and Dream's rather than merely adjacent to them, so "we implemented the same thing"
has to be a test, not a claim.

Runs BOTH implementations against the same weights, same prompt/answer, same seed:

    ours   dqeval.sampler.mc_nelbo_loglikelihood   (via the ported adapter, tf 5.13)
    theirs <LLaDA repo>/get_log_likelihood.py    (unmodified upstream)

Expected: bitwise identical. `forward_process` is a byte-faithful copy of theirs, so
with the RNG seeded identically every masking draw matches and the two numbers agree
to the last digit. A mismatch means the copy has drifted -- most likely someone
"cleaned up" the RNG call order.

Usage:
    python tests/parity/test_nelbo_vs_llada.py
    python tests/parity/test_nelbo_vs_llada.py --llada-path ~/foo/LLaDA --mc-num 16
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from dqeval.adapter import load
from dqeval.sampler import mc_nelbo_loglikelihood

PROMPT = "The capital of France is"
ANSWER = " Paris, a city known for its art and history."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llada-path", default=os.path.expanduser("~/foo/LLaDA"))
    ap.add_argument("--mc-num", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ref = os.path.join(a.llada_path, "get_log_likelihood.py")
    if not os.path.exists(ref):
        print(f"SKIP: LLaDA reference not found at {ref}")
        print("      clone github.com/ML-GSAI/LLaDA (pinned 96441d4, see third_party/LOCKFILE.md)")
        return 0

    adapter = load("llada-8b-base")
    tok = adapter.tokenizer
    prompt = tok(PROMPT, return_tensors="pt").input_ids[0].to(adapter.device)
    answer = tok(ANSWER, return_tensors="pt").input_ids[0].to(adapter.device)
    print(f"prompt={len(prompt)} tok, answer={len(answer)} tok, "
          f"mc_num={a.mc_num}, mask_id={adapter.mask_id}")

    # --- ours -------------------------------------------------------------
    torch.manual_seed(a.seed)
    mine = mc_nelbo_loglikelihood(
        adapter.logits, prompt, answer,
        mc_num=a.mc_num, batch_size=a.batch_size, mask_id=adapter.mask_id,
    )

    # --- theirs, unmodified ----------------------------------------------
    sys.path.insert(0, a.llada_path)
    from get_log_likelihood import get_log_likelihood  # noqa: E402

    torch.manual_seed(a.seed)
    theirs = get_log_likelihood(
        adapter.model, prompt, answer,
        mc_num=a.mc_num, batch_size=a.batch_size,
        cfg_scale=0.0, mask_id=adapter.mask_id,
    )
    theirs = float(theirs)

    diff = abs(mine - theirs)
    print(f"  ours   = {mine:.6f}")
    print(f"  theirs = {theirs:.6f}")
    print(f"  diff   = {diff:.3e}")

    ok = diff == 0.0
    print(f"\n  {'PASS' if ok else 'FAIL'}  MC-NELBO parity vs LLaDA reference")
    if not ok:
        print("  Our forward_process must stay a byte-faithful copy of theirs;")
        print("  check that the torch.randint / torch.randperm order is unchanged.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
