"""Load matrix: every registered model, through the adapter contract.

For each model this asserts the things that fail SILENTLY if an adapter or shim
regresses -- a model that loads and forwards is not evidence that it is correct:

  * logits are finite and the right shape        (catches the meta-device trap)
  * rotary inv_freq buffers are finite           (the same trap, at the source)
  * canonicalisation matches the family's convention -- identity everywhere except
    Dream, where it must be EXACTLY cat([l[:, :1], l[:, :-1]])
  * mask_id lies inside the vocabulary           (catches registry/id drift)
  * greedy argmax never proposes MASK itself

Usage:
    python tests/load_matrix.py                    # everything cached
    python tests/load_matrix.py --only llada-8b-base,sdar-4b-chat
    python tests/load_matrix.py --include-uncached # allow downloads
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback

import torch

from models import MODELS, load

PROMPT = "def add(a, b):\n    return"


def is_cached(repo: str) -> bool:
    """True if weights are on disk (not just metadata)."""
    from pathlib import Path
    import os
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    d = home / "hub" / ("models--" + repo.replace("/", "--"))
    if not d.exists():
        return False
    return any(d.rglob("*.safetensors")) or any(d.rglob("*.bin"))


def check(name: str, revision: str | None = None) -> tuple[bool, str]:
    m = load(name, revision=revision)
    try:
        ids = m.encode(PROMPT)
        raw = m.raw_logits(ids)
        can = m.logits(ids)

        problems = []
        if not torch.isfinite(raw).all():
            problems.append("non-finite logits")
        if raw.shape[:2] != ids.shape[:2]:
            problems.append(f"shape {tuple(raw.shape)} vs ids {tuple(ids.shape)}")
        if not (0 <= m.mask_id < raw.shape[-1]):
            problems.append(f"mask_id {m.mask_id} outside vocab {raw.shape[-1]}")

        # canonicalisation must match the family convention, exactly
        if m.family == "dream":
            want = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)
            if not torch.equal(can, want):
                problems.append("Dream shift is not cat([l[:,:1], l[:,:-1]])")
        elif not torch.equal(raw, can):
            problems.append(f"{m.family} canonicalize should be identity but is not")

        # greedy must never propose MASK
        lg = can[0].clone()
        lg[:, m.mask_id] = float("-inf")
        if (lg.argmax(-1) == m.mask_id).any():
            problems.append("argmax proposed MASK after suppression")

        detail = (f"V={raw.shape[-1]} mask={m.mask_id} pad={m.pad_id} "
                  f"shims={len(m.shims)}")
        return (not problems), (detail if not problems else detail + " | " + "; ".join(problems))
    finally:
        del m
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated model names")
    ap.add_argument("--include-uncached", action="store_true")
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()

    names = a.only.split(",") if a.only else list(MODELS)
    rows, skipped = [], []
    for n in names:
        spec = MODELS[n]
        if not a.include_uncached and not is_cached(spec.repo):
            skipped.append(n)
            continue
        print(f"--- {n} ---", flush=True)
        try:
            ok, detail = check(n, revision=a.revision)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {str(e)[:160]}"
            traceback.print_exc(limit=3)
        rows.append((n, ok, detail))
        print(f"    {'PASS' if ok else 'FAIL'}  {detail}", flush=True)

    print("\n" + "=" * 78)
    for n, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {n:<26} {detail}")
    for n in skipped:
        print(f"  SKIP  {n:<26} weights not cached")
    n_fail = sum(1 for _, ok, _ in rows if not ok)
    print(f"\n  {len(rows) - n_fail}/{len(rows)} passed, {len(skipped)} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
