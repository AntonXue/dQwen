"""Access to pinned upstream reference checkouts.

Upstream repos are reference material and test fixtures -- NEVER runtime
dependencies. Nothing here is on the import path during a normal eval. Two callers:

  * tests/parity/           mints goldens by running their code unmodified
  * families/*/sampler_upstream.py   the `--sampler upstream` escape hatch

Why the escape hatch exists at all, given we re-implement everything: a touchup
creates a code path with NO upstream counterpart to compare against. SDAR greedy is
the clean example -- greedy does not exist upstream (their script divides by
temperature then multinomials), so our greedy can never be parity-tested. Being able
to run their code on demand lets us at least bracket such a path instead of flying
blind. It also costs almost nothing, since the parity fixtures need it anyway.

Populate with `third_party/fetch.sh`.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"


def pins() -> dict:
    with open(_THIRD_PARTY / "pins.json") as f:
        return json.load(f)["repos"]


def path_for(name: str) -> Path:
    p = _THIRD_PARTY / name
    if not p.exists():
        raise FileNotFoundError(
            f"upstream reference {name!r} not fetched. Run third_party/fetch.sh"
        )
    return p


def head_of(name: str) -> str:
    out = subprocess.run(["git", "-C", str(path_for(name)), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def verify_pin(name: str, strict: bool = True) -> str:
    """Check the checkout sits at its pinned commit.

    Unpinned upstream is worse than no upstream: JetEngine's dynamic_threshold
    default has already drifted 0.9 -> 0.75 under opt-numbered research commits, so
    a number attributed to "their sampler" without a commit is unattributable.
    """
    want = pins()[name]["commit"]
    have = head_of(name)
    if not have.startswith(want) and not want.startswith(have):
        msg = f"{name} is at {have}, pinned to {want}. Run third_party/fetch.sh"
        if strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
    return have


@contextlib.contextmanager
def on_path(name: str, strict: bool = True):
    """Temporarily put a pinned upstream checkout on sys.path."""
    p = str(path_for(name))
    verify_pin(name, strict=strict)
    sys.path.insert(0, p)
    try:
        yield p
    finally:
        if p in sys.path:
            sys.path.remove(p)
