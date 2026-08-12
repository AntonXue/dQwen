"""TEMPORARY back-compat shim -- the tool moved to the repo root
(evalplus_driver.py). The 2026-08-12 MBPP+ lanes were launched against this
module path; delete this file once _runs/20260812-095023-mbpp-plus-generations
finishes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalplus_driver import *          # noqa: F401,F403
from evalplus_driver import main

if __name__ == "__main__":
    sys.exit(main())
