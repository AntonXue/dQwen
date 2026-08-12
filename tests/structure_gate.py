"""Gate: the dependency rules of dqeval, enforced by AST, not convention.

    PYTHONPATH=. <env>/python tests/structure_gate.py

RULES
  1. No file imports from a parent level: inside a package, dqeval imports
     must be RELATIVE (from .sibling import ...); an absolute `dqeval.`
     import inside a package member is a violation by spelling alone.
  2. adapter.py and samplers.py do not import each other (or anything
     else in dqeval): they are leaves.
  3. Package __init__ files import only their own members (+ leaves).
  4. Depth-0 files (cell_runner.py) may import packages, never the
     reverse: nothing anywhere imports dqeval.cell_runner except run.py
     and tests.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEAVES = {"dqeval/models/adapter.py", "dqeval/models/samplers.py",
          "dqeval/benchmarks/_grading.py"}


def dq_imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("dqeval"):
                yield node.module, "absolute"
            elif node.level > 0:
                yield node.module or ".", f"relative(level={node.level})"
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("dqeval"):
                    yield a.name, "absolute"


def main():
    fails = 0
    for path in sorted(ROOT.glob("dqeval/**/*.py")):
        rel = str(path.relative_to(ROOT))
        in_package = path.parent.name in ("models", "benchmarks")
        for mod, kind in dq_imports(path):
            ok = True
            if rel in LEAVES:
                ok = False                     # leaves import nothing from dqeval
            elif in_package and kind == "absolute":
                ok = False                     # rule 1: relative-only inside packages
            elif kind.startswith("relative") and "level=1" not in kind:
                ok = False                     # no `..` climbing, ever
            elif "cell_runner" in (mod or "") and rel != "run.py":
                ok = False                     # rule 4
            if not ok:
                print(f"FAIL {rel}: {kind} import of {mod!r}")
                fails += 1
    print("GATE " + ("FAILED" if fails else
                     "PASSED: no parent-level imports; leaves are leaves"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
