"""Gate: the dependency rules of this repo, enforced by AST, not convention.

    <env>/python tests/structure_gate.py

RULES
  1. Inside models/ and benchmarks/, imports of repo code are RELATIVE
     (from .sibling import ...); an absolute `models.`/`benchmarks.`
     import inside a package member is a violation by spelling alone.
  2. adapter.py, samplers.py and _grading.py are leaves: they import no
     repo code at all.
  3. No `..` climbing anywhere.
  4. Only run.py assembles the stack; nothing imports `run` except tests.
  5. The names `models` and `benchmarks` resolve to THIS repo (a future
     dependency claiming either would shadow silently otherwise).
"""

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PACKAGES = ("models", "benchmarks")
LEAVES = {"models/adapter.py", "models/samplers.py", "benchmarks/_grading.py"}


def repo_imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                yield node.module or ".", f"relative(level={node.level})"
            elif node.module and node.module.split(".")[0] in (*PACKAGES, "run"):
                yield node.module, "absolute"
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in (*PACKAGES, "run"):
                    yield a.name, "absolute"


def main():
    fails = 0
    files = [p for pkg in PACKAGES for p in sorted(ROOT.glob(f"{pkg}/*.py"))]
    for path in files + [ROOT / "run.py", ROOT / "summarize.py"]:
        rel = str(path.relative_to(ROOT))
        in_package = path.parent.name in PACKAGES
        for mod, kind in repo_imports(path):
            ok = True
            if rel in LEAVES:
                ok = False                     # rule 2
            elif in_package and kind == "absolute":
                ok = False                     # rule 1
            elif kind.startswith("relative") and "level=1" not in kind:
                ok = False                     # rule 3
            elif (mod or "").split(".")[0] == "run":
                ok = False                     # rule 4
            if not ok:
                print(f"FAIL {rel}: {kind} import of {mod!r}")
                fails += 1
    for name in PACKAGES:                      # rule 5
        spec = importlib.util.find_spec(name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None or ROOT not in origin.parents:
            print(f"FAIL: `import {name}` resolves to {origin}, not this repo")
            fails += 1
    print("GATE " + ("FAILED" if fails else
                     "PASSED: packages closed, leaves are leaves, names are ours"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
