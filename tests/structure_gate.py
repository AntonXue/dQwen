"""Gate: the dependency rules of this repo, enforced by AST, not convention.

    <env>/python tests/structure_gate.py

RULES
  1. Inside models/, imports of repo code are RELATIVE (from .sibling
     import ...); an absolute repo import inside the package is a
     violation by spelling alone. No `..` climbing anywhere.
  2. Leaves import no repo code: models/adapter.py, models/samplers.py,
     and benchmark_specs.py (it may import from the pinned lm-eval, which
     is a dependency, not repo code).
  3. Only run.py assembles the stack; nothing imports `run` except tests.
  4. `import models` resolves to THIS repo (a dependency claiming the
     generic name would shadow silently otherwise).
"""

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_TOP = ("models", "benchmark_specs", "run", "summarize")
LEAVES = {"models/adapter.py", "models/samplers.py", "benchmark_specs.py"}


def repo_imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                yield node.module or ".", f"relative(level={node.level})"
            elif node.module and node.module.split(".")[0] in REPO_TOP:
                yield node.module, "absolute"
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in REPO_TOP:
                    yield a.name, "absolute"


def main():
    fails = 0
    files = sorted(ROOT.glob("models/*.py")) + [
        ROOT / "benchmark_specs.py", ROOT / "run.py", ROOT / "summarize.py",
        ROOT / "trajectories.py"]
    for path in files:
        rel = str(path.relative_to(ROOT))
        in_package = path.parent.name == "models"
        for mod, kind in repo_imports(path):
            ok = True
            if rel in LEAVES:
                ok = False                     # rule 2
            elif in_package and kind == "absolute":
                ok = False                     # rule 1
            elif kind.startswith("relative") and "level=1" not in kind:
                ok = False                     # rule 1 (no `..`)
            elif (mod or "").split(".")[0] == "run":
                ok = False                     # rule 3
            if not ok:
                print(f"FAIL {rel}: {kind} import of {mod!r}")
                fails += 1
    spec = importlib.util.find_spec("models")  # rule 4
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None or ROOT not in origin.parents:
        print(f"FAIL: `import models` resolves to {origin}, not this repo")
        fails += 1
    print("GATE " + ("FAILED" if fails else
                     "PASSED: package closed, leaves are leaves, names are ours"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
