"""Parity gate for the released `generate` method.

THE claim: the method shipped inside each dQwen repo's modeling file is the
harness engine (models/samplers.py) -- token-identical, forward-identical --
under every mode it exposes. Lane A patches the method from models/ onto the
loaded model (same weights, same object: isolates the port); --from-hub skips
the patch and gates the file the Hub actually serves.

Usage:
    python tests/test_dqwen_release_sampler.py                     # 0.8B, lane A
    python tests/test_dqwen_release_sampler.py --model dqwen3-1.7b-base-v3
    python tests/test_dqwen_release_sampler.py --all --from-hub
"""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from models import load
from models.samplers import DecodeConfig, generate as harness_generate

PROMPTS = ["def add(a, b):\n    ", "The capital of France is"]
FIVE = ["dqwen3.5-0.8b-base-v3", "dqwen3.5-2b-base-v3", "dqwen3.5-4b-base-v3",
        "dqwen3.5-9b-base-v3", "dqwen3-1.7b-base-v3"]
G = 64
CASES = {  # name: (harness DecodeConfig kwargs, generate kwargs)
    "block32 greedy 1tok/step": (dict(block_length=32, steps_per_block=32),
                                 dict(block_length=32, tau=None)),
    "block32 tau0.9":           (dict(block_length=32, steps_per_block=32, commit="dynamic", confidence_threshold=0.9),
                                 dict(block_length=32, tau=0.9)),
    "free-canvas tau0.9":       (dict(block_length=G, steps_per_block=G, commit="dynamic", confidence_threshold=0.9),
                                 dict(block_length=None, tau=0.9)),
    "block32 static s8":        (dict(block_length=32, steps_per_block=8),
                                 dict(block_length=32, tau=None, steps_per_block=8)),
    "sampled t0.7 p0.9 seed0":  (dict(block_length=32, steps_per_block=32, temperature=0.7, top_p=0.9, seed=0),
                                 dict(block_length=32, tau=None, temperature=0.7, top_p=0.9, seed=0)),
    "stop-string":              (dict(block_length=16, steps_per_block=16),
                                 dict(block_length=16, tau=None)),
    "defaults (free, tau0.9, 512)": (dict(gen_length=512, block_length=512, steps_per_block=512, commit="dynamic", confidence_threshold=0.9),
                                 dict()),
}

def patch_from_release(adapter):
    fn = "modeling_dqwen3_5.py" if "DiffuQwen35" in type(adapter.model).__name__ else "modeling_dqwen3.py"
    spec = importlib.util.spec_from_file_location("dqwen_release_" + fn[:-3], Path(__file__).resolve().parent.parent / "models" / fn)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    src = getattr(mod, type(adapter.model).__name__)
    for k in ("generate", "_dg_top_k", "_dg_top_p", "_dg_assert_finite_rope", "_NEG_INF"):
        setattr(type(adapter.model), k, src.__dict__[k])
    return fn

def run(name, from_hub):
    adapter = load(name)
    src = "hub-served file" if from_hub else patch_from_release(adapter)
    print(f"\n{name}  [{src}]")
    fails = []
    for case, (hk, dk) in CASES.items():
        stop = ["\ndef", "\n\n"] if case == "stop-string" else None
        for p in PROMPTS:
            ids = adapter.encode(p)
            h = harness_generate(adapter, ids, DecodeConfig(order="low_confidence", **{"gen_length": G, "temperature": 0.0, **hk}), stop_strings=stop)
            m = adapter.model.generate(ids, adapter.tokenizer, stop_strings=stop, **({} if case.startswith("defaults") else {"gen_length": G}), **dk)
            ok = (torch.equal(h.gen_ids.cpu(), m.gen_ids.cpu()) and h.n_forward == m.n_forward
                  and torch.equal(h.commit_step, m.commit_step) and h.text == m.text)
            print(f"  {'PASS' if ok else 'FAIL'}  {case:26s} fwd={h.n_forward:<3} {m.text[:38]!r}")
            if not ok: fails.append((case, p))
    return fails

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=FIVE[0]); ap.add_argument("--all", action="store_true"); ap.add_argument("--from-hub", action="store_true")
    a = ap.parse_args(); fails = []
    for n in (FIVE if a.all else [a.model]): fails += run(n, a.from_hub)
    print("\n" + "=" * 70 + f"\n  GATE: {'PASS' if not fails else 'FAIL ' + str(fails)}"); return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
