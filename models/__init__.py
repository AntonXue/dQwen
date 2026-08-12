"""The model package: catalog, contract, decode, and the five families.

    from models import load, MODELS
    adapter = load("dqwen3.5-2b-base-v3", revision="step50000-swa")

Everything model-shaped lives here and the package is CLOSED: family files
import only their siblings (.adapter for the contract + loading helpers,
.samplers for decode types); nothing in it imports upward. adapter.py and
samplers.py deliberately do not import each other.

To register a model: add a ModelSpec entry to MODELS below (and, for a new
family, a <family>.py with a build() plus one BUILDERS line).
"""

import torch

from . import coda, dqwen, dream, llada
from .adapter import ModelAdapter, ModelSpec

MODELS: dict[str, ModelSpec] = {
    # ours (private org)
    # step checkpoints are HF revisions: load(..., revision="step30000-swa")
    "dqwen3.5-0.8b-base": ModelSpec("EER6b/dQwen3.5-0.8B-Base", "dqwen", 248061, 248044),
    "dqwen3.5-2b-base":   ModelSpec("EER6b/dQwen3.5-2B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-4b-base":   ModelSpec("EER6b/dQwen3.5-4B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-9b-base":   ModelSpec("EER6b/dQwen3.5-9B-Base",   "dqwen", 248061, 248044),
    # v3-MIXTURE production runs. Same recipe as the entries above (LR ladder, seed,
    # schedule) -- the mixture is the only deliberate change, so `-base` vs `-base-v3`
    # at a matched step count is the mixture A/B. 0.8B main == step50000-swa, which is
    # budget-matched to the released v1 (also 50k); step25000-swa is the half-budget leg.
    "dqwen3.5-0.8b-base-v3": ModelSpec("EER6b/dQwen3.5-0.8B-Base-v3", "dqwen", 248061, 248044),
    "dqwen3.5-2b-base-v3":   ModelSpec("EER6b/dQwen3.5-2B-Base-v3",   "dqwen", 248061, 248044),
    "dqwen3.5-4b-base-v3":   ModelSpec("EER6b/dQwen3.5-4B-Base-v3",   "dqwen", 248061, 248044),
    "dqwen3.5-9b-base-v3":   ModelSpec("EER6b/dQwen3.5-9B-Base-v3",   "dqwen", 248061, 248044),
    "dqwen3-0.6b-base":   ModelSpec("EER6b/dQwen3-0.6B-Base",   "dqwen", 151660, 151643),
    # v3-recipe backbone ablation: Qwen3-1.7B (FULL-attention) under the same recipe as the
    # hybrid 2B -- argparse.json differs in 2 of 22 keys. Also the same backbone as
    # Salesforce/CoDA-v0-Base, so this is the recipe-controlled CoDA comparison.
    # Qwen3 ids (151660/151643), NOT Qwen3.5's -- the publish script's own latent bug.
    "dqwen3-1.7b-base-v3": ModelSpec("EER6b/dQwen3-1.7B-Base-v3", "dqwen", 151660, 151643),
    "dqwen3-1.7b-base":   ModelSpec("EER6b/dQwen3-1.7B-Base",   "dqwen", 151660, 151643),
    # comparators
    "llada-8b-base":     ModelSpec("GSAI-ML/LLaDA-8B-Base",     "llada", 126336),
    "dream-7b-base":     ModelSpec("Dream-org/Dream-v0-Base-7B",     "dream", 151666),
    "dream-coder-7b-base": ModelSpec(
        "Dream-org/Dream-Coder-v0-Base-7B", "dream", 151666),
    # CoDA: Salesforce's masked DLM adapted from Qwen3-1.7B -- the SAME backbone as
    # our dQwen3-1.7B control, so this is the closest available recipe-isolating
    # comparison (differs in data + budget, not in starting weights).
    "coda-1.7b-base": ModelSpec(
        "Salesforce/CoDA-v0-Base", "coda", 151669, 151643,
        notes="AR-aligned raw logits (shift in _canonicalize); needs the "
              "CoDAModel._supports_sdpa shim on transformers>=4.54"),
}

BUILDERS = {
    "llada": llada.build,
    "dream": dream.build,
    "coda": coda.build,
    "dqwen": dqwen.build,
}
assert {spec.family for spec in MODELS.values()} <= set(BUILDERS), \
    "every family named in the MODELS catalog needs a builder here"


def load(name: str, revision=None, dtype=torch.bfloat16,
         device: str = "cuda") -> ModelAdapter:
    """Load a registered model, applying that family's compat shims."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    spec = MODELS[name]
    return BUILDERS[spec.family](spec, revision=revision, dtype=dtype,
                                 device=device)
