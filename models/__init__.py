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

from . import coda, dlm1b, dqwen, dream, llada, qwen3sym
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
    # DLM1B project (sibling repo ~/foo/DLM1B): AR->DLM adaptation of Qwen3-0.6B via
    # on-the-fly-tokenized OWT continued pretraining. `repo` is the local training-run
    # dir (NOT an HF hub id); `revision` selects the milestone checkpoint subdirectory
    # (380/760/1530/3050/5340/7600, or "main" for the highest step present). Same ids
    # as dqwen3-0.6b-base (same Qwen3-0.6B backbone, same MASK/PAD tokens).
    "dlm1b-owt-baseline-1b": ModelSpec(
        "/home/ayx98/foo/DLM1B/_runs/20260814-044535-owt-baseline-1b", "dlm1b",
        151660, 151643,
        notes="cross-repo: weights+modeling code live in ~/foo/DLM1B, tokenizer "
              "sourced from Qwen/Qwen3-0.6B (checkpoint dir ships no tokenizer)"),
    # DLM1B qwen3_sym conversion arms (Vista scratch; 2026-08-26 campaign).
    # Matched total compute: bdlm ck-10000 vs the two staged arms at ck-5000.
    # Each checkpoint runs through ITS OWN attn_mode (adapter-enforced; toy
    # Exp 4: an SDLM ckpt through free attention collapses).
    "qwen3sym-bdlm10k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_bdlm10k_20260826_122624",
        "qwen3sym", 151660, 151643,
        notes="BDLM direct arm: bidir from stock Qwen3-1.7B, 10k steps"),
    "qwen3sym-cdlm5k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_20260826_122649",
        "qwen3sym", 151660, 151643,
        notes="CDLM shared stage: causal attn + frozen-aug masking, 5k steps"),
    "qwen3sym-cdlm-bdlm5k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_free5k_20260826_123637",
        "qwen3sym", 151660, 151643,
        notes="CDLM->BDLM arm ('free'): bidir from cdlm ck-5000, 5k steps"),
    "qwen3sym-cdlm-sdlm5k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_sym5k_20260826_123637",
        "qwen3sym", 151660, 151643,
        notes="CDLM->SDLM arm ('sym'): symmetrized from cdlm ck-5000, 5k steps"),
}

BUILDERS = {
    "llada": llada.build,
    "dream": dream.build,
    "coda": coda.build,
    "dqwen": dqwen.build,
    "dlm1b": dlm1b.build,
    "qwen3sym": qwen3sym.build,
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
