"""The model contract.

A portable sampler needs exactly one thing from a model: logits for a canvas.
Everything else -- mask ids, whether the family needs an eval-side logit shift,
which transformers-5 compat shims to install, chat formatting -- is the adapter's
problem. That is what makes the model x sampler cross-product cheap.

TWO logit surfaces, and the distinction is load-bearing:

  raw_logits(ids)  exactly what the model returns.
                   NATIVE samplers use this -- they carry their own shift.
  logits(ids)      canonical: logits[:, i] is the distribution FOR position i.
                   PORTABLE samplers use this, so one engine runs on any family.

Dream is the only family whose raw output is AR-aligned; its adapter overrides
`_canonicalize` to apply `cat([l[:, :1], l[:, :-1]])` -- the same expression their
own sampler and eval wrapper each apply once. Running Dream's native sampler on
canonicalised logits would double-shift it: silent, off-by-one, no exception. That
is precisely why the two surfaces are separate methods rather than a config flag.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GenOutput:
    """One generation. bs=1 by construction -- see `dqeval.samplers`."""

    prompt_ids: torch.Tensor
    gen_ids: torch.Tensor
    text: str
    n_forward: int = 0

    @property
    def full_ids(self) -> torch.Tensor:
        return torch.cat([self.prompt_ids, self.gen_ids], dim=-1)


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about a model repo. `mask_id` is asserted against the config
    where the family exposes one, so a silent id drift cannot go unnoticed."""

    repo: str
    family: str
    mask_id: int
    pad_id: Optional[int] = None
    auto_class: str = "AutoModel"
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    # ---- ours (private org) ---------------------------------------------
    # step checkpoints are HF revisions: load(..., revision="step30000-swa")
    "dqwen3.5-0.8b-base": ModelSpec("EER6b/dQwen3.5-0.8B-Base", "dqwen", 248061, 248044),
    "dqwen3.5-2b-base":   ModelSpec("EER6b/dQwen3.5-2B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-4b-base":   ModelSpec("EER6b/dQwen3.5-4B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-9b-base":   ModelSpec("EER6b/dQwen3.5-9B-Base",   "dqwen", 248061, 248044),
    "dqwen3-0.6b-base":   ModelSpec("EER6b/dQwen3-0.6B-Base",   "dqwen", 151660, 151643),
    "dqwen3-1.7b-base":   ModelSpec("EER6b/dQwen3-1.7B-Base",   "dqwen", 151660, 151643),
    # ---- comparators -----------------------------------------------------
    "llada-8b-base":     ModelSpec("GSAI-ML/LLaDA-8B-Base",     "llada", 126336),
    "llada-8b-instruct": ModelSpec("GSAI-ML/LLaDA-8B-Instruct", "llada", 126336),
    "dream-7b-instruct": ModelSpec("Dream-org/Dream-v0-Instruct-7B", "dream", 151666),
    "dream-7b-base":     ModelSpec("Dream-org/Dream-v0-Base-7B",     "dream", 151666),
    "dream-coder-7b-base": ModelSpec(
        "Dream-org/Dream-Coder-v0-Base-7B", "dream", 151666),
    "dream-coder-7b-instruct": ModelSpec(
        "Dream-org/Dream-Coder-v0-Instruct-7B", "dream", 151666),
    "sdar-1.7b-chat": ModelSpec(
        "JetLM/SDAR-1.7B-Chat", "sdar", 151669, auto_class="AutoModelForCausalLM",
        notes="repo is missing fused_linear_diffusion_cross_entropy.py; see families/sdar/compat.py"),
    "sdar-4b-chat": ModelSpec(
        "JetLM/SDAR-4B-Chat", "sdar", 151669, auto_class="AutoModelForCausalLM",
        notes="same missing-module defect as SDAR-1.7B"),
}


class ModelAdapter(ABC):
    """Wraps a loaded model behind the two logit surfaces."""

    family: str = "?"

    def __init__(self, model, tokenizer, spec: ModelSpec,
                 revision: Optional[str] = None, shims: Optional[list[str]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.revision = revision or "main"
        self.shims = shims or []

    # -- ids ---------------------------------------------------------------
    @property
    def mask_id(self) -> int:
        return self.spec.mask_id

    @property
    def pad_id(self) -> int:
        if self.spec.pad_id is not None:
            return self.spec.pad_id
        tok = self.tokenizer
        for cand in (tok.pad_token_id, tok.eos_token_id):
            if cand is not None:
                return cand
        raise ValueError(f"{self.spec.repo}: no pad/eos token id available")

    @property
    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # -- the two logit surfaces -------------------------------------------
    @abstractmethod
    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Exactly what the model returns, unmodified. For NATIVE samplers."""

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw output to position-aligned. Identity for all but Dream."""
        return logits

    def logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Position-aligned logits. For PORTABLE samplers."""
        return self._canonicalize(self.raw_logits(input_ids, **kw))

    # -- text --------------------------------------------------------------
    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def chat(self, message: str) -> str:
        """Instruct-format a single user turn. Base models return it unchanged."""
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if tmpl is None:
            return message
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False, add_generation_prompt=True,
        )

    def describe(self) -> dict:
        """Provenance for the result ledger."""
        import transformers
        return {
            "model": self.spec.repo,
            "revision": self.revision,
            "family": self.family,
            "mask_id": self.mask_id,
            "shims": list(self.shims),
            "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        }


_FAMILY_MODULES = {
    "llada": "dqeval.families.llada.adapter",
    "dream": "dqeval.families.dream.adapter",
    "sdar": "dqeval.families.sdar.adapter",
    "dqwen": "dqeval.families.dqwen.adapter",
}


def load(name: str, revision: Optional[str] = None, dtype=torch.bfloat16,
         device: str = "cuda") -> ModelAdapter:
    """Load a registered model, applying that family's compat shims."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    spec = MODELS[name]
    mod = importlib.import_module(_FAMILY_MODULES[spec.family])
    return mod.build(spec, revision=revision, dtype=dtype, device=device)
