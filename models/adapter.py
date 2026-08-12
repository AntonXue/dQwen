"""The model contract: everything a decode engine needs from a model.

A sampler needs one thing -- logits for a canvas. Mask ids, compat shims,
logit alignment: all the adapter's problem, which is what keeps one engine
model-agnostic.

TWO logit surfaces, and the distinction is load-bearing:

  raw_logits(ids)  exactly what the model returns (reference code uses this).
  logits(ids)      canonical: logits[:, i] is the distribution FOR position i
                   (everything we build reads this one).

Dream and CoDA emit AR-aligned raw output; their `_canonicalize` applies the
same one-line shift (`cat([l[:, :1], l[:, :-1]])`) their own samplers apply.
Shifting twice -- or not at all -- is a silent off-by-one, which is why
alignment is a method pair, not a config flag.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about a model repo. `mask_id` is asserted against the config
    where the family exposes one, so a silent id drift cannot go unnoticed."""

    repo: str
    family: str
    mask_id: int
    pad_id: Optional[int] = None
    notes: str = ""


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

    # ids
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
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # the two logit surfaces
    @abstractmethod
    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Exactly what the model returns, unmodified. For NATIVE samplers."""

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw output to position-aligned. Identity for all but Dream."""
        return logits

    def logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Position-aligned logits. For PORTABLE samplers."""
        return self._canonicalize(self.raw_logits(input_ids, **kw))

    # text
    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


# HF dynamic-module loading. Comparators ship modeling code via
# trust_remote_code; resolving the CLASS before weights materialise is the
# hook each family uses to install compat shims at the moment they must land.


def resolve(repo: str, revision: Optional[str] = None, auto_class: str = "AutoModel"):
    """Return (config, model_class) for a remote-code repo, without loading weights."""
    cfg = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=True)
    amap = cfg.auto_map or {}
    ref = amap.get(auto_class)
    if ref is None:                      # fall back to whatever the repo does expose
        ref = amap.get("AutoModel") or amap.get("AutoModelForCausalLM")
    if ref is None:
        raise RuntimeError(f"{repo}: auto_map exposes no usable model class: {amap}")
    klass = get_class_from_dynamic_module(ref, repo, revision=revision)
    return cfg, klass


def tokenizer(repo: str, revision: Optional[str] = None):
    return AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)


def materialize(klass, repo: str, cfg, revision: Optional[str] = None,
                dtype=torch.bfloat16, device: str = "cuda"):
    model = klass.from_pretrained(repo, config=cfg, revision=revision, dtype=dtype)
    return model.to(device).eval()


def assert_finite_rope(model) -> None:
    """Guard against the tf5 meta-device buffer trap (the canonical writeup).

    transformers>=5 builds models on device("meta"); non-persistent buffers
    (rotary `inv_freq`) are absent from checkpoints and can materialise as
    garbage. tf5's repair lives in the base `_init_weights` -- remote-code
    models that override it never reach the repair. The failure is SILENT:
    correct shapes, noise logits (measured argmax agreement 20-40% vs
    native). Cheap to assert, so always asserted, every family.
    """
    bad = [n for n, b in model.named_buffers()
           if n.endswith("inv_freq") and not torch.isfinite(b).all()]
    if bad:
        raise RuntimeError(
            f"non-finite rotary inv_freq buffers: {bad[:3]}{'...' if len(bad) > 3 else ''}. "
            "This is the transformers>=5 meta-device trap -- the family's compat shim "
            "must recompute inv_freq AFTER from_pretrained materialises the model."
        )


# Upstream provenance ledger -- DATA, not plumbing: which commit each
# reference was vendored/verified from, and WHICH implementation produced
# WHICH published table. Unpinned upstream is worse than none (JetEngine's
# dynamic_threshold default drifted 0.9 -> 0.75 under research commits).
#   LLaDA published via their own lm-eval model: reproducible.
#   Dream published via a VENDORED modified lm-eval 0.4.8, never diffed.
#   SDAR published via LMDeploy (block=4, steps=4, tau=0.9 greedy), which
#     neither of their shipped samplers can express; family pruned
#     2026-08-12, pins kept for the manuscript's adjacent-decode prose.
UPSTREAM_PINS = {
    "LLaDA": "96441d4",        # ML-GSAI/LLaDA: generate.py (native sampler),
                               #   EVAL.md (published decode + task configs),
                               #   get_log_likelihood.py (MC-NELBO reference)
    "Dream": "31f94a6",        # DreamLM/Dream: eval/eval.py lm-eval wrapper
                               #   (the logit shift at :354)
    "Dream-Coder": "79d4387",  # DreamLM/Dream-Coder: variant of the above
    "SDAR": "6a12cdb",         # JetAstra/SDAR: generate.py
                               #   (block_diffusion_generate), OpenCompass configs
    "JetEngine": "bf8cb31",    # Labman42/JetEngine: SDAR inference engine,
                               #   ~15 remasking strategies, has a greedy path
}
