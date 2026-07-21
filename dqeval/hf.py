"""Thin helpers over HuggingFace dynamic-module loading.

Every comparator ships its modeling code via `trust_remote_code`, so we resolve the
class through the dynamic-module machinery rather than the Auto* factories. That
gives each family a hook to install its compat shims on the CLASS before any
weights are materialised -- which is where several of them have to happen.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


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
    """Guard against the tf5 meta-device buffer trap.

    transformers>=5 builds models under a bare `torch.device("meta")`. Rotary
    `inv_freq` is a non-persistent buffer, so it is absent from the checkpoint and
    materialises as uninitialised memory. tf5 repairs this inside the BASE
    `PreTrainedModel._init_weights`, but any remote-code model that OVERRIDES
    `_init_weights` (Dream and SDAR both do) never reaches that branch.

    The failure is silent: the model loads, forwards, and returns correctly-shaped
    logits that are noise. Measured cost when unrepaired: argmax agreement 20-40%
    against the native environment. Cheap to assert, so we always assert.
    """
    bad = [n for n, b in model.named_buffers()
           if n.endswith("inv_freq") and not torch.isfinite(b).all()]
    if bad:
        raise RuntimeError(
            f"non-finite rotary inv_freq buffers: {bad[:3]}{'...' if len(bad) > 3 else ''}. "
            "This is the transformers>=5 meta-device trap -- the family's compat shim "
            "must recompute inv_freq AFTER from_pretrained materialises the model."
        )
