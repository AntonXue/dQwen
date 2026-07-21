"""SDAR compat for transformers 5.13, plus one upstream repo defect.

Two DIFFERENT kinds of patch live here, and the distinction matters for how we
report results:

  * version shims  -- restore transformers 4.x behaviour. Verified, never invented.
  * a repo patch   -- works around a file MISSING from JetLM's HF repos. It is
                      version-independent: their repo fails to load under
                      transformers 4.57 too. Not a porting artefact of ours.

VERIFIED: with these, SDAR-1.7B/4B forward output under transformers 5.13 matches
transformers 4.57 at fixed torch; two of three probe runs were bitwise identical and
the residual is below the intra-environment noise floor (their attention is wrapped
in @torch.compile(max-autotune-no-cudagraphs), which is itself nondeterministic).
"""

from __future__ import annotations

import sys
import types
from typing import Optional

import torch
import transformers.dynamic_module_utils as dmu

MISSING_MODULE = "fused_linear_diffusion_cross_entropy"


# --------------------------------------------------------------------------
# Repo defect: modeling_sdar.py imports a module their repo does not ship.
# --------------------------------------------------------------------------
class _TrainingOnlySymbol:
    def __init__(self, *a, **kw):
        raise RuntimeError(
            "FusedLinearDiffusionCrossEntropyLoss is a training-only symbol and is "
            "not shipped in JetLM's HF repo. dqeval is inference-only."
        )


def _install_import_filter() -> None:
    """Stop transformers trying to FETCH the missing file.

    transformers resolves relative imports twice: once at fetch time (scanning the
    source for `from .X import`) and again at exec time. This handles the first --
    without it we fail with OSError before Python's import system is ever reached.
    """
    if getattr(dmu, "_dqeval_sdar_filter", False):
        return
    _orig = dmu.get_relative_imports

    def patched(module_file):
        return [r for r in _orig(module_file) if r != MISSING_MODULE]

    dmu.get_relative_imports = patched
    dmu._dqeval_sdar_filter = True


def _register_stub(pkg: str) -> None:
    name = f"{pkg}.{MISSING_MODULE}"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.FusedLinearDiffusionCrossEntropyLoss = _TrainingOnlySymbol
        sys.modules[name] = stub


def resolve(repo: str, revision: Optional[str] = None,
            auto_class: str = "AutoModelForCausalLM"):
    """Resolve SDAR's model class, working around the missing module.

    The dynamic package name embeds a commit sha we cannot know up front, and on a
    cold cache the module directory does not exist yet. So we let the first
    ModuleNotFoundError tell us the exact name, register the stub there, and retry.
    Nothing is written to the shared HuggingFace module cache.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _install_import_filter()
    cfg = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=True)
    amap = cfg.auto_map or {}
    ref = amap.get(auto_class) or amap.get("AutoModelForCausalLM") or amap.get("AutoModel")
    if ref is None:
        raise RuntimeError(f"{repo}: auto_map exposes no usable model class: {amap}")

    for _ in range(3):
        try:
            return cfg, get_class_from_dynamic_module(ref, repo, revision=revision)
        except ModuleNotFoundError as e:
            if not e.name or not e.name.endswith(MISSING_MODULE):
                raise
            _register_stub(e.name.rsplit("." + MISSING_MODULE, 1)[0])
    raise RuntimeError(f"{repo}: stub registration did not converge")


# --------------------------------------------------------------------------
# Version shims
# --------------------------------------------------------------------------
def _rope_default_4x(config=None, device=None, seq_len=None, **rope_kwargs):
    """Verbatim transformers 4.57.1 `_compute_default_rope_parameters`.

    tf>=5 removed the "default" key from ROPE_INIT_FUNCTIONS; SDAR's remote code
    (a fork of tf 4.52.4 modeling_qwen3.py) still indexes it.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def apply(cfg, klass) -> list[str]:
    log: list[str] = []
    mod = sys.modules[klass.__module__]

    # (1) tf 4.x carried pad_token_id on the base config; SDARModel.__init__ reads it.
    #     Verified by loading this config under tf 4.57.1: it resolves to None
    #     (their config.json has no such key).
    if not hasattr(cfg, "pad_token_id"):
        cfg.pad_token_id = None
        log.append("cfg.pad_token_id=None (verified == tf4.57 value)")

    # (2) ROPE_INIT_FUNCTIONS lost its "default" key -- same as Dream.
    if "default" not in mod.ROPE_INIT_FUNCTIONS:
        mod.ROPE_INIT_FUNCTIONS = dict(mod.ROPE_INIT_FUNCTIONS)
        mod.ROPE_INIT_FUNCTIONS["default"] = _rope_default_4x
        log.append("ROPE_INIT_FUNCTIONS['default'] restored (verbatim tf4.57 impl)")

    return log


def repair_rope(model) -> str:
    """THE SILENT ONE -- same root cause as Dream, different repair.

    tf>=5 materialises under a bare meta device, so the non-persistent `inv_freq`
    buffer (absent from the checkpoint) comes back as garbage; tf5's repair branch
    lives in the BASE _init_weights, which SDARPreTrainedModel overrides. Unlike
    Dream, SDAR's rotary module has no reset_parameters(), so we re-run its own
    rope_init_fn. Measured: tf4.57 inv_freq.sum()=5.150445; unshimmed tf5.13
    sum=nan with original_inv_freq still on meta; after repair, 5.150445 exactly.
    """
    n = 0
    for m in model.modules():
        if "RotaryEmbedding" not in type(m).__name__:
            continue
        fn = getattr(m, "rope_init_fn", None)
        conf = getattr(m, "config", None)
        if fn is None or conf is None:
            continue
        dev = next(model.parameters()).device
        inv_freq, _ = fn(conf, dev)
        m.register_buffer("inv_freq", inv_freq, persistent=False)
        m.original_inv_freq = inv_freq
        n += 1
    return f"post-load inv_freq recompute via rope_init_fn ({n} modules)"
