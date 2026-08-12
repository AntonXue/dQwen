"""Dream / Dream-Coder adapter -- the one family that needs an eval-side logit shift.

Dream's raw output is AR-aligned: `logits[:, i]` predicts position i+1. Every
consumer in their own codebase corrects for this immediately after each forward,
with the identical expression, in two independent places:

    generation_utils.py:421-422   (their sampler)
    eval/eval.py:354              (their lm-eval wrapper)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

So the shift belongs to the CONSUMER, and we put it in `_canonicalize`. The
consequence to keep in mind:

    portable samplers -> adapter.logits()      shift applied here, once
    Dream's native sampler -> adapter.raw_logits()   it shifts internally, itself

Feeding canonicalised logits to Dream's native sampler would apply the shift twice:
a silent off-by-one with no exception. That is the whole reason the two surfaces are
separate methods.
"""

from typing import Optional
import torch
from .adapter import (ModelAdapter, ModelSpec,
                           assert_finite_rope, materialize, resolve, tokenizer)
import sys
from transformers.generation.configuration_utils import GenerationConfig
from .adapter import ModelAdapter
from .samplers import GenOutput
from .samplers import DecodeConfig


class DreamAdapter(ModelAdapter):
    family = "dream"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, **kw)
        return out.logits if hasattr(out, "logits") else out[0]

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """AR-aligned -> position-aligned. Verbatim Dream's own expression."""
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> DreamAdapter:
    cfg, klass = resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    shims = apply_shims(cfg, klass)
    tok = tokenizer(spec.repo, revision=revision)
    model = materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    assert_finite_rope(model)          # shim (3) is silent if it regresses

    cfg_mask = getattr(cfg, "mask_token_id", None)
    if cfg_mask is not None and cfg_mask != spec.mask_id:
        raise ValueError(
            f"{spec.repo}: registry mask_id={spec.mask_id} but config says {cfg_mask}"
        )
    return DreamAdapter(model, tok, spec, revision=revision, shims=shims)


# (merged from dqeval/families/dream/compat.py)
"""Dream compat shims for transformers 5.13.

PRINCIPLE: a shim RESTORES a behaviour transformers 4.x had; it never invents one.
All four below were verified against transformers 4.57.1 source or measured in the
native environment. None is invented.

VERIFIED: with these shims and torch held fixed, Dream's forward output under
transformers 5.13 is BITWISE IDENTICAL to transformers 4.57, and 6 of 7 sampler
configurations reproduce native token-for-token (the 7th differs only at
temperature > 0, where torch's RNG stream changed between 2.5.1 and 2.7.1).

Shim 3 is the dangerous one: without it the model loads and forwards with NO
exception and returns noise (argmax agreement 20-38% vs native). It is the reason
golden fixtures are mandatory rather than nice to have.
"""


def _rope_default_4x(config=None, device=None, seq_len=None, **rope_kwargs):
    """Verbatim restoration of transformers 4.57.1 `_compute_default_rope_parameters`.

    tf>=5 deleted the "default" entry from ROPE_INIT_FUNCTIONS and inlined the body
    as a staticmethod that reads `config.rope_parameters["rope_theta"]`. Dream's
    remote code still indexes `ROPE_INIT_FUNCTIONS["default"]` and passes a config
    whose `rope_theta` is 1000000.0 under both versions (checked in both envs).
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
    return inv_freq, 1.0      # (inv_freq, attention_factor); factor unused for "default"


def apply_shims(cfg, klass) -> list[str]:
    log: list[str] = []
    mod = sys.modules[klass.__module__]

    # (1) ROPE_INIT_FUNCTIONS lost its "default" key.
    #     tf4.57 keys: default dynamic linear llama3 longrope yarn
    #     tf5.13 keys:         dynamic linear llama3 longrope proportional yarn
    #     Patched module-locally so global transformers is left untouched.
    if "default" not in mod.ROPE_INIT_FUNCTIONS:
        mod.ROPE_INIT_FUNCTIONS = dict(mod.ROPE_INIT_FUNCTIONS)
        mod.ROPE_INIT_FUNCTIONS["default"] = _rope_default_4x
        log.append("ROPE_INIT_FUNCTIONS['default'] restored (verbatim tf4.57 impl)")

    gc = mod.DreamGenerationConfig

    # (2) GenerationConfig.update() now calls self.validate(user_set_attributes=...).
    #     Dream overrides validate() with a 2-arg signature. Verified via
    #     inspect.getsource in both envs: their override is a no-op in 4.x too,
    #     so swallowing the kwargs is exact, not lossy.
    if not getattr(gc, "_dqeval_validate", False):
        gc.validate = lambda self, *a, **kw: None
        gc._dqeval_validate = True
        log.append("DreamGenerationConfig.validate(**kwargs) tolerated (no-op in 4.x too)")

    # (3) THE SILENT ONE. tf4.57 built models under accelerate's
    #     init_empty_weights(include_buffers=False), so non-persistent buffers stayed
    #     real and __init__'s computed inv_freq survived. tf>=5 uses a bare
    #     torch.device("meta"), so inv_freq is meta, is absent from the checkpoint,
    #     and materialises as garbage. tf5 repairs this in the BASE
    #     PreTrainedModel._init_weights -- but Dream OVERRIDES _init_weights, so that
    #     branch never runs. Repair must happen AFTER materialisation: hooking
    #     _init_weights does not work (the module is still on meta at that point).
    #     Measured: 4.57 inv_freq.sum()=5.150445; 5.13 unshimmed sum=-279837.44 with
    #     per-layer NaN; after shim, 5.150445 exactly.
    if not getattr(klass, "_dqeval_rope_init", False):
        _orig_fp = klass.from_pretrained.__func__

        def _from_pretrained(cls, *a, **kw):
            model = _orig_fp(cls, *a, **kw)
            for m in model.modules():
                if "RotaryEmbedding" in type(m).__name__ and hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            return model

        klass.from_pretrained = classmethod(_from_pretrained)
        klass._dqeval_rope_init = True
        log.append("post-load DreamRotaryEmbedding.reset_parameters() (meta-buffer repair)")

    # (4) GenerationConfig.from_model_config lost its `if decoder_config is not
    #     model_config:` guard. DreamConfig has no sub-text-config, so under 4.57 that
    #     block never ran; under 5.13 it runs unconditionally and explodes on Dream's
    #     custom fields (eps, steps, alg, alg_temp, ...). Sampler path only.
    if not getattr(gc, "_dqeval_from_model_config", False):
        def _from_model_config(cls, model_config):
            d = {k: v for k, v in model_config.to_dict().items() if v is not None}
            d.pop("_from_model_config", None)
            g = cls.from_dict(d, return_unused_kwargs=False, _from_model_config=True)
            decoder_config = model_config.get_text_config(decoder=True)
            if decoder_config is not model_config:          # the restored 4.57 guard
                default = GenerationConfig()
                dd = decoder_config.to_dict()
                for attr in g.to_dict():
                    if attr in dd and getattr(g, attr) == getattr(default, attr, object()):
                        setattr(g, attr, dd[attr])
            if g.return_dict_in_generate is False:
                if any(getattr(g, f, False) for f in g.extra_output_flags):
                    g.return_dict_in_generate = True
            g._original_object_hash = hash(g)
            return g

        gc.from_model_config = classmethod(_from_model_config)
        gc._dqeval_from_model_config = True
        log.append("DreamGenerationConfig.from_model_config: tf4.57 guard restored")

    return log


# (merged from dqeval/families/dream/sampler_native.py)
"""Dream's own sampler: `model.diffusion_generate`.

UNUSUAL AMONG OUR FAMILIES: Dream's sampler ships INSIDE the HF repo, loaded with
the weights via `trust_remote_code` (`generation_utils.py::DreamGenerationMixin`).
So for Dream, native IS upstream -- there is nothing to vendor and nothing to drift.
That is why there is no `sampler_upstream.py` here: it would be the same object.
The pinned `third_party/Dream` checkout is still useful as a reading reference and
for their lm-eval wrapper, but the sampler itself is whatever the model revision
carries, which is the honest thing to report anyway.

It also means the compat shims are doing real work at generation time: shim 4
(`DreamGenerationConfig.from_model_config`) exists purely for this path.

THE SHIFT: Dream's sampler applies `cat([l[:, :1], l[:, :-1]])` itself, immediately
after every forward (generation_utils.py:421-422). We therefore hand it the raw
model and never `adapter.logits` -- canonicalising first would shift twice, silently.

Their unmasking strategies (`alg`) and our `DecodeConfig.order` are the same ideas
under different names; the mapping is explicit below rather than guessed.
"""


# our vocabulary -> theirs. 'origin' is their default: sample per position, unmask
# by raw confidence. 'maskgit_plus' is the low-confidence-first ordering.
ORDER_TO_ALG = {
    "low_confidence": "maskgit_plus",
    "entropy": "entropy",
    "topk_margin": "topk_margin",
    "random": "origin",
}


@torch.no_grad()
def generate_native(adapter: ModelAdapter, prompt_ids: torch.Tensor, cfg: DecodeConfig, *,
             alg: str | None = None) -> GenOutput:
    """Dream `diffusion_generate`. `alg` overrides the ORDER_TO_ALG mapping."""
    if cfg.commit != "static":
        raise ValueError("Dream's sampler has no dynamic-commit policy (commit='static')")
    if alg is None:
        if cfg.order not in ORDER_TO_ALG:
            raise ValueError(f"no Dream alg for order={cfg.order!r}; pass alg= explicitly")
        alg = ORDER_TO_ALG[cfg.order]
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    # Dream decodes the whole canvas at once -- it has no block structure, so the
    # step budget is per-generation, not per-block.
    steps = cfg.steps_per_block * cfg.num_blocks

    out = adapter.model.diffusion_generate(
        prompt_ids,
        max_new_tokens=cfg.gen_length,
        steps=steps,
        temperature=cfg.temperature,
        top_p=cfg.top_p if cfg.top_p < 1.0 else None,
        top_k=cfg.top_k if cfg.top_k > 0 else None,
        alg=alg,
        alg_temp=0.0,
        mask_token_id=adapter.mask_id,
        return_dict_in_generate=True,
    )
    seq = out.sequences if hasattr(out, "sequences") else out
    gen_ids = seq[0, prompt_ids.shape[1]:]
    return GenOutput(prompt_ids=prompt_ids[0], gen_ids=gen_ids,
                     text=adapter.decode(gen_ids), n_forward=steps)
