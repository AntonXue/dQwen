"""CoDA adapter -- Salesforce's masked DLM adapted from Qwen3-1.7B.

WHY IT MATTERS: same backbone as our dQwen3-1.7B control (confirmed against
its config), so CoDA is the closest available recipe-isolating comparison --
it differs in data and budget only.

Second AR-aligned family (Dream is the other): their code shifts with the
same one-line cat after every forward (generation_utils.py), and
`_canonicalize` applies it (double-shift hazard: see the adapter contract).

Facts read out of their source, not assumed:
  * genuinely BIDIRECTIONAL -- forward builds a causal mask but the attention
    module never passes it and hardcodes is_causal=False; the mask is dead code
  * consequently padding masks are IGNORED: batching would silently attend
    across padding; bs=1 (which we are by construction) is required here
  * forward returns a plain (logits, loss) tuple and only takes the inference
    branch in eval mode (materialize calls .eval())
"""

import sys
from typing import Optional

import torch

from .adapter import (ModelAdapter, ModelSpec, assert_finite_rope,
                      materialize, resolve, tokenizer)


class CoDAAdapter(ModelAdapter):
    family = "coda"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, **kw)
        return out[0] if isinstance(out, tuple) else out.logits

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """Their inference-time shift, verbatim (generation_utils.py)."""
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> CoDAAdapter:
    cfg, klass = resolve(spec.repo, revision=revision)
    shims = apply_shims(cfg, klass)          # MUST precede materialisation
    tok = tokenizer(spec.repo, revision=revision)
    model = materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    assert_finite_rope(model)

    declared = getattr(cfg, "mask_token_id", None)
    if declared is not None and declared != spec.mask_id:
        raise ValueError(
            f"{spec.repo}: config mask_token_id={declared} != registry {spec.mask_id}")

    # UNTIED-HEAD GATE: CoDA ships a trained lm_head, but its config omits
    # tie_word_embeddings and tf5 defaults it True (shim 3 restores False).
    # A regression makes every logit wrong with nothing raised -- reject
    # shared storage AND bitwise-equal values.
    head, emb = model.lm_head.weight, model.model.embed_tokens.weight
    if head.data_ptr() == emb.data_ptr():
        raise RuntimeError(f"{spec.repo}: lm_head is TIED to embed_tokens (shares storage); "
                           "CoDA is untied -- compat shim 3 has regressed")
    if torch.equal(head, emb):
        raise RuntimeError(f"{spec.repo}: lm_head equals embed_tokens bitwise; "
                           "the trained head was overwritten by tying")
    return CoDAAdapter(model, tok, spec, revision=revision, shims=shims)


# Compat shims for transformers 5.13. PRINCIPLE: restore verified 4.x
# behaviour, never invent. Shim 1 is documented prior art (ADLMC
# coda/old/environment.md, verified on tf 4.57.1). Shim 2 is the
# meta-buffer trap (see adapter.assert_finite_rope): CoDA has both
# preconditions, and its rotary has no reset_parameters(), so the repair
# recomputes inv_freq with CoDA's OWN default_rope_frequencies.
# NOT needed here (recorded so nobody re-adds it): the tf5 rope_theta ->
# rope_parameters migration -- CoDA ships its own rotary class.


def apply_shims(cfg, klass) -> list[str]:
    log: list[str] = []
    mod = sys.modules[klass.__module__]

    # (1) transformers PR #39423 (v4.54.0, breaking, no deprecation period) replaced
    #     the lenient `_autoset_attn_implementation` with a strict init-time
    #     `_sdpa_can_dispatch` check. CoDA has a two-class layout: the OUTER
    #     CoDALanguageModel declares _supports_sdpa, the INNER CoDAModel does not,
    #     so the outer init passes and then explodes constructing the inner:
    #       ValueError: CoDAModel does not support an attention implementation
    #       through torch.nn.functional.scaled_dot_product_attention yet.
    #     On 4.49 the old path silently fell back to eager and CoDA loaded fine.
    #     Must be set BEFORE from_pretrained. CoDA's attention calls SDPA with
    #     is_causal=False explicitly (attention.py), so declaring support is
    #     accurate, not a lie to get past a check.
    inner = getattr(mod, "CoDAModel", None)
    if inner is not None and not getattr(inner, "_supports_sdpa", False):
        inner._supports_sdpa = True
        log.append("CoDAModel._supports_sdpa = True (restores pre-4.54 dispatch)")

    # (2) tf5 standardises rope parameters on PreTrainedConfig and SYNTHESISES a
    #     `rope_scaling` dict carrying `rope_type`. The shipped CoDA config.json has
    #     `"rope_scaling": null`, so under 4.x this branch in CoDAModel.__init__ was
    #     DEAD:
    #         if rope_scaling is not None: rope_scaling = RopeScaling(**rope_scaling)
    #     Under tf5 it fires and CoDA's 4-field RopeScaling dataclass (factor,
    #     low_freq_factor, high_freq_factor, original_context_len) rejects rope_type.
    #     Restoring None is provably the right repair, not merely the convenient one:
    #     CoDARotaryEmbedding raises NotImplementedError("Scaling is not implemented")
    #     for ANY non-None scaling, so None is the only path that has ever worked.
    #     (ADLMC/coda/old/environment.md judged the rope migration harmless for CoDA;
    #     that was verified on 4.57, and it does NOT hold on 5.x.)
    _FIELDS = {"factor", "low_freq_factor", "high_freq_factor", "original_context_len"}
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict) and not set(rs) <= _FIELDS:
        cfg.rope_scaling = None
        log.append(f"rope_scaling {sorted(rs)} -> None (tf5 synthesised; checkpoint ships null)")

    # (3) CoDA is UNTIED -- its checkpoint ships a real `lm_head.weight` (311 tensors)
    #     -- but its config.json OMITS `tie_word_embeddings`, and tf5's PreTrainedConfig
    #     defaults that to True. Under 4.x this was harmless because tying happens in
    #     `post_init()` and CoDALanguageModel.__init__ NEVER CALLS post_init (it ends at
    #     `self.apply(self._init_weights)`); only the inner CoDAModel does. Under tf5 the
    #     loader consults tie state directly, so an unrestored default would overwrite the
    #     trained head with the embedding matrix -- silent, and catastrophic for logits.
    #     Asserted after load in adapter.build(): lm_head must NOT share storage with
    #     embed_tokens and must match the checkpoint.
    if getattr(cfg, "tie_word_embeddings", None):
        cfg.tie_word_embeddings = False
        log.append("tie_word_embeddings True -> False (CoDA is untied; tf5 default, absent from config.json)")

    # (4) tf5's `_finalize_model_loading` reads `self.all_tied_weights_keys`, which the
    #     base class populates in `post_init()`. CoDALanguageModel never calls post_init,
    #     so the attribute is absent and loading dies with AttributeError. 4.x had no such
    #     attribute and no such read, so an EMPTY mapping is the exact restoration: this
    #     model registers no tied weights (see shim 3). We do NOT call post_init() itself
    #     -- that would additionally run tie_weights and gradient-checkpointing setup that
    #     4.x never ran for this class.
    if not getattr(klass, "_dqeval_tied_keys", False):
        _orig_init = klass.__init__

        def _init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        klass.__init__ = _init
        klass._dqeval_tied_keys = True
        log.append("all_tied_weights_keys={} on CoDALanguageModel (post_init never called in 4.x either)")

    # (5) THE SILENT ONE -- see module docstring. Repair must happen AFTER
    #     materialisation; hooking _init_weights does not work (still on meta).
    if not getattr(klass, "_dqeval_rope_init", False):
        _orig_fp = klass.from_pretrained.__func__

        def _from_pretrained(cls, *a, **kw):
            model = _orig_fp(cls, *a, **kw)
            for m in model.modules():
                if type(m).__name__ != "CoDARotaryEmbedding":
                    continue
                buf = getattr(m, "inv_freq", None)
                if buf is None:
                    continue
                # ALWAYS recompute: what a meta-materialised buffer holds is
                # torch's choice -- garbage (non-finite) on 2.7/x86, ZEROS on
                # 2.10/aarch64. Zeros pass every finiteness predicate and are
                # position-blindness (the 2026-08-14 silent CoDA campaign row);
                # a brokenness test cannot be trusted, recomputation is cheap
                # pure config data.
                head_dim = getattr(model.config, "head_dim", None) or (
                    model.config.hidden_size // model.config.num_attention_heads)
                fresh = mod.default_rope_frequencies(
                    head_dim, theta=model.config.rope_theta)
                m.inv_freq = fresh.to(device=buf.device, dtype=buf.dtype)
            return model

        klass.from_pretrained = classmethod(_from_pretrained)
        klass._dqeval_rope_init = True
        log.append("post-load CoDARotaryEmbedding.inv_freq recompute (meta-buffer repair)")

    return log
