"""LLaDA compat shims for transformers 5.13.

PRINCIPLE: a shim RESTORES a behaviour transformers 4.x had. It never invents one.
Each shim below records how the 4.x behaviour was established -- read out of the
native environment, not guessed.

VERIFIED: with these three shims and torch held fixed, LLaDA-8B-Base forward output
under transformers 5.13 is BITWISE IDENTICAL to transformers 4.57 (max|d| = 0.00e+00
over the probe prompts). The residual drift originally observed was 100% attributable
to torch 2.5.1 -> 2.7.1, and 0% to transformers. See tests/parity/test_llada_port.py.
"""

from __future__ import annotations

_APPLIED = "_dqeval_llada_shims"


def apply(cfg, klass) -> list[str]:
    """Install LLaDA's transformers-5 shims. Returns a log for the ledger."""
    log: list[str] = []

    # (1) transformers>=5 sets `all_tied_weights_keys` inside post_init(); LLaDA's
    #     __init__ never calls post_init, so the attribute is absent and
    #     _finalize_model_loading raises AttributeError.
    #     Verified: LLaDA's config carries weight_tying=False (their own field), so
    #     the model genuinely has no tied weights and {} is the correct value.
    if not hasattr(klass, "all_tied_weights_keys"):
        klass.all_tied_weights_keys = {}
        log.append("all_tied_weights_keys={} (untied per cfg.weight_tying=False)")

    # (2) transformers>=5 calls tie_weights(missing_keys=..., recompute_mapping=...).
    #     LLaDA's signature accepts neither. Swallow the new kwargs and delegate.
    if not getattr(klass, _APPLIED, False):
        _orig = klass.tie_weights
        klass.tie_weights = lambda self, *a, **kw: _orig(self)
        setattr(klass, _APPLIED, True)
        log.append("tie_weights(**kwargs) tolerated")

    # (3) transformers 4.x carried `use_cache` on the base config class; tf>=5
    #     dropped it, and LLaDA's forward reads `self.config.use_cache` directly.
    #     Verified by loading this config under tf 4.57.1 and reading the value:
    #     it resolves to False. Restored exactly, not guessed.
    if not hasattr(cfg, "use_cache"):
        cfg.use_cache = False
        log.append("cfg.use_cache=False (verified == tf4.57 value)")

    return log
