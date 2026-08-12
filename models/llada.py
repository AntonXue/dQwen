"""LLaDA adapter.

LLaDA is a pure masked diffusion LM: its raw output is already position-aligned
(`logits[:, i]` is the distribution for position i), so `_canonicalize` is the
inherited identity and there is no eval-side shift. Confirmed by reading their
`generate.py`, which consumes `model(x).logits` directly with no reindexing.
"""

from typing import Optional
import torch
from .adapter import (ModelAdapter, ModelSpec,
                           assert_finite_rope, materialize, resolve, tokenizer)
import torch.nn.functional as F
from .adapter import ModelAdapter
from .samplers import GenOutput
from .samplers import DecodeConfig
from .adapter import on_path


class LLaDAAdapter(ModelAdapter):
    family = "llada"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_ids=input_ids, **kw).logits

    # _canonicalize: identity (inherited) -- LLaDA needs no shift.


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> LLaDAAdapter:
    cfg, klass = resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    shims = apply_shims(cfg, klass)
    tok = tokenizer(spec.repo, revision=revision)
    model = materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    assert_finite_rope(model)
    return LLaDAAdapter(model, tok, spec, revision=revision, shims=shims)


"""LLaDA compat shims for transformers 5.13.

PRINCIPLE: a shim RESTORES a behaviour transformers 4.x had. It never invents one.
Each shim below records how the 4.x behaviour was established -- read out of the
native environment, not guessed.

VERIFIED: with these three shims and torch held fixed, LLaDA-8B-Base forward output
under transformers 5.13 is BITWISE IDENTICAL to transformers 4.57 (max|d| = 0.00e+00
over the probe prompts). The residual drift originally observed was 100% attributable
to torch 2.5.1 -> 2.7.1, and 0% to transformers. See the port-evidence _claude doc (20260721-184634).
"""


_APPLIED = "_dqeval_llada_shims"


def apply_shims(cfg, klass) -> list[str]:
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


"""LLaDA's own sampler, vendored from `generate.py` (ML-GSAI/LLaDA @ 96441d4).

This is the "sampler shipped with LLaDA" path: the algorithm is reproduced exactly,
so a number produced here is theirs, not ours. `tests/test_llada_sampler.py`
runs this against the unmodified upstream file and requires token-identical output.

TOUCHUPS -- the complete list, each one behaviour-preserving:

  1. `-np.inf` -> `float("-inf")`. Identical value (np.inf is a plain Python float);
     drops the numpy dependency so the repo stays torch-only.
  2. Signature adapted to the dqeval sampler contract
     `generate(adapter, prompt_ids, cfg) -> GenOutput`. The body is untouched.
  3. Logits come from `adapter.raw_logits` rather than `model(x).logits`. For LLaDA
     these are the same thing -- it is position-aligned and needs no shift -- but
     native samplers use raw_logits by convention, since a family that shifts
     internally (Dream) must not receive canonicalised logits.

DELIBERATELY NOT CHANGED, though our unified engine does it differently:

  * LLaDA does NOT suppress the MASK id before the argmax. Our engine does. Left
    faithful here; if the two ever diverge on a prompt, this is the first suspect.
  * Confidence is read from softmax of the RAW logits, before temperature. Faithful.
  * `add_gumbel_noise` returns logits unchanged at temperature 0, so greedy works
    without a division -- unlike SDAR's script, which cannot express greedy at all.
"""


NEG_INF = float("-inf")


def add_gumbel_noise(logits, temperature):
    """Verbatim LLaDA generate.py:8-19.

    float64 is deliberate on their side (arXiv:2409.02908: low-precision Gumbel-max
    helps perplexity but hurts generation quality). Do not downcast.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Verbatim LLaDA generate.py:22-40. Linear noise schedule => even split."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_native(adapter: ModelAdapter, prompt_ids: torch.Tensor,
                    cfg: DecodeConfig) -> GenOutput:
    """LLaDA `generate()`, body verbatim. `prompt_ids` is [1, L]."""
    if cfg.order not in ("low_confidence", "random"):
        raise ValueError(
            f"LLaDA's sampler implements 'low_confidence' and 'random' only; "
            f"got order={cfg.order!r}. Use the unified sampler for other orders."
        )
    if cfg.commit != "static":
        raise ValueError("LLaDA's sampler has no dynamic-commit policy (commit='static' only)")
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    mask_id = adapter.mask_id
    prompt = prompt_ids
    gen_length, block_length = cfg.gen_length, cfg.block_length
    steps, temperature, cfg_scale = cfg.steps_per_block, cfg.temperature, cfg.cfg_scale
    remasking = cfg.order

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long).to(adapter.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    n_forward = 0

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length:
                 prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = adapter.raw_logits(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = adapter.raw_logits(x)
            n_forward += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:  # "random"
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = NEG_INF

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, NEG_INF)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    gen_ids = x[0, prompt.shape[1]:]
    return GenOutput(prompt_ids=prompt_ids[0], gen_ids=gen_ids,
                     text=adapter.decode(gen_ids), n_forward=n_forward)


"""Escape hatch: run LLaDA's `generate.py` verbatim from the pinned checkout.

Not used for normal evals -- `sampler_native.py` is the vendored equivalent, and
the parity tests (`tests/test_llada_sampler.py`, `tests/test_nelbo_vs_llada.py`) asserts the two agree token-for-token. This exists so that claim is
checkable on demand, and so a touched-up path can be bracketed against real upstream
when it has no parity counterpart.

Requires the pinned LLaDA checkout under third_party/ (models.UPSTREAM_PINS has the commit).
"""


@torch.no_grad()
def generate_upstream(adapter: ModelAdapter, prompt_ids: torch.Tensor,
                      cfg: DecodeConfig) -> GenOutput:
    with on_path("LLaDA"):
        from generate import generate as _upstream_generate

        # NOTE the units differ, and this is exactly the sort of off-by-N a parity
        # test exists to catch: LLaDA's `steps` is the TOTAL across all blocks
        # (it does `steps = steps // num_blocks` internally), whereas DecodeConfig
        # carries steps PER BLOCK. Convert, do not pass through.
        total_steps = cfg.steps_per_block * cfg.num_blocks

        out = _upstream_generate(
            adapter.model, prompt_ids,
            steps=total_steps,
            gen_length=cfg.gen_length,
            block_length=cfg.block_length,
            temperature=cfg.temperature,
            cfg_scale=cfg.cfg_scale,
            remasking=cfg.order,          # 'low_confidence' | 'random'
            mask_id=adapter.mask_id,
        )

    gen_ids = out[0, prompt_ids.shape[1]:]
    return GenOutput(prompt_ids=prompt_ids[0], gen_ids=gen_ids,
                     text=adapter.decode(gen_ids), n_forward=total_steps)
