"""SDAR adapter (JetLM/SDAR-*-Chat).

SDAR is a block-diffusion model built on a Qwen3 fork. Its head predicts the token
AT each position (it was retrained with a block-diffusion objective), so raw output
is position-aligned and `_canonicalize` is the identity -- confirmed by reading
their `block_diffusion_generate`, which samples `model(cur_x).logits` directly with
no reindexing.

Two SDAR-specific wrinkles handled in compat.py:
  * their HF repo is missing a module it imports (fails under tf4 too);
  * `auto_map["AutoModel"]` points at SDARModel, which has NO lm_head -- so we must
    resolve AutoModelForCausalLM or we would silently be reading hidden states
    instead of logits.
"""

from __future__ import annotations

from typing import Optional

import torch

from dqeval import models as hf
from dqeval.models import ModelAdapter, ModelSpec



class SDARAdapter(ModelAdapter):
    family = "sdar"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_ids=input_ids, **kw).logits

    # _canonicalize: identity (inherited) -- SDAR needs no shift.


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> SDARAdapter:
    cfg, klass = resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    shims = apply_shims(cfg, klass)
    tok = hf.tokenizer(spec.repo, revision=revision)
    model = hf.materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    shims.append(repair_rope(model))
    hf.assert_finite_rope(model)

    if not hasattr(model, "lm_head"):
        raise RuntimeError(
            f"{spec.repo}: resolved class {klass.__name__} has no lm_head -- "
            "auto_class must be AutoModelForCausalLM, not AutoModel"
        )
    return SDARAdapter(model, tok, spec, revision=revision, shims=shims)


# ==========================================================================
# (merged from dqeval/families/sdar/compat.py)
# ==========================================================================

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


def apply_shims(cfg, klass) -> list[str]:
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


# ==========================================================================
# (merged from dqeval/families/sdar/sampler_native.py)
# ==========================================================================

"""SDAR's own sampler, vendored from `generate.py` (JetAstra/SDAR @ 6a12cdb).

Their sampler lives in their GitHub repo, not the HF repo -- but it runs against the
HF weights unmodified, because the shipped `modeling_sdar.py` supports the custom
`store_kv=` forward kwarg it depends on.

TOUCHUPS -- the complete list:

  1. GREEDY. This is the significant one. Upstream's
     `sample_with_temperature_topk_topp` does `logits / temperature` then
     `torch.multinomial`, unconditionally -- so `temperature=0` divides by zero and
     greedy is UNREACHABLE in their script. Yet SDAR's published evaluation setup is
     greedy (`do_sample=False`, README + eval_sdar_lmdeploy.py). We add an argmax
     branch at temperature 0.
     ==> This path has NO upstream counterpart, so it CANNOT be parity-tested.
         Bracket it with `--sampler upstream` at a small temperature instead.
  2. Signature adapted to the dqeval contract.
  3. `entropy_bounded` is NOT implemented. Upstream's branch is broken: it computes
     "entropy" from `x0_p` (the scalar probability of the sampled token, not a
     distribution) and references `transfer_index` before assignment. Raising is
     more honest than reproducing a bug we would then have to explain.
  4. BLOCK MASK. Their `generate.py` passes a plain 3-D bool tensor as
     `attention_mask`, which `modeling_sdar.py:76` forwards to
     `flex_attention(block_mask=...)` -- but that argument requires a `BlockMask`
     object, so upstream's generate.py CANNOT RUN AS SHIPPED (it dies with
     "'Tensor' object has no attribute 'BLOCK_SIZE'", in their native environment
     too). We build the equivalent BlockMask with `create_block_mask`, copying the
     lambda from SDAR's OWN training path (modeling_sdar.py:798-803) so the
     construction is theirs, not invented.

  5. *** CURRENTLY BLOCKED under transformers 5.13. *** The prefill runs but stores
     no KV (`len(past_key_values) == 0` afterwards), so `past_key_values` is not
     reaching SDAR's attention module; every later block then sees `kv_len == q_len`
     while its block-causal mask expects the full prefix, and flex_attention rejects
     the shape. Diagnosed, not guessed -- measured directly after prefill.
     The fault is in SDAR's remote code meeting tf-5 cache plumbing (their attention
     also does `past_key_value[self.layer_idx]`, and DynamicCache is no longer
     subscriptable in 5.13). It needs its own compat shim, analogous to Dream's
     `from_model_config` shim. Until then `generate()` raises with this diagnosis,
     and this shim is the only path to SDAR generation.

WHICH THRESHOLD: `confidence_threshold` defaults to 0.85 in their generate.py, 0.75
in JetEngine, and 0.9 in the LMDeploy path that produced their published table. We
take it from DecodeConfig (the published LMDeploy value is 0.9). See
third_party/LOCKFILE.md.

This sampler calls `adapter.model` directly rather than `adapter.raw_logits`, because
it needs SDAR's extended forward signature (attention_mask as a block-causal 3-D
mask, position_ids, past_key_values, store_kv). Native samplers are allowed to reach
into their family's specifics; that is what the native/portable split is for.
"""


import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from dqeval.models import ModelAdapter
from dqeval.samplers import GenOutput
from dqeval.samplers import DecodeConfig

NEG_INF = float("-inf")


def _top_k_logits(logits, k):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    return torch.where(logits < values[..., -1, None],
                       torch.full_like(logits, NEG_INF), logits)


def _top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    idx = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                        -1, sorted_indices, mask)
    return logits.masked_fill(idx, NEG_INF)


def _sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Upstream `sample_with_temperature_topk_topp`, plus the greedy branch."""
    shape, vocab = logits.shape[:-1], logits.shape[-1]
    logits = logits.reshape(-1, vocab)

    if temperature == 0.0:
        # TOUCHUP 1 -- not upstream. Confidence still comes from the untempered
        # softmax, matching what upstream would compute at temperature 1.
        probs = F.softmax(logits, dim=-1)
        token = probs.argmax(dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        if top_k > 0:
            logits = _top_k_logits(logits, top_k)
        if top_p < 1.0:
            logits = _top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)

    token_prob = torch.gather(probs, -1, token)
    return token.view(*shape), token_prob.view(*shape)


def _as_block_mask(mask_bool: torch.Tensor):
    """3-D bool tensor -> flex_attention BlockMask (TOUCHUP 4).

    The lambda is copied verbatim from SDAR's own training path,
    modeling_sdar.py:798-803, so the construction is theirs.
    """
    from torch.nn.attention.flex_attention import create_block_mask
    return create_block_mask(
        lambda b, h, q_idx, kv_idx: mask_bool[b, q_idx, kv_idx],
        B=mask_bool.size(0), H=None,
        Q_LEN=mask_bool.size(1), KV_LEN=mask_bool.size(2),
        device=mask_bool.device.type,
        _compile=False,
    )


def _num_transfer_tokens(block_length, steps):
    """Verbatim upstream generate.py:49-54."""
    base, remainder = block_length // steps, block_length % steps
    out = torch.zeros(steps, dtype=torch.int64) + base
    out[:remainder] += 1
    return out


@torch.no_grad()
def generate_native(adapter: ModelAdapter, prompt_ids: torch.Tensor,
             cfg: DecodeConfig) -> GenOutput:
    """SDAR `block_diffusion_generate`, body faithful. `prompt_ids` is [1, L]."""
    if cfg.order == "entropy":
        raise NotImplementedError(
            "upstream's entropy_bounded branch is broken (entropy computed from a "
            "scalar probability; transfer_index used before assignment). Use the "
            "unified sampler for entropy ordering."
        )
    strategy = {
        ("low_confidence", "static"): "low_confidence_static",
        ("low_confidence", "dynamic"): "low_confidence_dynamic",
        ("sequential", "static"): "sequential",
    }.get((cfg.order, cfg.commit))
    if strategy is None:
        raise ValueError(f"SDAR has no strategy for order={cfg.order!r} commit={cfg.commit!r}")
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    model = adapter.model
    mask_id = adapter.mask_id
    dev = adapter.device
    block_length = cfg.block_length
    prompt_length = prompt_ids.shape[1]

    past_key_values = DynamicCache()
    num_blocks = (prompt_length + cfg.gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # block-causal 3-D mask: block i attends to blocks <= i, fully within a block
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=dev, dtype=torch.bool))
    attn = (block_mask.repeat_interleave(block_length, dim=0)
                      .repeat_interleave(block_length, dim=1).unsqueeze(0))
    position_ids = torch.arange(total_length, device=dev).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=dev)
    x[:, :prompt_length] = prompt_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    n_forward = 0

    if prefill_length > 0:
        model(x[:, :prefill_length],
              attention_mask=_as_block_mask(attn[:, :prefill_length, :prefill_length]),
              position_ids=position_ids[:, :prefill_length],
              past_key_values=past_key_values, use_cache=True, store_kv=True)
        n_forward += 1

        # KNOWN BLOCKER -- see module docstring item 5. Detect it here rather than
        # letting it surface 40 frames deep as an opaque flex_attention shape error.
        if len(past_key_values) == 0:
            raise NotImplementedError(
                "SDAR native sampler is blocked under transformers 5.13: the prefill "
                "ran but stored no KV (len(cache)==0), so past_key_values is not "
                "reaching SDAR's attention module and every later block sees "
                "kv_len == q_len while its mask expects the full prefix.\n"
                "Root cause is in SDAR's remote code, not ours, and it needs its own "
                "compat shim (analogous to Dream's from_model_config shim)."
            )

    schedule = _num_transfer_tokens(block_length, cfg.steps_per_block)

    for nb in range(prefill_blocks, num_blocks):
        lo, hi = nb * block_length, (nb + 1) * block_length
        cur = x[:, lo:hi].clone()
        cur_attn = _as_block_mask(attn[:, lo:hi, :hi])
        cur_pos = position_ids[:, lo:hi]

        for step in range(cfg.steps_per_block + 1):
            mask_index = cur == mask_id
            if mask_index.sum() == 0:
                model(cur, attention_mask=cur_attn, position_ids=cur_pos,
                      past_key_values=past_key_values, use_cache=True, store_kv=True)
                n_forward += 1
                break

            logits = model(cur, attention_mask=cur_attn, position_ids=cur_pos,
                           past_key_values=past_key_values, use_cache=True,
                           store_kv=False).logits
            n_forward += 1

            x0, x0_p = _sample(logits, temperature=cfg.temperature,
                               top_k=cfg.top_k, top_p=cfg.top_p)
            k = int(schedule[min(step, len(schedule) - 1)])

            if strategy == "sequential":
                transfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur.shape[0]):
                    first = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                    transfer[j, first:first + k] = True
            elif strategy == "low_confidence_static":
                conf = torch.where(mask_index, x0_p, torch.tensor(NEG_INF, device=dev))
                transfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    transfer[j, torch.topk(conf[j], k).indices] = True
            else:  # low_confidence_dynamic
                conf = torch.where(mask_index, x0_p, torch.tensor(NEG_INF, device=dev))
                transfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    high = conf[j] > cfg.confidence_threshold
                    if int(high.sum()) >= k:
                        transfer[j] = high
                    else:
                        transfer[j, torch.topk(conf[j], k).indices] = True

            cur[transfer] = x0[transfer]

        x[:, lo:hi] = cur

    gen_ids = x[0, prompt_length:prompt_length + cfg.gen_length]
    return GenOutput(prompt_ids=prompt_ids[0], gen_ids=gen_ids,
                     text=adapter.decode(gen_ids), n_forward=n_forward)
