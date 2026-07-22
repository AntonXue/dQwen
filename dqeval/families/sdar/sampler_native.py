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
     `from_model_config` shim. Until then `generate()` raises with this diagnosis.
     WORKAROUND: the unified sampler with PRESETS["sdar_published"] runs the same
     algorithm without the KV-cache path, and works today.

WHICH THRESHOLD: `confidence_threshold` defaults to 0.85 in their generate.py, 0.75
in JetEngine, and 0.9 in the LMDeploy path that produced their published table. We
take it from DecodeConfig, and PRESETS["sdar_published"] carries 0.9. See
third_party/LOCKFILE.md.

This sampler calls `adapter.model` directly rather than `adapter.raw_logits`, because
it needs SDAR's extended forward signature (attention_mask as a block-causal 3-D
mask, position_ids, past_key_values, store_kv). Native samplers are allowed to reach
into their family's specifics; that is what the native/portable split is for.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from dqeval.adapter import GenOutput, ModelAdapter
from dqeval.config import DecodeConfig

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
def generate(adapter: ModelAdapter, prompt_ids: torch.Tensor,
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
                "compat shim (analogous to Dream's from_model_config shim).\n"
                "Use the unified sampler with PRESETS['sdar_published'] meanwhile -- "
                "it reproduces the same algorithm without the KV-cache path."
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
