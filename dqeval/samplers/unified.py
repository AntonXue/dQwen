"""The portable sampler: one block-diffusion engine, any family.

Deliberately basic for now -- correctness first, generality later. It talks to a
model only through `adapter.logits(ids)`, so it runs unchanged on LLaDA, Dream,
SDAR and dQwen weights. That is the whole point of the adapter contract.

BATCH SIZE 1, by construction. Not a limitation we impose but the regime the
reference implementations live in: LLaDA's `generate.py` and SDAR's
`block_diffusion_generate` both hardcode `torch.full((1, ...))`. Batching them
would itself be a divergence from upstream with no counterpart to parity-test
against. It also sidesteps our own champion decode's batch non-invariance, where
a problem's canvas position depends on its batchmates (measured 2-5pp swings).
Parallelism belongs above this function: map over prompts, shard over GPUs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dqeval.adapter import GenOutput, ModelAdapter
from dqeval.config import DecodeConfig

NEG_INF = float("-inf")


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, NEG_INF)


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    ordered, idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
    drop = cum > p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    return logits.masked_fill(drop.scatter(-1, idx, drop), NEG_INF)


def _propose(logits: torch.Tensor, cfg: DecodeConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position token proposal + its confidence. `logits` is [T, V].

    Confidence is read from the softmax of the UNMODIFIED logits (LLaDA's
    convention) so that the unmasking order does not silently depend on the
    temperature/top-k/top-p filtering applied to the sampling distribution.
    """
    conf_all = F.softmax(logits.float(), dim=-1)
    if cfg.greedy:
        x0 = logits.argmax(dim=-1)
    else:
        filtered = _apply_top_p(_apply_top_k(logits.float() / cfg.temperature, cfg.top_k),
                                cfg.top_p)
        x0 = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1).squeeze(-1)
    return x0, conf_all.gather(-1, x0.unsqueeze(-1)).squeeze(-1)


def _score(logits: torch.Tensor, x0: torch.Tensor, conf: torch.Tensor,
           cfg: DecodeConfig) -> torch.Tensor:
    """Higher score == unmask earlier."""
    if cfg.order == "low_confidence":
        return conf
    if cfg.order == "random":
        return torch.rand_like(conf)
    if cfg.order == "sequential":
        return torch.arange(conf.numel(), 0, -1, device=conf.device, dtype=conf.dtype)
    p = F.softmax(logits.float(), dim=-1)
    if cfg.order == "entropy":
        return (p * p.clamp_min(1e-12).log()).sum(-1)      # negative entropy
    if cfg.order == "topk_margin":
        top2 = p.topk(2, dim=-1).values
        return top2[:, 0] - top2[:, 1]
    raise ValueError(f"unknown order {cfg.order!r}")


def _transfer_schedule(n_masked: int, steps: int, device) -> torch.Tensor:
    """How many positions to commit at each step -- LLaDA's even split."""
    base, rem = divmod(n_masked, steps)
    out = torch.full((steps,), base, dtype=torch.long, device=device)
    out[:rem] += 1
    return out


def _stop_cut(text: str, stop_strings) -> "int | None":
    """Index of the earliest stop-string occurrence in text, or None."""
    if not stop_strings:
        return None
    hits = [text.find(s) for s in stop_strings if s]
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else None


@torch.no_grad()
def generate(adapter: ModelAdapter, prompt_ids: torch.Tensor,
             cfg: DecodeConfig, stop_strings=None) -> GenOutput:
    """Block-diffusion decode of ONE prompt. `prompt_ids` is [1, L].

    `stop_strings`: in append mode, after each block the generated text is checked
    for these; the first hit ends decoding and truncates the output there. Because
    a masked DLM does not self-terminate the way an AR model emits EOS, this is what
    keeps generative tasks (humaneval's `\\ndef`/`\\nclass`, gsm8k's `\\n\\n`) from
    running to the full canvas and ending mid-statement. It also saves the forwards
    that would fill the rest of the canvas -- a large speedup at block_length=1.
    Only meaningful in append mode (full/window see the whole canvas at once).
    """
    if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1:
        raise ValueError(f"bs=1 only; got prompt_ids of shape {tuple(prompt_ids.shape)}")
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    dev = adapter.device
    mask_id = adapter.mask_id
    p_len = prompt_ids.size(1)

    canvas = torch.full((1, p_len + cfg.gen_length), mask_id, dtype=torch.long, device=dev)
    canvas[:, :p_len] = prompt_ids
    n_forward = 0
    stop_text = None       # set when a stop string is hit in append mode

    for b in range(cfg.num_blocks):
        lo = p_len + b * cfg.block_length
        hi = lo + cfg.block_length
        # "append" grows the canvas block by block; "full"/"window" see it all.
        end = hi if cfg.mode == "append" else canvas.size(1)

        n_masked = int((canvas[0, lo:hi] == mask_id).sum())
        schedule = _transfer_schedule(n_masked, cfg.steps_per_block, dev)

        for step in range(cfg.steps_per_block):
            masked = canvas[0, lo:hi] == mask_id
            if not masked.any():
                break

            logits = adapter.logits(canvas[:, :end])[0, lo:hi].clone()
            n_forward += 1
            logits[:, mask_id] = NEG_INF      # never propose MASK itself

            x0, conf = _propose(logits, cfg)
            score = _score(logits, x0, conf, cfg).masked_fill(~masked, NEG_INF)

            if cfg.commit == "dynamic":
                take = (conf > cfg.confidence_threshold) & masked
                if int(take.sum()) < int(schedule[step]):
                    take = torch.zeros_like(masked)
                    take[score.topk(int(schedule[step])).indices] = True
            else:
                k = int(min(schedule[step], masked.sum()))
                take = torch.zeros_like(masked)
                if k > 0:
                    take[score.topk(k).indices] = True

            blk = canvas[0, lo:hi]
            blk[take] = x0[take]
            canvas[0, lo:hi] = blk

        # early stop: after a block completes, blocks 0..b are a contiguous revealed
        # prefix in BOTH append and full mode (both reveal blocks left-to-right), so
        # checking the decoded prefix for a stop string works for either. In full
        # mode this also avoids denoising the trailing blocks once the answer ends.
        if stop_strings and cfg.mode in ("append", "full"):
            gen_text = adapter.decode(canvas[0, p_len:hi])
            cut = _stop_cut(gen_text, stop_strings)
            if cut is not None:
                stop_text = gen_text[:cut]
                break

    gen_ids = canvas[0, p_len:]
    text = adapter.decode(gen_ids) if stop_text is None else stop_text
    return GenOutput(
        prompt_ids=prompt_ids[0],
        gen_ids=gen_ids,
        text=text,
        n_forward=n_forward,
    )
