"""Decode configuration: the one knob space every sampler is parameterised by.

The claim this file encodes is that LLaDA, Dream and dQwen do not need
different decode APIs -- their published samplers are points in the space
below, and the parity tests (`tests/test_llada_sampler.py`, `tests/test_nelbo_vs_llada.py`) holds our engine token-identical to theirs at
the matching configuration.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F


Order = Literal["low_confidence", "entropy", "topk_margin", "random", "sequential"]
Commit = Literal["static", "dynamic"]


@dataclass(frozen=True)
class DecodeConfig:
    """All decode knobs. Samplers read what they need and ignore the rest.

    Note `temperature == 0.0` means ARGMAX, never `logits / 0`. Several upstream
    reference scripts divide unconditionally and so cannot express greedy at all
    (SDAR's `generate.py` is the clearest case) -- we treat greedy as a first-class
    branch instead, and record it as a touchup wherever it diverges from upstream.
    """

    # canvas
    # Decoding is always full-canvas: the whole answer canvas is visible from
    # the start and blocks are revealed left-to-right (LLaDA/Dream native).
    gen_length: int = 1024   # LLaDA-style default answer canvas
    block_length: int = 32          # semi-AR granularity within the canvas
    # schedule
    steps_per_block: int = 32
    # token sampling
    temperature: float = 0.0        # 0.0 => argmax
    top_k: int = 0                  # 0 => disabled
    top_p: float = 1.0              # 1.0 => disabled
    # unmasking order
    order: Order = "low_confidence"
    # commit policy
    commit: Commit = "static"
    confidence_threshold: float = 0.9   # only read when commit == "dynamic"
    # guidance
    cfg_scale: float = 0.0          # classifier-free guidance (LLaDA native)
    # reproducibility
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.gen_length <= 0 or self.block_length <= 0:
            raise ValueError("gen_length and block_length must be positive")
        if self.gen_length % self.block_length != 0:
            raise ValueError(
                f"gen_length ({self.gen_length}) must be a multiple of "
                f"block_length ({self.block_length})"
            )
        if self.steps_per_block <= 0:
            raise ValueError("steps_per_block must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0 (0 == greedy)")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def num_blocks(self) -> int:
        return self.gen_length // self.block_length

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


@dataclass
class GenOutput:
    """One generation. bs=1 by construction -- see `generate` below."""

    prompt_ids: torch.Tensor
    gen_ids: torch.Tensor
    text: str
    n_forward: int = 0
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
def generate(adapter, prompt_ids: torch.Tensor,
             cfg: DecodeConfig, stop_strings=None) -> GenOutput:
    """Block-diffusion decode of ONE prompt. `prompt_ids` is [1, L].

    `stop_strings`: after each block completes, the decoded prefix (contiguous,
    since blocks reveal left-to-right) is checked for these; the first hit ends
    decoding and truncates the output there. Because a masked DLM does not
    self-terminate the way an AR model emits EOS, this is what keeps generative
    tasks (humaneval's `\\ndef`/`\\nclass`, gsm8k's `\\n\\n`) from running to the
    full canvas and ending mid-statement -- and it saves the forwards that would
    denoise the rest of the canvas (why HumanEval measures ~110 forwards, not
    gen_length).
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
    stop_text = None       # set when a stop string is hit

    for b in range(cfg.num_blocks):
        lo = p_len + b * cfg.block_length
        hi = lo + cfg.block_length
        end = canvas.size(1)   # the model always sees the whole canvas

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

        # early stop: after a block completes, blocks 0..b are a contiguous
        # revealed prefix, so checking it for a stop string is sound -- and it
        # avoids denoising the trailing blocks once the answer has ended.
        if stop_strings:
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


"""Monte-Carlo NELBO log-likelihood for masked diffusion LMs.

A masked DLM has no AR chain-rule likelihood, so "loglikelihood" is a MODELLING
CHOICE, not a well-defined quantity. LLaDA, Dream and Dream-Coder all use the
MC-NELBO estimator below, so we use it too -- that is what makes our multiple-choice
numbers comparable to theirs rather than merely adjacent to them.

    log p(answer | prompt)  ~=  - E_t [ (1/p_mask) * sum_{i masked} CE_i ]

over `mc_num` Monte-Carlo mask draws. `mc_num=1` is EXACT for single-token targets
(MMLU/ARC letters, per LLaDA App. B.5); ~128 for multi-token.

`forward_process` is a BYTE-FAITHFUL copy of LLaDA's reference implementation --
same RNG operations in the same order -- so a seeded run reproduces their number
exactly. That is not stylistic fidelity, it is the correctness gate: verified to
reproduce LLaDA-8B-Base at -40.403095 vs -40.403095, diff 0.00e+00.
See tests/test_nelbo_vs_llada.py.

NO eval-side logit shift is applied here. The estimator reads the masked-position
logit directly, which requires POSITION-ALIGNED logits -- so callers must pass
`adapter.logits` (canonical), never `adapter.raw_logits`. For Dream that means the
shift is applied exactly once, which is precisely what their own eval wrapper does
before computing loglikelihood (eval/eval.py:354).

Originally `ablations/paper_evals/mc_nelbo.py` in the ADLMC research repo.
"""


def forward_process(batch: torch.Tensor, prompt_index: torch.Tensor, mask_id: int):
    """Byte-faithful copy of LLaDA `get_log_likelihood.py:7-26`. Do not 'clean up'.

    The exact sequence of RNG calls is load-bearing: it is what lets a seeded run
    of this file reproduce LLaDA's published estimator bit-for-bit. Any refactor
    that reorders `torch.randint` / `torch.randperm` silently breaks the gate.
    """
    b, l = batch.shape
    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(
        torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b,
                       device=batch.device)
    ).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat(
        (torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device),
         is_mask),
        dim=1,
    )
    noisy_batch = torch.where(is_mask, mask_id, batch)
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


@torch.no_grad()
def mc_nelbo_loglikelihood(logits_fn: Callable[[torch.Tensor], torch.Tensor],
                           prompt: torch.Tensor, answer: torch.Tensor, *,
                           mc_num: int = 128, batch_size: int = 16,
                           mask_id: int = 248061) -> float:
    """log p(answer | prompt). `prompt` and `answer` are 1-D LongTensors.

    `logits_fn(input_ids[B, L]) -> logits[B, L, V]`, position-aligned.
    cfg_scale = 0 path (LLaDA's default).
    """
    if mc_num % batch_size != 0:
        raise ValueError(f"mc_num ({mc_num}) must be divisible by batch_size ({batch_size})")

    seq = torch.cat([prompt, answer])[None, :].repeat(batch_size, 1)
    prompt_index = torch.arange(seq.shape[1], device=seq.device) < len(prompt)

    losses = []
    for _ in range(mc_num // batch_size):
        noisy, p_mask = forward_process(seq, prompt_index, mask_id)
        m = noisy == mask_id
        logits = logits_fn(noisy)
        loss = F.cross_entropy(logits[m], seq[m], reduction="none") / p_mask[m]
        losses.append((loss.sum() / batch_size).item())
    return -sum(losses) / len(losses)
