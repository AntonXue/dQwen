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
See tests/parity/test_nelbo_vs_llada.py.

NO eval-side logit shift is applied here. The estimator reads the masked-position
logit directly, which requires POSITION-ALIGNED logits -- so callers must pass
`adapter.logits` (canonical), never `adapter.raw_logits`. For Dream that means the
shift is applied exactly once, which is precisely what their own eval wrapper does
before computing loglikelihood (eval/eval.py:354).

Originally `ablations/paper_evals/mc_nelbo.py` in the ADLMC research repo.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


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


def loglikelihood_options(adapter, prompt_ids: torch.Tensor,
                          options: "list[torch.Tensor]", *,
                          mc_num: Optional[int] = None,
                          batch_size: int = 16) -> "list[float]":
    """Score a SET of candidate continuations with ONE estimator setting.

    USE THIS -- not repeated `loglikelihood()` calls -- whenever the scores will be
    COMPARED against each other, which is every multiple-choice task.

    Why: `loglikelihood()` chooses mc_num per continuation (1 if single-token, else
    128). For a mixed-length option set that silently scores one option EXACTLY and
    another with a 128-draw stochastic LOWER BOUND, then argmaxes across two
    different estimands. It raises nothing and yields a plausible-looking accuracy.

    Bare-letter MMLU/ARC cannot trigger it -- " A".." D" are single tokens in every
    tokenizer we support (verified across LLaDA, Dream, SDAR, dQwen). Any option set
    with punctuation, parenthesised letters, or free text can.

    Length caveat that remains even with a uniform estimator: multi-token options
    accumulate more masked-position CE terms, so longer continuations score lower
    for reasons unrelated to correctness. That is what lm-eval's `acc_norm` exists
    to compensate for; this function does not normalise, it only guarantees that
    every option was measured the same way.
    """
    if not options:
        return []
    lengths = {int(o.flatten().numel()) for o in options}
    if mc_num is None:
        # one setting for the whole set, decided by the WHOLE set, not per option
        mc_num = 1 if lengths == {1} else 128
    if mc_num == 1 and lengths != {1}:
        raise ValueError(
            f"mc_num=1 is exact only for single-token continuations, but option "
            f"lengths are {sorted(lengths)}. Pass an explicit mc_num >= 2."
        )
    bs = 1 if mc_num == 1 else min(batch_size, mc_num)
    return [
        mc_nelbo_loglikelihood(adapter.logits, prompt_ids.flatten(), o.flatten(),
                               mc_num=mc_num, batch_size=bs, mask_id=adapter.mask_id)
        for o in options
    ]


def loglikelihood(adapter, prompt_ids: torch.Tensor, answer_ids: torch.Tensor, *,
                  mc_num: Optional[int] = None, batch_size: int = 16) -> float:
    """Adapter-friendly wrapper for ONE continuation.

    For comparing several candidates, use `loglikelihood_options` instead -- see the
    estimator-mixing hazard documented there.

    Note this uses `adapter.logits` (canonical), not `raw_logits` -- see module
    docstring. Passing raw logits for Dream would be an off-by-one with no error.
    """
    prompt_ids = prompt_ids.flatten()
    answer_ids = answer_ids.flatten()
    if mc_num is None:
        # A single masked position has no Monte-Carlo variance to average away:
        # one draw is the exact marginal. LLaDA App. B.5 uses mc_num=1 here too.
        mc_num = 1 if answer_ids.numel() == 1 else 128
    bs = 1 if mc_num == 1 else min(batch_size, mc_num)
    return mc_nelbo_loglikelihood(
        adapter.logits, prompt_ids, answer_ids,
        mc_num=mc_num, batch_size=bs, mask_id=adapter.mask_id,
    )
