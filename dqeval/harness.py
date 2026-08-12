"""lm-evaluation-harness wrapper: any dqeval family as an lm-eval `LM`.

Registered as model type "dqeval". Drive it through lm-eval's `simple_evaluate`:

    simple_evaluate(model="dqeval",
                    model_args="pretrained=llada-8b-base",
                    tasks=["mmlu"], ...)

`pretrained` is a dqeval REGISTRY NAME (see dqeval.adapter.MODELS), not a raw HF
repo -- so the family's compat shims and the correct logit surface are applied
automatically. That is the point: the same wrapper serves LLaDA, Dream and
dQwen with no per-family branching here.

TWO METHODS the harness needs:

  loglikelihood(context, continuation)  -> multiple choice (mmlu, arc, ...)
      MC-NELBO via `adapter.logits` (CANONICAL, not raw). Routing through the
      canonical surface is what makes Dream correct -- its raw logits are
      AR-aligned, and scoring them directly would be an off-by-one with no error.
      Single-token continuations (mmlu/arc letters) take the exact mc_num=1 path.

  generate_until(context, until)        -> generative (gsm8k, humaneval, ...)
      Our bs=1 block-diffusion sampler, stop-string aware. Not exercised by the
      loglikelihood tasks; wired here so the same wrapper covers both.

NOTE we do NOT suppress the MASK id in loglikelihood. LLaDA's reference does not
either, and `dqeval.nelbo` is gated bit-exact against it -- adding suppression here
would silently break that parity. MASK suppression belongs to generation only.

CFG is not applied (cfg_scale defaults to 0): dqeval reports a common disclosed
decode protocol, not LLaDA's per-task classifier-free-guidance tuning. cfg_scale is
a documented variant, not a default.
"""

from __future__ import annotations

import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from dqeval.adapter import load
from dqeval import sampler
from dqeval.sampler import DecodeConfig

# Model-end stop strings, shared by every generation driver — truncation
# rules move scores, so there is exactly one copy. The ``` fence catches
# models (Dream) that wrap completions in markdown, which task-level `until`
# strings miss; python source never contains ``` so it is safe to stop on.
EOT_BASE = ("```", "<|im_end|>", "<|endoftext|>")


def eot_stops(adapter):
    """EOT_BASE plus the model's own pad token, if it has one."""
    pad = adapter.tokenizer.pad_token
    return list(EOT_BASE) + ([pad] if pad else [])


def truncate_at(text, stops):
    """Cut `text` at the first occurrence of any stop string."""
    for st in stops:
        if st and st in text:
            text = text.split(st)[0]
    return text


@register_model("dqeval")
class DQEvalLM(LM):
    def __init__(self, pretrained, revision=None, *, batch_size=1,
                 max_length=2048, gen_length=1024, block_length=32,
                 steps_per_block=32, order="low_confidence",
                 top_p=1.0, top_k=0, mc_num=128, mc_bs=16,
                 temperature=0.0, cfg_scale=0.0,
                 commit="static", confidence_threshold=0.9, **kw):
        super().__init__()
        self.adapter = load(pretrained, revision=revision)
        self.max_length = int(max_length)
        self.mc_num = int(mc_num)
        self.mc_bs = int(mc_bs)
        # decode config for generate_until; loglikelihood tasks ignore it
        self.decode = DecodeConfig(
            gen_length=int(gen_length), block_length=int(block_length),
            steps_per_block=int(steps_per_block), order=order,
            top_p=float(top_p), top_k=int(top_k),
            temperature=float(temperature), cfg_scale=float(cfg_scale),
            commit=commit, confidence_threshold=float(confidence_threshold),
        )

    # -- required by lm-eval for request chunking --------------------------
    @property
    def eot_token_id(self):
        return self.adapter.pad_id

    @property
    def max_gen_toks(self):
        return self.decode.gen_length

    def tok_encode(self, s):
        return self.adapter.tokenizer(s, add_special_tokens=False)["input_ids"]

    def tok_decode(self, ids):
        return self.adapter.decode(torch.as_tensor(ids))

    def _prep(self, r):
        """(context_ids, continuation_ids) with the context length-budgeted."""
        ctx = self.tok_encode(r.args[0])
        cont = self.tok_encode(r.args[1])
        budget = self.max_length - len(cont)
        if budget < len(ctx):
            ctx = ctx[-budget:] if budget > 0 else ctx[:1]
        return ctx, cont

    # -- multiple choice: mmlu, arc, hellaswag, piqa, winogrande, ... ------
    @torch.inference_mode()
    def loglikelihood(self, requests):
        dev = self.adapter.device
        mask_id = self.adapter.mask_id
        out = [None] * len(requests)

        # FAST PATH. Multiple-choice tasks issue one request per (question, option)
        # with the SAME context and single-token continuations (mmlu/arc letters).
        # The masked canvas [ctx, MASK] is then identical across a question's options,
        # so one forward serves them all -- we just read a different vocab index per
        # option. This is the exact mc_num=1 MASK-slot marginal, provably identical to
        # scoring each option with its own forward, and still bs=1 (no batch-invariance
        # exposure). It removes the ~4x redundant forwards multiple-choice would spend.
        groups: dict[tuple, list[tuple[int, int]]] = {}
        for i, r in enumerate(requests):
            ctx, cont = self._prep(r)
            if len(cont) == 1:
                groups.setdefault(tuple(ctx), []).append((i, cont[0]))
            else:
                # multi-token continuation -> full MC-NELBO (no shared-forward reuse)
                p = torch.tensor(ctx, device=dev)
                a = torch.tensor(cont, device=dev)
                ll = sampler.mc_nelbo_loglikelihood(
                    self.adapter.logits, p, a,
                    mc_num=self.mc_num, batch_size=self.mc_bs, mask_id=mask_id,
                )
                out[i] = (ll, False)

        for ctx, items in groups.items():
            canvas = torch.tensor([*ctx, mask_id], device=dev)[None, :]
            # position-aligned logit at the single masked (answer) slot
            logits = self.adapter.logits(canvas)[0, len(ctx)]
            # Use the SAME op nelbo uses (F.cross_entropy on the native-dtype logits),
            # not log_softmax(logits.float()), so the shared-forward result is BITWISE
            # identical to scoring each option through mc_nelbo. ll = -CE.
            toks = torch.tensor([t for _, t in items], device=dev)
            ce = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0).expand(len(items), -1), toks, reduction="none")
            for (i, _), c in zip(items, ce):
                out[i] = (-c.item(), False)   # is_greedy=False, LLaDA convention

        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "rolling loglikelihood (perplexity tasks) is not implemented; "
            "not needed for mmlu/arc/gsm8k/humaneval"
        )

    # -- generative: gsm8k, minerva_math, humaneval, mbpp, bbh -------------
    @torch.inference_mode()
    def generate_until(self, requests):
        return [self._generate_one(r)[0] for r in requests]

    def _generate_one(self, r):
        """(final_text, GenOutput) for one request — factored out so
        RecordingLM below can record raw text + n_forward per request."""
        ctx = r.args[0]
        kw = r.args[1] if len(r.args) > 1 and isinstance(r.args[1], dict) else {}
        # stops = the task's own `until` strings + the model-end set (masked
        # DLMs don't self-terminate the way an AR model emits EOS).
        stops = list(kw.get("until", []) or []) + eot_stops(self.adapter)
        ids = self.adapter.encode(ctx)
        # stop_strings early-stops the decode itself (saves the forwards that
        # would fill the rest of the canvas); truncate_at is the post-hoc
        # backstop covering full/window mode, which decodes the whole canvas.
        gen = sampler.generate(self.adapter, ids, self.decode,
                               stop_strings=stops)
        return truncate_at(gen.text, stops), gen


class RecordingLM(DQEvalLM):
    """DQEvalLM that also keeps one record per generated request:
    (task, doc_id, raw text, truncated text, n_forward). dqeval.grid dumps
    these as the per-sample sidecar, which is what makes regrading
    (HumanEval+/MBPP+, format studies) a CPU re-pass instead of a GPU rerun.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def generate_until(self, requests):
        out = []
        for r in requests:
            text, gen = self._generate_one(r)
            self.records.append(dict(task=r.task_name, doc_id=r.doc_id,
                                     raw=gen.text, final=text,
                                     n_forward=gen.n_forward))
            out.append(text)
        return out
