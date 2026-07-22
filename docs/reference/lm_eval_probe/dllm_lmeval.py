#!/usr/bin/env python
"""PoC: wrap the dQwen3.5 masked-diffusion model as an lm-evaluation-harness
custom model. Two methods:
  generate_until  -> champion decode (block_append_generate)
  loglikelihood   -> MASK-slot scoring (one-shot DLM, exact for 1-token conts)

Run via:  python run_probe.py  (which registers this + calls simple_evaluate)
"""
from __future__ import annotations
import sys, torch
from pathlib import Path

# --- transformers 5.x compat: lm_eval hf_vlms.py already patched in this venv ---
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ablations.paper_evals.common import load_model_tokenizer            # noqa: E402
from ablations.paper_evals.dlm_decode import block_append_generate       # noqa: E402
from ablations.paper_evals.mc_nelbo import mc_nelbo_loglikelihood         # noqa: E402
from lm_eval.api.model import LM                                          # noqa: E402
from lm_eval.api.registry import register_model                          # noqa: E402


@register_model("dllm")
class DLLM(LM):
    """Masked-diffusion LM for lm-eval-harness."""

    def __init__(self, pretrained, batch_size=8, max_length=1024,
                 block=32, steps_per_block=32, mc_num=128, mc_bs=16, **kw):
        super().__init__()
        self.model, self.tok = load_model_tokenizer(pretrained)
        self.model.eval()
        self.bs = int(batch_size)
        self.max_length = int(max_length)
        self.block = int(block)
        self.spb = int(steps_per_block)
        self.mc_num = int(mc_num)   # MC-NELBO samples for multi-token continuations
        self.mc_bs = int(mc_bs)     # MC replicas per forward (mc_num % mc_bs == 0)
        self.MASK = self.tok.mask_token_id
        self.PAD = self.tok.pad_token_id
        self.device = next(self.model.parameters()).device

    # ---- required by lm-eval; simple_evaluate reads these for chunking ----
    @property
    def eot_token_id(self):
        return self.PAD

    @property
    def max_gen_toks(self):
        return 512

    def tok_encode(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    # ---------------- generative tasks (gsm8k, humaneval) ----------------
    @torch.inference_mode()
    def generate_until(self, requests):
        # requests: list of Instance; .args = (context_str, gen_kwargs)
        ctxs = [r.args[0] for r in requests]
        stops = [r.args[1].get("until", []) if len(r.args) > 1 and isinstance(r.args[1], dict) else []
                 for r in requests]
        out = [None] * len(ctxs)
        for lo in range(0, len(ctxs), self.bs):
            chunk = ctxs[lo:lo + self.bs]
            gens, _ = block_append_generate(
                self.model, self.tok, chunk, block=self.block,
                steps_per_block=self.spb, temp=0.0, sigma_scale=0.0,
                max_length=self.max_length)
            for j, g in enumerate(gens):
                # strip pad + cut at the first requested stop string
                g = g.replace(self.tok.pad_token, "")
                for st in ("<|im_end|>", "<|endoftext|>"):
                    if st in g:
                        g = g.split(st)[0]
                for u in stops[lo + j]:
                    if u and u in g:
                        g = g.split(u)[0]
                out[lo + j] = g
        return out

    # ------- multiple-choice (mmlu/arc) -> MC-NELBO loglikelihood -------
    @torch.inference_mode()
    def loglikelihood(self, requests):
        # LLaDA/Dream-faithful MC-NELBO (ablations/paper_evals/mc_nelbo.py, validated
        # bit-exact vs LLaDA on LLaDA-8B-Base). requests: .args=(context, continuation).
        # Single-token targets (MMLU/ARC letters) use mc_num=1 (exact; == MASK-slot).
        # is_greedy=False (LLaDA convention; is_check_greedy defaults off).
        logits_fn = lambda x: self.model(input_ids=x).logits
        res = []
        for r in requests:
            ctx = self.tok_encode(r.args[0])
            cont = self.tok_encode(r.args[1])
            budget = self.max_length - len(cont)
            if budget < len(ctx):
                ctx = ctx[-budget:] if budget > 0 else ctx[:1]
            p = torch.tensor(ctx, device=self.device)
            a = torch.tensor(cont, device=self.device)
            mc = 1 if len(cont) == 1 else self.mc_num
            bs = 1 if len(cont) == 1 else self.mc_bs
            ll = mc_nelbo_loglikelihood(logits_fn, p, a, mc_num=mc,
                                        batch_size=bs, mask_id=self.MASK)
            res.append((ll, False))
        return res

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("rolling LL not needed for gsm8k/mmlu/he")
