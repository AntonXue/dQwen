"""Diffusion LM from Qwen3.5; the BOS/logit shift preserves the AR next-token bias."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.modeling_outputs import ModelOutput


# Qwen3.5 tokenizer ids
MASK_TOKEN_ID = 248061  # <|fim_middle|>
PAD_TOKEN_ID = 248044   # <|endoftext|>
BOS_TOKEN_ID = 248073   # <tts_text_bos>



@dataclass
class DiffusionGenerateOutput:
    """One generation (batch size one). `gen_ids` is the full answer canvas,
    `text` its decode (None without a tokenizer), `n_forward` the number of
    model forwards spent, and `commit_step[i]` the forward on which answer
    position i was committed (-1: never, canvas cut by an early stop)."""
    gen_ids: torch.Tensor
    text: Optional[str]
    n_forward: int
    commit_step: torch.Tensor


class DiffuQwen35Config(Qwen3_5TextConfig):
    model_type = "diffuqwen35"


class DiffuQwen35(Qwen3_5ForCausalLM):
    config_class = DiffuQwen35Config

    def __init__(self, config: DiffuQwen35Config):
        super().__init__(config)
        # Flip only the full_attention layers; GatedDeltaNet is causal by construction.
        for layer in self.model.layers:
            if layer.block_type == "full_attention":
                layer.self_attn.is_causal = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        compute_logits: bool = True,
        **kwargs,
    ) -> ModelOutput:
        """Prepend BOS, hidden is (B, L+1, d), logits are (B, L, V)."""
        B, _ = input_ids.shape
        bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([bos, input_ids], dim=1)
        if attention_mask is not None:
            ones = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([ones, attention_mask], dim=1)

        inputs_embeds = self.model.embed_tokens(input_ids)

        # Avoids HF rebuilding a causal mask on the full_attention layers.
        # full_attention: 4D additive; linear_attention: 2D padding.
        if attention_mask is None or attention_mask.all():
            attn_mask = {"full_attention": None, "linear_attention": None}
        else:
            _, L1, _ = inputs_embeds.shape
            mask_4d = attention_mask.reshape(B, 1, 1, L1).expand(B, 1, L1, L1)
            mask_4d = (1.0 - mask_4d.to(inputs_embeds.dtype)) * -10000
            attn_mask = {"full_attention": mask_4d, "linear_attention": attention_mask}

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
        hidden = out.last_hidden_state
        logits = self.lm_head(hidden[:, :-1, :]) if compute_logits else None
        return ModelOutput(hidden=hidden, logits=logits)

    # ------------------------------------------------------------------ generation
    # Ported verbatim from the dQwen evaluation harness (models/samplers.py) --
    # the engine every number in the technical report was produced with -- and
    # held token-identical to it by tests/test_dqwen_release_sampler.py.
    _NEG_INF = float("-inf")

    def _dg_top_k(self, logits, k):
        if k <= 0:
            return logits
        k = min(k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < kth, float("-inf"))

    def _dg_top_p(self, logits, p):
        if p >= 1.0:
            return logits
        ordered, idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
        drop = cum > p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        return logits.masked_fill(drop.scatter(-1, idx, drop), float("-inf"))

    def _dg_assert_finite_rope(self):
        # transformers>=5 builds on device("meta"); a non-persistent rotary
        # buffer can materialise as garbage or zeros and the failure is SILENT
        # (right shapes, noise logits). inv_freq[0] == theta**0 == 1.0 exactly.
        if getattr(self, "_dg_rope_checked", False):
            return
        bad = [n for n, b in self.named_buffers()
               if n.endswith("inv_freq")
               and (not torch.isfinite(b).all() or abs(float(b.reshape(-1)[0]) - 1.0) > 1e-3)]
        if bad:
            raise RuntimeError(f"broken rotary inv_freq buffers {bad[:3]}: the model was "
                               "materialised without rotary init (transformers>=5 meta-device trap)")
        self._dg_rope_checked = True

    def _dg_tokenizer(self):
        # Loaded once from the repo the weights came from; None if this model was
        # built from a bare config and no repo name is known.
        if not hasattr(self, "_dg_tok"):
            name = getattr(self, "name_or_path", "") or getattr(self.config, "_name_or_path", "")
            if name:
                from transformers import AutoTokenizer
                self._dg_tok = AutoTokenizer.from_pretrained(name)
            else:
                self._dg_tok = None
        return self._dg_tok

    @torch.no_grad()
    def generate(
        self,
        prompt,
        tokenizer=None,
        *,
        gen_length: int = 512,
        block_length: Optional[int] = None,
        tau: Optional[float] = 0.9,
        steps_per_block: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        order: str = "low_confidence",
        stop_strings=None,
        seed: Optional[int] = None,
    ) -> DiffusionGenerateOutput:
        """Masked-diffusion decoding of ONE prompt: a string, or its ids as [1, L].

        `tokenizer` may be omitted: it is loaded once from the repo this model came
        from (`name_or_path`) and cached, so `model.generate("def f(x):").text` is
        the whole call. Pass one explicitly to control it.

        The answer canvas of `gen_length` MASK tokens is appended to the prompt
        and revealed left to right in blocks of `block_length` (default None:
        one block, the whole canvas -- the report's protocol, evaluated there at
        gen_length=1024). Within a block, each forward proposes every masked
        position and commits some of them:

          tau=<float>   commit every position whose confidence exceeds tau, and
                        at least one per forward, so it is never slower than one
                        token per step (default 0.9: accuracy at or above one
                        token per step at roughly 5x fewer forwards on the whole
                        canvas, 10-30x with block_length=32, whose per-block
                        early stop also ends short completions sooner).
          tau=None      commit a fixed number per forward so the block finishes
                        in exactly `steps_per_block` forwards (default: one
                        token per step). Aggressive budgets collapse where the
                        threshold does not -- e.g. 4 tokens per step roughly
                        halves HumanEval -- so lower `steps_per_block` with care.

        `temperature=0.0` is argmax; otherwise sample with top-k/top-p, with
        confidence always read from the unfiltered distribution. `stop_strings`
        (needs `tokenizer`) end decoding once a completed block's text contains
        one; a masked DLM does not emit EOS, so this is what keeps a completion
        from running to the full canvas. `order` ranks candidates:
        "low_confidence" (default), "entropy", "topk_margin", "random",
        "sequential".
        """
        if tokenizer is None:
            tokenizer = self._dg_tokenizer()
        if isinstance(prompt, str):
            if tokenizer is None:
                raise ValueError("a string prompt needs a tokenizer; pass one, or load the model from its repo")
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = prompt
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(f"batch size one only; got {tuple(input_ids.shape)}")
        if block_length is None:
            block_length = gen_length
        if steps_per_block is None:
            steps_per_block = block_length
        if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
            raise ValueError("gen_length must be a positive multiple of block_length")
        if steps_per_block <= 0:
            raise ValueError("steps_per_block must be positive")
        if temperature < 0.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("temperature must be >= 0 and top_p in (0, 1]")
        if stop_strings and tokenizer is None:
            raise ValueError("stop_strings needs a tokenizer")
        self._dg_assert_finite_rope()
        if seed is not None:
            torch.manual_seed(seed)

        dev = input_ids.device
        mask_id = MASK_TOKEN_ID
        p_len = input_ids.size(1)
        greedy = temperature == 0.0
        NEG_INF = self._NEG_INF

        canvas = torch.full((1, p_len + gen_length), mask_id, dtype=torch.long, device=dev)
        canvas[:, :p_len] = input_ids
        n_forward = 0
        stop_text = None
        commit_step = torch.full((gen_length,), -1, dtype=torch.long)
        decode = (lambda ids: tokenizer.decode(ids, skip_special_tokens=False)) if tokenizer else None

        for b in range(gen_length // block_length):
            lo = p_len + b * block_length
            hi = lo + block_length

            n_masked = int((canvas[0, lo:hi] == mask_id).sum())
            base, rem = divmod(n_masked, steps_per_block)
            schedule = torch.full((steps_per_block,), base, dtype=torch.long, device=dev)
            schedule[:rem] += 1

            for step in range(steps_per_block):
                masked = canvas[0, lo:hi] == mask_id
                if not masked.any():
                    break

                logits = self(input_ids=canvas).logits[0, lo:hi].clone()
                n_forward += 1
                logits[:, mask_id] = NEG_INF

                conf_all = F.softmax(logits.float(), dim=-1)
                if greedy:
                    x0 = logits.argmax(dim=-1)
                else:
                    filtered = self._dg_top_p(self._dg_top_k(logits.float() / temperature, top_k), top_p)
                    x0 = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1).squeeze(-1)
                conf = conf_all.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

                if order == "low_confidence":
                    score = conf
                elif order == "random":
                    score = torch.rand_like(conf)
                elif order == "sequential":
                    score = torch.arange(conf.numel(), 0, -1, device=dev, dtype=conf.dtype)
                elif order in ("entropy", "topk_margin"):
                    p = F.softmax(logits.float(), dim=-1)
                    if order == "entropy":
                        score = (p * p.clamp_min(1e-12).log()).sum(-1)
                    else:
                        top2 = p.topk(2, dim=-1).values
                        score = top2[:, 0] - top2[:, 1]
                else:
                    raise ValueError(f"unknown order {order!r}")
                score = score.masked_fill(~masked, NEG_INF)

                if tau is not None:
                    take = (conf > tau) & masked
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
                commit_step[b * block_length + take.nonzero(as_tuple=True)[0].cpu()] = n_forward - 1

            if stop_strings:
                gen_text = decode(canvas[0, p_len:hi])
                hits = [h for h in (gen_text.find(s) for s in stop_strings if s) if h >= 0]
                if hits:
                    stop_text = gen_text[:min(hits)]
                    break

        gen_ids = canvas[0, p_len:]
        text = None if decode is None else (decode(gen_ids) if stop_text is None else stop_text)
        return DiffusionGenerateOutput(gen_ids=gen_ids, text=text, n_forward=n_forward,
                                       commit_step=commit_step)

    @classmethod
    def from_qwen3_5(
        cls,
        base_model_name: str = "Qwen/Qwen3.5-0.8B",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "DiffuQwen35":
        """Seed from a pretrained Qwen3.5 VLM checkpoint (text weights)."""
        # sdpa: honors our is_causal=False flip, and avoids a flash-attn dependency.
        vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_name, dtype=dtype, attn_implementation="sdpa"
        )
        cfg = DiffuQwen35Config(**vlm.config.text_config.to_dict())
        cfg._attn_implementation = "sdpa"
        model = cls(cfg)
        model.model.load_state_dict(vlm.model.language_model.state_dict(), strict=False)

        # 0.8B/2B/4B tie lm_head to embed_tokens; 9B/27B don't tie.
        if cfg.tie_word_embeddings:
            model.tie_weights()
        else:
            model.lm_head.load_state_dict(vlm.lm_head.state_dict())

        del vlm
        return model.to(dtype)


# Ensures that save_pretrained emits auto_map and copies this file.
DiffuQwen35Config.register_for_auto_class()
DiffuQwen35.register_for_auto_class("AutoModel")
