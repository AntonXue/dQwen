"""qwen3sym adapter -- DLM1B's Qwen3-1.7B conversion arms (CDLM/BDLM/SDLM).

Local training-run dirs on Vista scratch (cross-repo: modeling code lives in
`~/foo/DLM1B/qwen3_sym/modeling_symqwen3.py`, imported directly -- NOT via the
checkpoint's auto_map copy, so the sym_mirror attention registration is
guaranteed present). `revision` selects a checkpoint subdirectory
("checkpoint-5000"); `main`/None prefers the run-dir root (the trainer's final
save_model output) and falls back to the highest checkpoint-*.

THE ONE RULE (toy Exp 4: releasing the constraint collapses instantly): a
checkpoint is ALWAYS evaluated through the attn_mode its config carries --
sym checkpoints through the sym forward, causal through causal, bidir through
bidir. This adapter reads config.attn_mode and refuses to guess: a checkpoint
without the field is an error, never a silent bidir default.

Forward contract: SymQwen3.forward == DiffuQwen3's (BOS prepended internally,
logits = lm_head(hidden[:, :-1])), so DQwenAdapter's logit surfaces apply
verbatim. Same ids as the dqwen3 family (mask=151660, pad=151643).
Checkpoints ship their tokenizer (DLM1B port fix); Qwen/Qwen3-1.7B is the
fallback for pre-fix dirs.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch

from .adapter import ModelSpec, assert_finite_rope, tokenizer
from .dqwen import DQwenAdapter

sys.path.insert(0, str(Path(os.environ.get(
    "DLM1B", Path.home() / "foo" / "DLM1B")) / "qwen3_sym"))

TOKENIZER_FALLBACK = "Qwen/Qwen3-1.7B"


class Qwen3SymAdapter(DQwenAdapter):
    family = "qwen3sym"


def _checkpoint_dir(base: str, revision: Optional[str]) -> str:
    root = Path(base)
    if revision in (None, "main", "-"):
        if (root / "model.safetensors").exists() or (root / "model.safetensors.index.json").exists():
            return str(root)                  # trainer's final save_model output
        cands = sorted(root.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-", 1)[1]))
        if not cands:
            raise FileNotFoundError(f"{base}: no final model and no checkpoint-* dirs")
        return str(cands[-1])
    tag = revision if revision.startswith("checkpoint-") else f"checkpoint-{revision}"
    path = root / tag
    if not path.is_dir():
        raise FileNotFoundError(f"{base}: no such checkpoint dir {path}")
    return str(path)


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> Qwen3SymAdapter:
    from modeling_symqwen3 import SymQwen3  # DLM1B sibling repo (path above)

    ckpt_dir = _checkpoint_dir(spec.repo, revision)
    model = SymQwen3.from_pretrained(ckpt_dir, attn_implementation="sdpa",
                                     dtype=dtype)
    mode = getattr(model.config, "attn_mode", None)
    if mode not in ("bidir", "causal", "sym", "mix"):
        raise ValueError(
            f"{ckpt_dir}: config carries no valid attn_mode ({mode!r}) -- "
            "refusing to guess the forward; eval must run the training mode")
    model.set_attn_mode(mode)                 # sym/mix ckpts get their own forward (mix lambda from config)
    model = model.to(device).eval()
    assert_finite_rope(model)

    try:
        tok = tokenizer(ckpt_dir)             # ships with post-fix checkpoints
    except (OSError, ValueError):
        tok = tokenizer(TOKENIZER_FALLBACK)

    vocab = model.config.vocab_size
    if not spec.mask_id < vocab:
        raise ValueError(f"{ckpt_dir}: mask_id {spec.mask_id} outside vocab {vocab}")
    return Qwen3SymAdapter(model, tok, spec,
                           revision=revision, shims=[f"attn_mode={mode}"])
