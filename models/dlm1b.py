"""dlm1b adapter -- AR->DLM adaptation checkpoints from the DLM1B project
(sibling repo `~/foo/DLM1B`; cross-repo dependency, not published to the hub).

Local checkpoint directories, not an HF hub repo: `spec.repo` is a training
run's output dir (e.g. `_runs/20260814-044535-owt-baseline-1b`) and
`revision` selects the milestone subdirectory (`checkpoint-380`, ...,
`checkpoint-7600`) -- the same "revision is a first-class argument" idiom
dqwen.py uses for HF-hub step checkpoints, just resolved against a local
path instead of a hub ref. `revision=None`/"main" resolves to the highest
milestone found on disk (mirrors hub "main" == latest).

Architecture (DiffuQwen3, `~/foo/DLM1B/dlm1b/modeling_dqwen3.py`) is
Qwen3ForCausalLM with attention forced bidirectional (`is_causal = False`
on every layer) and a BOS-shift INTERNALISED in forward() (prepends BOS,
returns `lm_head(hidden[:, :-1, :])`) -- structurally identical to the
EER6b/dQwen3-0.6B-Base row already in dqwen.py's family (same ids: mask=
151660, pad=151643), so this reuses that family's adapter class verbatim
and supplies only its own build(). One difference from every hub-based
family: the checkpoint directory carries model weights + modeling code
(`auto_map`, verified present in checkpoint-380) but NOT a tokenizer --
DLM1B's own training script (train_adapt.py) sources the tokenizer from
the base model (Qwen/Qwen3-0.6B) for the same reason, so we do too.
"""

from pathlib import Path
from typing import Optional

import torch

from .adapter import ModelSpec, assert_finite_rope, materialize, resolve, tokenizer
from .dqwen import DQwenAdapter

TOKENIZER_REPO = "Qwen/Qwen3-0.6B"  # checkpoints ship no tokenizer files


class DLM1BAdapter(DQwenAdapter):
    family = "dlm1b"

    # raw_logits / _canonicalize (identity): inherited unchanged from DQwenAdapter --
    # same BOS-shift-internalised contract, verified against modeling_dqwen3.py.


def _checkpoint_dir(base: str, revision: Optional[str]) -> str:
    root = Path(base)
    if revision in (None, "main", "-"):
        cands = sorted(root.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-", 1)[1]))
        if not cands:
            raise FileNotFoundError(f"{base}: no checkpoint-* subdirectories found")
        return str(cands[-1])   # highest step == "main"
    tag = revision if revision.startswith("checkpoint-") else f"checkpoint-{revision}"
    path = root / tag
    if not path.is_dir():
        raise FileNotFoundError(f"{base}: no such checkpoint dir {path}")
    return str(path)


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> DLM1BAdapter:
    ckpt_dir = _checkpoint_dir(spec.repo, revision)
    cfg, klass = resolve(ckpt_dir)            # local dir; no HF revision to resolve
    tok = tokenizer(TOKENIZER_REPO)           # checkpoint ships no tokenizer
    model = materialize(klass, ckpt_dir, cfg, dtype=dtype, device=device)
    assert_finite_rope(model)

    vocab = model.config.vocab_size
    if not spec.mask_id < vocab:
        raise ValueError(f"{ckpt_dir}: mask_id {spec.mask_id} outside vocab {vocab}")
    return DLM1BAdapter(model, tok, spec, revision=revision, shims=[])
