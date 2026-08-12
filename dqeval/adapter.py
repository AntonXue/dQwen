"""The model contract.

A portable sampler needs exactly one thing from a model: logits for a canvas.
Everything else -- mask ids, whether the family needs an eval-side logit shift,
which transformers-5 compat shims to install, chat formatting -- is the adapter's
problem. That is what makes the model x sampler cross-product cheap.

TWO logit surfaces, and the distinction is load-bearing:

  raw_logits(ids)  exactly what the model returns.
                   NATIVE samplers use this -- they carry their own shift.
  logits(ids)      canonical: logits[:, i] is the distribution FOR position i.
                   PORTABLE samplers use this, so one engine runs on any family.

TWO families have AR-aligned raw output -- Dream and CoDA -- and both override
`_canonicalize` with `cat([l[:, :1], l[:, :-1]])`, the same expression each of
their own samplers applies once after every forward. Running either family's
native sampler on canonicalised logits would double-shift it: silent,
off-by-one, no exception. That is precisely why the two surfaces are separate
methods rather than a config flag.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GenOutput:
    """One generation. bs=1 by construction -- see `dqeval.sampler`."""

    prompt_ids: torch.Tensor
    gen_ids: torch.Tensor
    text: str
    n_forward: int = 0

    @property
    def full_ids(self) -> torch.Tensor:
        return torch.cat([self.prompt_ids, self.gen_ids], dim=-1)


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about a model repo. `mask_id` is asserted against the config
    where the family exposes one, so a silent id drift cannot go unnoticed."""

    repo: str
    family: str
    mask_id: int
    pad_id: Optional[int] = None
    auto_class: str = "AutoModel"
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    # ---- ours (private org) ---------------------------------------------
    # step checkpoints are HF revisions: load(..., revision="step30000-swa")
    "dqwen3.5-0.8b-base": ModelSpec("EER6b/dQwen3.5-0.8B-Base", "dqwen", 248061, 248044),
    "dqwen3.5-2b-base":   ModelSpec("EER6b/dQwen3.5-2B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-4b-base":   ModelSpec("EER6b/dQwen3.5-4B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-9b-base":   ModelSpec("EER6b/dQwen3.5-9B-Base",   "dqwen", 248061, 248044),
    # v3-MIXTURE production runs. Same recipe as the entries above (LR ladder, seed,
    # schedule) -- the mixture is the only deliberate change, so `-base` vs `-base-v3`
    # at a matched step count is the mixture A/B. 0.8B main == step50000-swa, which is
    # budget-matched to the released v1 (also 50k); step25000-swa is the half-budget leg.
    "dqwen3.5-0.8b-base-v3": ModelSpec("EER6b/dQwen3.5-0.8B-Base-v3", "dqwen", 248061, 248044),
    "dqwen3.5-2b-base-v3":   ModelSpec("EER6b/dQwen3.5-2B-Base-v3",   "dqwen", 248061, 248044),
    "dqwen3.5-4b-base-v3":   ModelSpec("EER6b/dQwen3.5-4B-Base-v3",   "dqwen", 248061, 248044),
    "dqwen3.5-9b-base-v3":   ModelSpec("EER6b/dQwen3.5-9B-Base-v3",   "dqwen", 248061, 248044),
    # ⚠ DEAD REPOS (404 as of 2026-08-05) -- the LR-sweep a0/a1 diagnostic pair and the
    # 4B arm-b2 test were deleted when the v2* ablation repos were consolidated. Kept so
    # the `ov_*` runs in _runs/ remain traceable; do not use without re-pointing them.
    "dqwen3.5-0.8b-base-v1": ModelSpec("EER6b/dQwen3.5-0.8B-Base-test-v1", "dqwen", 248061, 248044),
    "dqwen3.5-0.8b-base-v2": ModelSpec("EER6b/dQwen3.5-0.8B-Base-test-v2", "dqwen", 248061, 248044),
    "dqwen3.5-4b-base-v2":   ModelSpec("EER6b/dQwen3.5-4B-Base-test",       "dqwen", 248061, 248044),
    "dqwen3-0.6b-base":   ModelSpec("EER6b/dQwen3-0.6B-Base",   "dqwen", 151660, 151643),
    # v3-recipe backbone ablation: Qwen3-1.7B (FULL-attention) under the same recipe as the
    # hybrid 2B -- argparse.json differs in 2 of 22 keys. Also the same backbone as
    # Salesforce/CoDA-v0-Base, so this is the recipe-controlled CoDA comparison.
    # Qwen3 ids (151660/151643), NOT Qwen3.5's -- the publish script's own latent bug.
    "dqwen3-1.7b-base-v3": ModelSpec("EER6b/dQwen3-1.7B-Base-v3", "dqwen", 151660, 151643),
    "dqwen3-1.7b-base":   ModelSpec("EER6b/dQwen3-1.7B-Base",   "dqwen", 151660, 151643),
    # ---- comparators -----------------------------------------------------
    "llada-8b-base":     ModelSpec("GSAI-ML/LLaDA-8B-Base",     "llada", 126336),
    "llada-8b-instruct": ModelSpec("GSAI-ML/LLaDA-8B-Instruct", "llada", 126336),
    "dream-7b-instruct": ModelSpec("Dream-org/Dream-v0-Instruct-7B", "dream", 151666),
    "dream-7b-base":     ModelSpec("Dream-org/Dream-v0-Base-7B",     "dream", 151666),
    "dream-coder-7b-base": ModelSpec(
        "Dream-org/Dream-Coder-v0-Base-7B", "dream", 151666),
    "dream-coder-7b-instruct": ModelSpec(
        "Dream-org/Dream-Coder-v0-Instruct-7B", "dream", 151666),
    "sdar-1.7b-chat": ModelSpec(
        "JetLM/SDAR-1.7B-Chat", "sdar", 151669, auto_class="AutoModelForCausalLM",
        notes="repo is missing fused_linear_diffusion_cross_entropy.py; see families/sdar/compat.py"),
    # CoDA: Salesforce's masked DLM adapted from Qwen3-1.7B -- the SAME backbone as
    # our dQwen3-1.7B control, so this is the closest available recipe-isolating
    # comparison (differs in data + budget, not in starting weights).
    "coda-1.7b-base": ModelSpec(
        "Salesforce/CoDA-v0-Base", "coda", 151669, 151643,
        notes="AR-aligned raw logits (shift in _canonicalize); needs the "
              "CoDAModel._supports_sdpa shim on transformers>=4.54"),
    "coda-1.7b-instruct": ModelSpec(
        "Salesforce/CoDA-v0-Instruct", "coda", 151669, 151643,
        notes="same shims as coda-1.7b-base"),
    "sdar-4b-chat": ModelSpec(
        "JetLM/SDAR-4B-Chat", "sdar", 151669, auto_class="AutoModelForCausalLM",
        notes="same missing-module defect as SDAR-1.7B"),
}


class ModelAdapter(ABC):
    """Wraps a loaded model behind the two logit surfaces."""

    family: str = "?"

    def __init__(self, model, tokenizer, spec: ModelSpec,
                 revision: Optional[str] = None, shims: Optional[list[str]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.revision = revision or "main"
        self.shims = shims or []

    # -- ids ---------------------------------------------------------------
    @property
    def mask_id(self) -> int:
        return self.spec.mask_id

    @property
    def pad_id(self) -> int:
        if self.spec.pad_id is not None:
            return self.spec.pad_id
        tok = self.tokenizer
        for cand in (tok.pad_token_id, tok.eos_token_id):
            if cand is not None:
                return cand
        raise ValueError(f"{self.spec.repo}: no pad/eos token id available")

    @property
    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # -- the two logit surfaces -------------------------------------------
    @abstractmethod
    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Exactly what the model returns, unmodified. For NATIVE samplers."""

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw output to position-aligned. Identity for all but Dream."""
        return logits

    def logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Position-aligned logits. For PORTABLE samplers."""
        return self._canonicalize(self.raw_logits(input_ids, **kw))

    # -- text --------------------------------------------------------------
    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def chat(self, message: str) -> str:
        """Instruct-format a single user turn. Base models return it unchanged."""
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if tmpl is None:
            return message
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False, add_generation_prompt=True,
        )

    def describe(self) -> dict:
        """Provenance for the result ledger."""
        import transformers
        return {
            "model": self.spec.repo,
            "revision": self.revision,
            "family": self.family,
            "mask_id": self.mask_id,
            "shims": list(self.shims),
            "env": {"transformers": transformers.__version__, "torch": torch.__version__},
        }


_FAMILY_MODULES = {
    "llada": "dqeval.families.llada",
    "dream": "dqeval.families.dream",
    "sdar": "dqeval.families.sdar",
    "coda": "dqeval.families.coda",
    "dqwen": "dqeval.families.dqwen",
}


def load(name: str, revision: Optional[str] = None, dtype=torch.bfloat16,
         device: str = "cuda") -> ModelAdapter:
    """Load a registered model, applying that family's compat shims."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    spec = MODELS[name]
    mod = importlib.import_module(_FAMILY_MODULES[spec.family])
    return mod.build(spec, revision=revision, dtype=dtype, device=device)


# ==========================================================================
# (merged from dqeval/hf.py)
# ==========================================================================

"""Thin helpers over HuggingFace dynamic-module loading.

Every comparator ships its modeling code via `trust_remote_code`, so we resolve the
class through the dynamic-module machinery rather than the Auto* factories. That
gives each family a hook to install its compat shims on the CLASS before any
weights are materialised -- which is where several of them have to happen.
"""


from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def resolve(repo: str, revision: Optional[str] = None, auto_class: str = "AutoModel"):
    """Return (config, model_class) for a remote-code repo, without loading weights."""
    cfg = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=True)
    amap = cfg.auto_map or {}
    ref = amap.get(auto_class)
    if ref is None:                      # fall back to whatever the repo does expose
        ref = amap.get("AutoModel") or amap.get("AutoModelForCausalLM")
    if ref is None:
        raise RuntimeError(f"{repo}: auto_map exposes no usable model class: {amap}")
    klass = get_class_from_dynamic_module(ref, repo, revision=revision)
    return cfg, klass


def tokenizer(repo: str, revision: Optional[str] = None):
    return AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)


def materialize(klass, repo: str, cfg, revision: Optional[str] = None,
                dtype=torch.bfloat16, device: str = "cuda"):
    model = klass.from_pretrained(repo, config=cfg, revision=revision, dtype=dtype)
    return model.to(device).eval()


def assert_finite_rope(model) -> None:
    """Guard against the tf5 meta-device buffer trap.

    transformers>=5 builds models under a bare `torch.device("meta")`. Rotary
    `inv_freq` is a non-persistent buffer, so it is absent from the checkpoint and
    materialises as uninitialised memory. tf5 repairs this inside the BASE
    `PreTrainedModel._init_weights`, but any remote-code model that OVERRIDES
    `_init_weights` (Dream and SDAR both do) never reaches that branch.

    The failure is silent: the model loads, forwards, and returns correctly-shaped
    logits that are noise. Measured cost when unrepaired: argmax agreement 20-40%
    against the native environment. Cheap to assert, so we always assert.
    """
    bad = [n for n, b in model.named_buffers()
           if n.endswith("inv_freq") and not torch.isfinite(b).all()]
    if bad:
        raise RuntimeError(
            f"non-finite rotary inv_freq buffers: {bad[:3]}{'...' if len(bad) > 3 else ''}. "
            "This is the transformers>=5 meta-device trap -- the family's compat shim "
            "must recompute inv_freq AFTER from_pretrained materialises the model."
        )


# ==========================================================================
# (merged from dqeval/upstream.py)
# ==========================================================================

"""Access to pinned upstream reference checkouts.

Upstream repos are reference material and test fixtures -- NEVER runtime
dependencies. Nothing here is on the import path during a normal eval. Two callers:

  * tests/parity/           mints goldens by running their code unmodified
  * families/<fam>.py generate_upstream()   the upstream escape hatch

Why the escape hatch exists at all, given we re-implement everything: a touchup
creates a code path with NO upstream counterpart to compare against. SDAR greedy is
the clean example -- greedy does not exist upstream (their script divides by
temperature then multinomials), so our greedy can never be parity-tested. Being able
to run their code on demand lets us at least bracket such a path instead of flying
blind. It also costs almost nothing, since the parity fixtures need it anyway.

Populate with `third_party/fetch.sh`.
"""


import contextlib
import json
import subprocess
import sys
from pathlib import Path

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"


def pins() -> dict:
    with open(_THIRD_PARTY / "pins.json") as f:
        return json.load(f)["repos"]


def path_for(name: str) -> Path:
    p = _THIRD_PARTY / name
    if not p.exists():
        raise FileNotFoundError(
            f"upstream reference {name!r} not fetched. Run third_party/fetch.sh"
        )
    return p


def head_of(name: str) -> str:
    out = subprocess.run(["git", "-C", str(path_for(name)), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def verify_pin(name: str, strict: bool = True) -> str:
    """Check the checkout sits at its pinned commit.

    Unpinned upstream is worse than no upstream: JetEngine's dynamic_threshold
    default has already drifted 0.9 -> 0.75 under opt-numbered research commits, so
    a number attributed to "their sampler" without a commit is unattributable.
    """
    want = pins()[name]["commit"]
    have = head_of(name)
    if not have.startswith(want) and not want.startswith(have):
        msg = f"{name} is at {have}, pinned to {want}. Run third_party/fetch.sh"
        if strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
    return have


@contextlib.contextmanager
def on_path(name: str, strict: bool = True):
    """Temporarily put a pinned upstream checkout on sys.path."""
    p = str(path_for(name))
    verify_pin(name, strict=strict)
    sys.path.insert(0, p)
    try:
        yield p
    finally:
        if p in sys.path:
            sys.path.remove(p)
