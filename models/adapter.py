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

import contextlib
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


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

    # ids
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

    # the two logit surfaces
    @abstractmethod
    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Exactly what the model returns, unmodified. For NATIVE samplers."""

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw output to position-aligned. Identity for all but Dream."""
        return logits

    def logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """Position-aligned logits. For PORTABLE samplers."""
        return self._canonicalize(self.raw_logits(input_ids, **kw))

    # text
    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


"""Thin helpers over HuggingFace dynamic-module loading.

Every comparator ships its modeling code via `trust_remote_code`, so we resolve the
class through the dynamic-module machinery rather than the Auto* factories. That
gives each family a hook to install its compat shims on the CLASS before any
weights are materialised -- which is where several of them have to happen.
"""


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


"""Access to pinned upstream reference checkouts.

Upstream repos are reference material and test fixtures -- NEVER runtime
dependencies. Nothing here is on the import path during a normal eval. Two callers:

  * the parity tests        mint goldens by running their code unmodified
  * model_families/<fam>.py generate_upstream()   the upstream escape hatch

Why the escape hatch exists at all, given we re-implement everything: a touchup
creates a code path with NO upstream counterpart to compare against. SDAR greedy is
the clean example -- greedy does not exist upstream (their script divides by
temperature then multinomials), so our greedy can never be parity-tested. Being able
to run their code on demand lets us at least bracket such a path instead of flying
blind. It also costs almost nothing, since the parity fixtures need it anyway.

Populate by cloning each repo at its pin (UPSTREAM_PINS below
carries org, role, and provenance per repo).
"""


_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"


# Pinned upstream commits, with each repo's role and -- critically -- WHICH
# implementation produced WHICH published table. Unpinned upstream is worse
# than no upstream: JetEngine's dynamic_threshold default has already
# drifted 0.9 -> 0.75 under research-numbered commits.
#   LLaDA published via lm-eval (their own register_model): reproducible.
#   Dream published via a VENDORED, modified lm-eval 0.4.8
#     (eval_instruct/lm_eval) -- same version we pin, but their
#     modifications have never been diffed; treat their harness numbers
#     accordingly.
#   SDAR's published table came from NEITHER shipped sampler but from
#     LMDeploy (block=4, steps=4, low_confidence_dynamic, tau=0.9, greedy)
#     -- full story in models/sdar.py; our SDAR numbers are re-measurements.
UPSTREAM_PINS = {
    "LLaDA": "96441d4",        # ML-GSAI/LLaDA: generate.py (native sampler),
                               #   EVAL.md (published decode + task configs),
                               #   get_log_likelihood.py (MC-NELBO reference)
    "Dream": "31f94a6",        # DreamLM/Dream: eval/eval.py lm-eval wrapper
                               #   (the logit shift at :354)
    "Dream-Coder": "79d4387",  # DreamLM/Dream-Coder: variant of the above
    "SDAR": "6a12cdb",         # JetAstra/SDAR: generate.py
                               #   (block_diffusion_generate), OpenCompass configs
    "JetEngine": "bf8cb31",    # Labman42/JetEngine: SDAR inference engine,
                               #   ~15 remasking strategies, has a greedy path
}


def path_for(name: str) -> Path:
    p = _THIRD_PARTY / name
    if not p.exists():
        raise FileNotFoundError(
            f"upstream reference {name!r} not present. Clone it at its pin "
            f"(org in the UPSTREAM_PINS comments), e.g.\n"
            f"  git clone <org-url>/{name} {p} && "
            f"git -C {p} checkout {UPSTREAM_PINS[name]}"
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
    want = UPSTREAM_PINS[name]
    have = head_of(name)
    if not have.startswith(want) and not want.startswith(have):
        msg = (f"{name} is at {have}, pinned to {want} "
               "(see UPSTREAM_PINS in models/adapter.py)")
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
