"""Decode configuration: the one knob space every sampler is parameterised by.

The claim this file encodes is that LLaDA, Dream, SDAR and dQwen do not need four
different decode APIs -- their published samplers are four points in the space
below. Each family's official recipe therefore appears as a PRESET, not as a
separate code path. `tests/parity/` is what keeps that claim honest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

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

    # --- canvas -----------------------------------------------------------
    # Decoding is ALWAYS full-canvas: the whole answer canvas is visible from
    # the start and blocks are revealed left-to-right (LLaDA/Dream native
    # style). There used to be a `mode` knob here ("append" grew the canvas
    # block-by-block with no trailing masks, biasing the model toward short
    # stub answers; "window" was never implemented beyond the type). append
    # lost the head-to-head (dQwen3.5-9B HumanEval 62.20 -> 64.63 full;
    # LLaDA 32.32 -> 32.93 = its published number) and its one genuine user,
    # SDAR, is out of the comparator set -- both modes REMOVED 2026-08-12.
    gen_length: int = 1024   # LLaDA-style default answer canvas
    block_length: int = 32          # semi-AR granularity within the canvas
    # --- schedule ---------------------------------------------------------
    steps_per_block: int = 32
    # --- token sampling ---------------------------------------------------
    temperature: float = 0.0        # 0.0 => argmax
    top_k: int = 0                  # 0 => disabled
    top_p: float = 1.0              # 1.0 => disabled
    # --- unmasking order --------------------------------------------------
    order: Order = "low_confidence"
    # --- commit policy ----------------------------------------------------
    commit: Commit = "static"
    confidence_threshold: float = 0.9   # only read when commit == "dynamic"
    # --- correction / guidance -------------------------------------------
    sigma_scale: float = 0.0        # ReMDM-style remasking (dQwen)
    cfg_scale: float = 0.0          # classifier-free guidance (LLaDA)
    # --- reproducibility --------------------------------------------------
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

    def with_(self, **kw) -> "DecodeConfig":
        """Return a copy with fields overridden -- handy for sweeps."""
        return replace(self, **kw)


@dataclass(frozen=True)
class Preset:
    """A named DecodeConfig plus where its numbers come from.

    `provenance` is not decoration. SDAR alone ships four disagreeing sampler
    implementations with three different confidence thresholds (0.85 in their
    generate.py, 0.75 in JetEngine, 0.9 in the LMDeploy path that actually
    produced their published table), so a config without a citation is unusable.
    """

    config: DecodeConfig
    provenance: str


PRESETS: dict[str, Preset] = {
    # ---- LLaDA -----------------------------------------------------------
    # Their EVAL.md reports this triple explicitly, and shows results moving
    # 45.0 -> 50.4 across gen_length/steps/block_length -- so it must be pinned.
    "llada_official_1024": Preset(
        DecodeConfig(
            gen_length=1024, block_length=1024, steps_per_block=1024,
            temperature=0.0, order="low_confidence", commit="static",
        ),
        "ML-GSAI/LLaDA@96441d4 EVAL.md (gen_length=steps=block_length=1024)",
    ),
    "llada_official_256": Preset(
        DecodeConfig(
            gen_length=256, block_length=256, steps_per_block=256,
            temperature=0.0, order="low_confidence", commit="static",
        ),
        "ML-GSAI/LLaDA@96441d4 EVAL.md (gen_length=steps=block_length=256)",
    ),
    # ---- dQwen -----------------------------------------------------------
    # ("sdar_published", the one block-append preset, was removed with append
    # mode on 2026-08-12 -- SDAR is out of the comparator set. Its receipts,
    # should it return: block=4, steps=4, dynamic threshold=0.9 per the
    # LMDeploy path behind their published table; see third_party/LOCKFILE.md.)
    #
    # Historical champion was block-append, but a head-to-head found
    # full-canvas BEATS it: HumanEval dQwen3.5-9B append 62.20 -> full 64.63,
    # LLaDA 32.32 -> 32.93 (=published). dQwen3.5 is trained with STANDARD DLM
    # masking (random masking over the full sequence), so full-canvas is the
    # decode aligned with its training -- append was the mismatch.
    "dqwen_champion": Preset(
        DecodeConfig(
            gen_length=1024, block_length=32, steps_per_block=32,
            temperature=0.0,
            order="low_confidence", commit="static", sigma_scale=0.0,
        ),
        "gen=1024, block=32, steps_per_block=32, low_confidence, full-canvas, greedy "
        "(gen=1024 matches LLaDA; supersedes ADLMC block_append; full > append +2.43 HE)",
    ),
}


def get_preset(name: str) -> DecodeConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    return PRESETS[name].config
