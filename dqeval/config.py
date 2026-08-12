"""Decode configuration: the one knob space every sampler is parameterised by.

The claim this file encodes is that LLaDA, Dream and dQwen do not need
different decode APIs -- their published samplers are points in the space
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
    # Decoding is always full-canvas: the whole answer canvas is visible from
    # the start and blocks are revealed left-to-right (LLaDA/Dream native).
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

    `provenance` is not decoration: upstream repos ship multiple disagreeing
    sampler configs, so a preset without a citation is unusable.
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
    "dqwen_champion": Preset(
        DecodeConfig(
            gen_length=1024, block_length=32, steps_per_block=32,
            temperature=0.0,
            order="low_confidence", commit="static", sigma_scale=0.0,
        ),
        "gen=1024 (matches LLaDA), block=32, steps_per_block=32, "
        "low_confidence, greedy",
    ),
}


def get_preset(name: str) -> DecodeConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    return PRESETS[name].config
