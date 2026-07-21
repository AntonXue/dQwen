# dQwen

Reproducible evaluation for diffusion language models — the **dQwen3.5 / dQwen3**
family alongside **LLaDA**, **Dream / Dream-Coder**, and **SDAR**, all in one
environment.

Two things it is built to answer, for any model:

- *"What does it score with the sampler its authors shipped?"*
- *"What does it score with ours?"*

Model and sampler are orthogonal, so those are cells in a cross-product rather than
one number per model.

## Design in one paragraph

A portable sampler needs exactly one thing from a model: logits for a canvas.
Everything else — mask ids, compat shims, whether the family needs an eval-side logit
shift, chat formatting — is the **adapter's** problem. Adapters expose **two** logit
surfaces: `raw_logits()` (what the model returns; native samplers use it, since they
carry their own shift) and `logits()` (position-aligned; portable samplers use it).
Each family's published decode recipe is a **preset** in one `DecodeConfig` space, not
a separate code path — a claim that `tests/parity/` is there to keep honest.

```python
from dqeval.adapter import load
from dqeval.config import DecodeConfig, get_preset
from dqeval.samplers import unified

m = load("llada-8b-base")                      # compat shims applied automatically
out = unified.generate(m, m.encode("def add(a, b):\n    "),
                       DecodeConfig(gen_length=32, block_length=32,
                                    steps_per_block=32, mode="full"))

ours = load("dqwen3.5-2b-base", revision="step30000-swa")   # checkpoints are HF revisions
```

## Why one environment is possible

dQwen3.5 requires transformers >= 5.x; every comparator pins 4.4x–4.5x, and neither
side loads on the other's version. So the comparators are **ported** — ~90 lines of
compat shims total (`dqeval/families/*/compat.py`).

Those shims are **archaeology, not invention**: each restores a behaviour transformers
4.x actually had, verified by reading it out of the native environment rather than
guessed. With torch held fixed, the ported models are **bitwise identical** to their
native environment (max|d| = 0.00e+00). All observed numerical drift is attributable
to torch 2.5.1 → 2.7.1, none to transformers — which is why `torch` is pinned exactly
in `envs/requirements-eval.txt` as part of the reproducibility contract.

Two of the three comparators had a **silent** failure mode where the model loads,
forwards, raises nothing, and returns noise. `dqeval/hf.py::assert_finite_rope()`
exists so it can never be silent again.

## Reproducibility notes

`third_party/LOCKFILE.md` pins upstream commits and — importantly — records **which
implementation produced which published table**. This is not bookkeeping pedantry:
SDAR ships four disagreeing sampler implementations with three different confidence
thresholds, their published numbers come from a fifth path (LMDeploy), and their own
reference script cannot express the greedy decoding their paper reports. Numbers we
produce for SDAR are therefore **re-measurements, not reproductions**, and are
labelled as such.

## Status

Adapters and a basic portable sampler are working for all four families. The lm-eval
harness layer is next. See `_claude/` for the running log and full evidence.
