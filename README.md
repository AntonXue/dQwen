# dQwen

Reproducible evaluation for diffusion language models — the **dQwen3.5 / dQwen3**
family alongside **LLaDA** and **Dream / Dream-Coder**, all in one environment.
(SDAR loads for multiple-choice scoring, but generation needs a compat shim
that does not exist yet; see `models/sdar.py`.)

One flat entrypoint: `run.py`. One cell = (model, revision, decode, benchmark,
shard) = one provenance-stamped JSONL. Cells are independent, idempotent, and
safe to requeue — that is the whole system.

Two things it is built to answer, for any model:

- *"What does it score with the sampler its authors shipped?"*
- *"What does it score with ours?"*

Model and sampler are orthogonal, so those are cells in a cross-product rather than
one number per model.

## Running an eval cell

```bash
PY=/ssd1/ayx98/miniconda3/envs/qwen35/bin/python     # the only env that works
CUDA_VISIBLE_DEVICES=1 $PY run.py MODEL REVISION DECODE BENCHMARK [K/N]
```

Examples:

```bash
$PY run.py dqwen3.5-2b-base-v3 step50000-swa block32-tau0.8 gsm8k 2/8
$PY run.py dqwen3.5-9b-base-v3 main block32-static-s8 humaneval
$PY run.py Qwen/Qwen3.5-2B - ar gsm8k          # AR counterpart cell
$PY run.py --list                               # models, benchmarks, schemes
```

SLURM form: `$PY run.py manifest.jsonl $SLURM_ARRAY_TASK_ID`, where the
manifest is a JSONL of cells (`{"model": ..., "revision": ..., "decode": ...,
"benchmark": ..., "shard": [k, n]}` per line). One array task per line;
requeue failures freely — completed cells exit in seconds.

### The five cell fields

| field | values |
|---|---|
| MODEL | dqeval registry name (`--list`); for `ar` cells, a bare HF id |
| REVISION | HF revision (`step25000-swa`, `step50000-swa`); `main` or `-` for default |
| DECODE | `ar` · `mc-nelbo` (mmlu only) · `block32-static-sK` · `standard-static-sK` · `block32-tauT` · `standard-tauT` |
| BENCHMARK | `humaneval` · `humaneval-plus` · `mbpp` · `mbpp-plus` · `mbpp-fence` · `mbpp-plus-fence` · `gsm8k` · `math` · `mmlu` |
| K/N | stripe shard: docs `[k::n]`, `k` in `0..n-1`. Omit for the whole set |

Notes that prevent wrong numbers:

- **Shards are stripes, never chunks** — benchmark difficulty drifts with
  position, so "first N" is a biased sample. Never quote a partial shard's
  aggregate; merge all N first.
- **`gsm8k` and `math` should be sharded** (8 and 16 ways respectively);
  code benchmarks run whole.
- `mbpp` vs `mbpp-fence` are different prompts (different generations).
  The "+" columns are ordinary cells: `humaneval-plus` shares humaneval's
  prompts (denser tests), `mbpp-plus` runs EvalPlus's 378 sanitized
  problems under our scaffold via `benchmark_specs.py` (mbpp-plus section) (stock
  lm-eval `mbpp_plus` never executes the plus suite; the file carries the
  published-number comparability caveat).
- Few-shot counts, gen lengths, greedy decoding, bs=1, and the math-only
  attention backend are all fixed by the cell — nothing to remember.

### Outputs

`_runs/grid_v1/<benchmark>/<model>@<revision>__<benchmark>__<decode>__sKofN.jsonl`

One file per cell, four record kinds:

- `meta` — the cell, launch time, resolved HF repo/revision, full decode
  config, doc-set fingerprint, library versions, wall clock
- `sample` — per generated doc: raw text, truncated text, n_forward
- `lm_eval_sample` — per doc: the graded metrics
- `summary` — aggregate results; **its presence is the completeness
  sentinel** (reruns skip finished cells by checking it)

`_runs/README.md` is the directory contract: `_runs` holds raw artifacts
only (stores `grid_v1/` + `regrades/`, plus stamped launch dirs); coalescing
is `python summarize.py [filter]`, read-only.

### GPU etiquette on this box

Check `nvidia-smi` first; GPUs 0/3 are usually someone's training run.
A 2B HumanEval cell takes ~20 s/doc-shard; 9B GSM8K shards run hours.

## One section per benchmark

`benchmark_specs.py` is the entire evaluation protocol: one section per
benchmark (HUMANEVAL, MBPP_PLUS, GSM8K, MMLU, ...), each a complete task
config — prompt template, frozen few-shot examples, stops, metrics — plus
the code-grading knobs, with lm-eval as the execution engine underneath.
`tests/task_freeze_gate.py` holds the stock-derived specs byte-identical
to the pinned lm-eval (doc fingerprints + rendered prompts); run it
whenever the specs change or the pin is bumped.

## Design in one paragraph

A portable sampler needs exactly one thing from a model: logits for a canvas.
Everything else — mask ids, compat shims, whether the family needs an eval-side logit
shift, chat formatting — is the **adapter's** problem. Adapters expose **two** logit
surfaces: `raw_logits()` (what the model returns; native samplers use it, since they
carry their own shift) and `logits()` (position-aligned; portable samplers use it).
Each family's published decode recipe is a **preset** in one `DecodeConfig` space, not
a separate code path — a claim that the parity tests (`tests/test_llada_sampler.py`, `tests/test_nelbo_vs_llada.py`) is there to keep honest.

```python
from models import load
from models.samplers import DecodeConfig, generate

m = load("llada-8b-base")                      # compat shims applied automatically
out = generate(m, m.encode("def add(a, b):\n    "),
               DecodeConfig(gen_length=32, block_length=32,
                            steps_per_block=32))

ours = load("dqwen3.5-2b-base-v3", revision="step50000-swa")   # checkpoints are HF revisions
```

## Why one environment is possible

dQwen3.5 requires transformers >= 5.x; every comparator pins 4.4x–4.5x, and neither
side loads on the other's version. So the comparators are **ported** — ~90 lines of
compat shims total (the compat sections of `models/*.py`).

Those shims are **archaeology, not invention**: each restores a behaviour transformers
4.x actually had, verified by reading it out of the native environment rather than
guessed. With torch held fixed, the ported models are **bitwise identical** to their
native environment (max|d| = 0.00e+00). All observed numerical drift is attributable
to torch 2.5.1 → 2.7.1, none to transformers — which is why `torch` is pinned exactly
in `requirements.txt` as part of the reproducibility contract
(`setup_env.sh` finishes the install with two lm-eval source patches).

Two of the three comparators had a **silent** failure mode where the model loads,
forwards, raises nothing, and returns noise. `models/adapter.py::assert_finite_rope()`
exists so it can never be silent again.

## Reproducibility notes

`UPSTREAM_PINS` in `models/adapter.py` pins upstream commits and — importantly —
records **which implementation produced which published table**. This is not
bookkeeping pedantry:
SDAR ships four disagreeing sampler implementations with three different confidence
thresholds, their published numbers come from a fifth path (LMDeploy), and their own
reference script cannot express the greedy decoding their paper reports. Numbers we
produce for SDAR are therefore **re-measurements, not reproductions**, and are
labelled as such.

## Status

All four families run through the grid (`run.py`); the runner passed its
deploy gates 2026-08-12 (shard-merge lossless, template fidelity, 10/10
spot check). See `_claude/` for the running log and full evidence.
