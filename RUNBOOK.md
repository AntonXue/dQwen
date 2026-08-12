# RUNBOOK — running eval cells

One entrypoint: `run.py`. One cell = (model, revision, decode, benchmark,
shard) = one output file. Cells are independent, idempotent, and safe to
requeue — that is the whole system.

## The command

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

## The five cell fields

| field | values |
|---|---|
| MODEL | dqeval registry name (`--list`); for `ar` cells, a bare HF id |
| REVISION | HF revision (`step25000-swa`, `step50000-swa`); `main` or `-` for default |
| DECODE | `ar` · `mc-nelbo` (mmlu only) · `block32-static-sK` · `standard-static-sK` · `block32-tauT` · `standard-tauT` |
| BENCHMARK | `humaneval` · `mbpp` · `mbpp-fence` · `gsm8k` · `math` · `mmlu` |
| K/N | stripe shard: docs `[k::n]`, `k` in `0..n-1`. Omit for the whole set |

Notes that prevent wrong numbers:

- **Shards are stripes, never chunks** — benchmark difficulty drifts with
  position, so "first N" is a biased sample. Never quote a partial shard's
  aggregate; merge all N first.
- **`gsm8k` and `math` should be sharded** (8 and 16 ways respectively);
  code benchmarks run whole.
- `mbpp` vs `mbpp-fence` are different prompts (different generations).
  HumanEval+ is a regrade of saved generations (same prompts, denser
  tests) — no new cells. MBPP+ is NOT: it uses EvalPlus's edited
  sanitized prompts, so it needs its own generation runs.
- Few-shot counts, gen lengths, greedy decoding, bs=1, and the math-only
  attention backend are all fixed by the cell — nothing to remember.

## Outputs

`_runs/grid_v1/<benchmark>/<model>@<revision>__<benchmark>__<decode>__sKofN.jsonl`

One file per cell, four record kinds:

- `meta` — the cell, resolved HF repo/revision, full decode config,
  doc-set fingerprint, library versions, wall clock
- `sample` — per generated doc: raw text, truncated text, n_forward
- `lm_eval_sample` — per doc: the graded metrics
- `summary` — aggregate results; **its presence is the completeness
  sentinel** (reruns skip finished cells by checking it)

## Directory contract

Everything in `_runs/` is **raw** (see `_runs/README.md` for the full map):

- Stores: `grid_v1/` (generation cells) and `regrades/` (CPU regrades,
  `<yyyymmdd-hhmmss>__<model>@<revision>__<benchmark>-plus__from-<source>.jsonl`,
  written there by `dqeval/evalplus_driver.py --from` automatically).
- Everything else is a launch dir, `<yyyymmdd-hhmmss>-<description>/`:
  lane scripts, console logs, smoke scratch of one launch, frozen after.
- Coalescing is read-only and lives outside `_runs`:
  `python summarize.py [filter]` prints one line per cell (grid + regrades).
- Cell filenames carry no timestamp on purpose — the filename is the cell's
  identity, which is what makes reruns idempotent and lets shards land from
  different nodes into one tree. The launch time is `launched_at` in each
  cell's `meta` record, and `summarize.py` prints it.

## GPU etiquette on this box

Check `nvidia-smi` first; GPUs 0/3 are usually someone's training run.
A 2B HumanEval cell takes ~20 s/doc-shard; 9B GSM8K shards run hours.
