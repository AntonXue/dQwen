# `_claude/` — dQwen working log & progression

**Project goal.** A **release-quality evaluation package** for the dQwen3.5 / dQwen3
family and its diffusion-LM comparators (LLaDA, Dream / Dream-Coder, SDAR), all under
**one environment**, with lm-eval-harness on top. Two things it must support:

1. Load any model and call **its own shipped sampler** — "the number with LLaDA's own
   sampler".
2. Run **our sampler** on the same weights — "the number with ours". Model x sampler
   is a cross-product, not one number per model.

Plus reproducing the numbers those papers claim, so our comparison tables are
defensible rather than cited.

**Why this dir.** The finer-grained running timeline — what was run, the numbers, the
decisions and their rationale, the negatives — at low ceremony, so a fresh Claude (or
future-me) on a different machine can reconstruct the *progression* without git
archaeology or chat scrollback. Mirrors `~/foo/ADLMC/_claude/`.

## Conventions
- **Filename:** `yyyymmdd-hhmmss-short-description.md`. One topic per file.
- **Append-only:** don't rewrite old entries; add a new file that supersedes and
  reference the old one by name. The directory IS the timeline.
- **Write for the next reader:** 1–2 lines of context, then findings. Tables > prose
  for numbers. Always say *what was run*, *where artifacts live* (paths / HF repo ids),
  and *what to do next*.
- **Record refuted priors explicitly.** Several load-bearing design choices here exist
  because a measurement contradicted an earlier belief; without the refutation written
  down, the next reader re-proposes the wrong thing.
- Cold-start reading order: this dir in filename order → `third_party/LOCKFILE.md`
  (which upstream implementation produced which published table) → the auto-memory
  index at `~/.claude/projects/-home-ayx98-foo-ADLMC/memory/MEMORY.md`.

## Sister projects
- `~/foo/ADLMC/` — the research repo this is extracted from. Training, ablations, and
  the eval prehistory (`ablations/paper_evals/`, `eval/`). Its `_claude/` holds the
  dQwen3.5 family training history and the eval-standardization decisions.

## Index (update when adding files)

| file | what it holds |
|---|---|
| `20260721-184634-dqeval-bootstrap-port-evidence-adapters-samplers.md` | ⭐⭐ **Repo bootstrap + the port evidence.** Our models need tf 5.13, comparators pin 4.4x–4.5x, and the sets are **disjoint** — so LLaDA/Dream/SDAR were **ported** with ~90 lines of shims (3/4/3, zero invented). **Decomposed the drift: 100% torch, 0% transformers ⇒ the port is BITWISE-exact and the acceptance gate is bitwise, not a tolerance** (an earlier tolerance recommendation came from a confounded comparison — do not reintroduce). **Silent failure class** hit 2/3 families: tf5 meta-device wipes non-persistent rotary `inv_freq`, tf5's repair lives in base `_init_weights` which Dream and SDAR both override ⇒ model loads, forwards, returns noise, raises nothing (argmax 20–40%) — hence mandatory goldens + `assert_finite_rope`. **Two logit surfaces** (`raw_logits`/`logits`): Dream alone is AR-aligned and shifts in its own sampler AND eval wrapper, so canonicalising for its native sampler would double-shift silently. **Two priors refuted:** Dream's sampler is NOT HF-coupled (plain class, 6/7 bit-identical), and SDAR DOES ship a sampler (`JetAstra/SDAR/generate.py`, runnable on HF weights via `store_kv`). **SDAR minefield:** four disagreeing implementations, thresholds 0.85/0.75/**0.9**, published table came from **LMDeploy**, and their own script can't do greedy ⇒ our SDAR numbers are re-measurements. **Built + GPU-verified:** all four families load through one adapter contract (incl. dQwen3.5 at `revision=step30000-swa`), and the unified bs=1 sampler runs on all four — **token-for-token identical to LLaDA's own `generate.py`** at matched config, which validates "native decode is a preset, not a code path". |
