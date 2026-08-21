# HANDOFF to the SLURM Claude: C.8 extension — five more model rows at HE-164, canvas 256

> 2026-08-20 ~21:57, workstation → cluster. The big boss (via Anton) wants
> the §4.5-style decode-order material extended beyond the four models you
> already ran: the REST of the dQwen3.5 family (4B, 2B, 0.8B), CoDA, and
> the dQwen3-1.7B control — same benchmark set, same canvas, same six
> configurations. This completes the picture your 20260818 campaign
> started: decode-order behavior across SIZES within one family, plus the
> hybrid-vs-full-attention contrast (control vs 2B) and the same-backbone
> anchor (CoDA vs control) that the demo campaign dropped for page space.
> Everything below is additive to `20260818-140400-…`'s dataset; your
> RESULTS + rsync docs' conventions all carry over.

## 1. The ambition, so the data serves it

The manuscript's decode-order finding (C.8/§4.5) is currently measured on
four 7–9B-class models. The boss wants to know whether it holds:
- **across scale** — do 0.8B/2B/4B decode as near-in-order as the 9B, or
  does small-model uncertainty scatter the commit order?
- **across backbone** — control-1.7B vs dQwen3.5-2B is the width-matched
  hybrid/full-attention pair; their §4.5 rows are the first place decode
  ORDER (not accuracy) can separate the architectures.
- **across adaptation lineage** — CoDA is the same-backbone comparator
  (Qwen3-1.7B substrate, like our control), decoded by a third-party
  recipe. If its order statistics match ours, order behavior is a
  property of masked-DLM adaptation generally, not our recipe.
The deliverable stays DATA ONLY: the manuscript's scripts compute the
statistics and render the plots; per-record `passed` is annotation.

## 2. Scheme names — the boss's list vs canonical ids

The request names schemes by TOTAL steps; the records keep canonical ids
(Anton confirmed the mapping by commissioning this doc):

| boss name | canonical scheme id | commits/fwd |
|---|---|---|
| full256-s256 | `standard-static-s256` | 1 |
| full256-s128 | `standard-static-s128` | 2 |
| full256-tau0.9 | `standard-tau0.9` | adaptive |
| block16-s256 | `block16-static-s16` (16/block × 16 blocks) | 1 |
| block16-s128 | `block16-static-s8` (8/block × 16 blocks) | 2 |
| block16-tau0.9 | `block16-tau0.9` | adaptive |

I.e. exactly `schemes_for(256)` — identical to your last campaign, so
the new rows are directly comparable to the four existing ones.

## 3. The job

**5 models × 164 problems × 6 schemes × both stop variants** = 1,476
records/model (984 early-stop + 492 `_nostop`), **7,380 generations**.
On `git pull` you get one commit: `trajectories.py`'s MODELS dict now
carries all nine rows (each new entry verified to resolve in the models
catalog). Revisions: family + control at `step50000-swa` (the released
100B leg, matching the 9B row; NB the release dry-run found
`main == step50000-swa` on the hub anyway, so the pin is doubly safe),
CoDA at `main`.

    OUT=_runs/20260818-140400-c8-trajectories-he164-g256   # SAME dir --
    # the extension lands beside the existing 8 files; per-model filenames
    # cannot collide, and one dir = one dataset for the manuscript.
    python trajectories.py --model dqwen3.5-4b-base-v3 --gen 256 --problems all --out $OUT
    python trajectories.py --model dqwen3.5-4b-base-v3 --gen 256 --problems all --out $OUT --no-early-stop
    # ... same pair for dqwen3.5-2b-base-v3, dqwen3.5-0.8b-base-v3,
    #     dqwen3-1.7b-base-v3, coda-1.7b-base

Do NOT relaunch the four completed models (idempotent resume would no-op
them, but they'd still load weights — wasted lane time). Your
`--schemes` sharding and merge-with-audit machinery from `sbatch_c8`
apply unchanged if you want per-scheme parallelism; with 4 of 5 models
being ≤2B-class, five plain lanes (both variants sequential per lane)
under 4h walls likely suffices — the 4B is the slowest lane. All five
models are long-cached on $SCRATCH (campaign rows since part2).

## 4. CoDA notes (the one family with sharp edges)

- Its **untied-head gate** runs at load; a trip on GH200 is a REAL find,
  not noise (the tie fired on transformers 4.57.1 too — never assume a
  pin makes the shim removable). Post-rope-fix CoDA has been clean on
  Vista since 20260814-023800.
- Canvas 256 vs its native 768: fine for this exhibit (trajectory
  material is off-protocol by design and disclosed).
- It is the weakest model (HE ≈ 23 at protocol): expect LOW pass rates —
  that is signal, not failure; `passed=false` records are kept and
  marked, per the standing rule.

## 5. Correctness + hand-back (your own conventions, restated)

- Per-generation invariants are in-driver (s256 permutation, s128
  two-per-forward, block staircase, τ cap, progress floor). Zero trips
  over 5,904 last time; treat the first problem of the smallest lane as
  the smoke.
- Merge shards (if used) with your duplicate/consistency audit; append
  the five new canonical files beside the existing eight.
- Hand back like `20260818-233000`: per-file line counts (984/492), md5s,
  a short results doc (wall times, pass counts as context), and the same
  additive `--ignore-existing --exclude 'shards/'` rsync line for the
  workstation — the dir will then hold 13 model files + shards.

## 6. Not in scope

No grid/store writes, no statistics, no rendering, no re-runs of the
nine existing files (four models + the canvas-128 demo dir), no protocol
claims. Data only, as ever.
