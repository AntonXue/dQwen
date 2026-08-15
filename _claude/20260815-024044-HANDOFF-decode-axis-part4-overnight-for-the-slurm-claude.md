# HANDOFF to the SLURM Claude: part4 decode-axis runs — tonight's launch

> 2026-08-15 ~02:40, workstation → cluster. **This is the only live cluster
> work.** (Yesterday's 4-shot handoff `20260814-130859` was reversed the
> same day — `20260814-133042` has the stand-down; nothing from that episode
> applies. GSM8K is stock 8-shot, the campaign store is final at 1,154.)
>
> Source of truth for the *why*: the manuscript repo's
> `VistaCoder_Technical_Report/_claude/20260815-023130-EVAL-REQUEST-…`
> (decided with Anton 02:19–02:31). You likely don't have that repo on
> Vista, so this doc is **self-sufficient** — every criterion and number you
> need is inlined. Everything here is **additive**: no re-runs, no protocol
> edits, no archiving step this time (all 450 cells are new filenames; the
> deterministic-name fast-skip works FOR you, not against you).

## 1. Why (one paragraph)

§4 was restructured: decoding is now a second *axis* on the section's two
questions, which created two figure obligations. `fig:decode-retention`
(§4.1, retention vs steps for the matched pair) is drawable at ~50B today
and wants the control's ~100B leg to match Tables 2/3. `fig:decode-frontier`
(§4.3, absolute accuracy vs **measured forwards**, size-class panels)
**cannot be drawn at all**: every comparator has exactly one generation
config in the store (verified: `block32-static-s32` + `mc-nelbo` only, for
all four comparators AND `dqwen3-1.7b-base-v3@step50000-swa`), while our
nine family rows have full 11-decode sweeps.

## 2. What you get on `git pull`

`manifests/make_manifests.py` gains a `part4` section emitting four new
manifests (part1/2/3 verified byte-identical after the change):

| file | cells | contents |
|---|--:|---|
| `part4a-tau-smoke.jsonl` | 4 | τ0.9 HumanEval, one per comparator — **the gate** |
| `part4b-accel-cheap.jsonl` | 179 | control@50k + CoDA, full 11-decode sweep, HE+GSM8K+MBPP |
| `part4c-accel-big.jsonl` | 207 | LLaDA + Dream + Dream-Coder, HE+GSM8K |
| `part4d-accel-big-mbpp.jsonl` | 60 | the same three, MBPP (appendix panel) |
| `part4f-ceilings.jsonl` | 5 | `standard-static-s1024` HE for the 4 comparators + control@50k (over-run addition, see §4b) |

Counts were double-verified (independent dry run reproduced the request's
4/179/207/60 exactly; 4f is Anton's over-run extension on top of the
request — it deliberately supersedes its "no ceilings beyond the six"
line). A τ-tail phase (0.98/0.99, 252 cells) was staged and then CUT by
Anton the same night as too ambitious — if you see `part4e-tau-tail` in
git history, it is dead; the τ grid ends at 0.95. Dedup is against parts
1–3 AND earlier part4 files, so the 4 smoke cells appear only in 4a and
fast-skip inside 4b/4c.

**No new models.** All five part4 models were campaign rows — everything
should be cache-hit. `run.py --prewarm manifests/part4b-accel-cheap.jsonl`
(and 4c) should print all-cached; if $SCRATCH got purged since 08-14,
prewarm before queueing.

## 3. ⚠ PHASE 4a FIRST — the τ gate (4 cells, unsharded, minutes)

The τ (threshold-commit) rule has **never run outside dQwen3.5**. The block
decoder demonstrably works on all four comparators; thresholding is a
different commit rule against non-dQwen tokenizers and stop conventions. Do
not queue 200+ cells on an untested path.

**Pass criteria — all three, per model,** judged against that model's own
s32 HumanEval cell (reference values from the store, mean over per-sample
`n_forward`):

| model | s32 pass@1 | s32 mean n_forward | τ0.9 expectation |
|---|--:|--:|---|
| coda-1.7b-base | 23.17 | 218.7 | n_forward strictly BELOW 218.7 |
| llada-8b-base | 32.93 | 115.7 | … below 115.7 |
| dream-7b-base | 43.90 | 91.3 | … below 91.3 |
| dream-coder-7b-base | 67.07 | 91.3 | … below 91.3 |

1. **`n_forward` present on EVERY per-sample record and plausible** —
   strictly below the s32 mean (our family's τ0.9 lands at 46–73% of s32).
   ⚠ If it EQUALS the s32 value: τ is being ignored. ⚠ If it equals the
   generation length (1024): the model never emits a stop and is filling
   the canvas.
2. **Accuracy not collapsed** vs the s32 pass@1 above. A few points down is
   fine. ~0 means the commit rule is broken *for that family* — hold that
   model, not the phase.
3. **Samples logged** (standing rule: never a pass@1-only run). Missing
   sample records = stop and fix wiring first.

**Per-family verdicts:** a failing model holds ITS phase; the others
proceed independently. Report any special τ handling — it becomes a
protocol disclosure in the manuscript appendix, so it must not stay tacit.

## 4. Ordering and sizing

Run **4a**, then **4b** (both 1.7B — cheap, fills lanes while the big
models stage), with **4c queued behind. If the night runs short, 4c is the
one that matters** — it is the only phase whose figure cannot exist without
it; 4b's retention upgrade has a disclosed-caveat fallback at 50B. Then
**4d and 4f** on remaining capacity.

### 4b. Why 4f exists, and the saturation finding (Anton's pass, ~03:00)

4f gives every frontier panel its own one-at-a-time ceiling — and for
LLaDA, `standard-static-s1024` IS the published decode
(gen=steps=block=1024), so that cell doubles as a protocol-reproduction
anchor. Related finding for the figures: the τ grid SATURATES at the fast
end — 2B@50k HE has τ0.9 at 57.0 forwards and τ0.95 at 64.9 vs s32's
114.9, accuracy flat (9B likewise: 47.9 / 54.2 / 93.7, τ0.9 actually
highest at 64.63). A τ0.98/0.99 tail was considered and CUT; the
saturation is narrated as a finding, not extended with more cells.

Sizing for your cost model (which landed within 3% on the campaign): a full
11-decode sweep ≈ **5.5 s32-equivalents** per model-benchmark (fixed-step
ladder (2+4+8+16+32)/32 ≈ 1.9×, six thresholds ≈ 3.5×; τ cells run FASTER
than s32, ~half). So 4c ≈ 33 s32-equivalents across three 7–9B models on
HE+GSM8K; 4b trivial; 4d ≈ 6 s32-equivalents of MBPP. **4f** ≈ 11
HE-s32-equivalents per 7–9B cell (1024-step canvas, early-stop helps).
Tile with `plan_tiles.py` as usual; same QOS discipline (20/40, ≤16 lanes,
8h walls); requeue freely — completed cells fast-skip.

## 5. Protocol locks (publication cells — non-negotiable)

- **GATE path only**: sdpa=math, global canvas 1024, bs=1, greedy. The
  runner does all of this by default — just don't override anything.
- **Same store as the campaign** (`_runs/grid_v1/` on your side). Never mix
  hardwares in one store; these are GH200 cells like everything else.
- **Always log samples; `n_forward` on every per-sample record.** The
  frontier's x-axis is the MEAN over per-sample `n_forward` — a missing or
  constant n_forward destroys the figure silently. (The runner records it
  via the sidecar; the smoke gate exists to catch any family where it
  doesn't.)
- **Touch nothing existing**: no re-runs, no part1/2/3 edits, GSM8K stays
  stock 8-shot (the 08-14 reversal stands), no MATH500/MMLU/plus/fence
  acceleration cells, no new ceiling cells.
- The pre-08-09 n=80 stepsweep record (~19pp inflated) stays dead.

## 6. Done = this checklist

1. 4a verdict table recorded (per-model: τ0.9 n_forward vs reference,
   pass@1 vs reference, samples yes/no).
2. `run.py --pending manifests/part4X.jsonl` prints 0 for every completed
   phase.
3. `python summarize.py --merge` prints the new stripe-groups all MERGED,
   zero INCOMPLETE, zero MIXED-PROTOCOL. Expected new groups: +50 gsm8k
   (2 cheap models × 10 new decodes + 3 big × 10); mbpp +20 from 4b, +30
   from 4d. HumanEval cells are unsharded (no groups).
4. Cell-count check: **455** new files total across humaneval/ (55 = 50
   sweep incl. the 4 smoke + 5 ceilings), gsm8k/ (300), mbpp/ (100; 40
   without 4d).
5. A results doc in `_claude/` (your usual form): phases completed, store,
   the 4a table, any τ special handling, merge snapshot. Push.
6. Tell Anton it's ready to pull. Workstation side then runs its usual
   additive rsync (no --delete, tmp excluded):
   `rsync -avz --info=progress2 --exclude '*.tmp*' \`
   `  antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/grid_v1/ _runs/grid_v1_vista/`
   — identical existing files no-op; only the ~450 new cells transfer. The
   manuscript's `scripts/decode_retention.py` / `decode_frontier.py` read
   `grid_v1_vista` and take over from there.

## 7. Notes for the manuscript side (relay in your results doc)

- **τ0.95 is in every sweep** (family rows already on disk; comparators
  land tonight) — the figures should draw the full τ grid 0.5–0.95, and
  the fast-end saturation (§4b numbers) is a finding to narrate, not a
  gap. A 0.98/0.99 tail was considered and cut (Anton: not that
  ambitious); the τ grid ends at 0.95.
- **HumanEval+ under acceleration needs NO cells, ever**: HE+ shares
  HumanEval's prompts, decoding is greedy, and every cell stores raw
  samples — the "do accelerated decodes survive stricter tests" aside is a
  CPU-side regrade of saved generations.
- Deliberately NOT run (unchanged): math500/mmlu/fence acceleration (mmlu
  cannot generate; the frontier's benchmark trio is locked), the CoDA
  native-g768 cell (needs a BENCH row; still parked), τ beyond 0.95 or
  below 0.5 (tail cut; fast end already dense), non-32 block sizes
  (protocol-locked).

— the workstation Claude. Gate first, 4c is the night's payload, 4d/4f on
remaining capacity. Both §4 figures become real the moment 4c lands.
