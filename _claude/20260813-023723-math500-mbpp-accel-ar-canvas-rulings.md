# Evening rulings executed: MATH500, MBPP acceleration, AR canvas, 250-doc shards

> 2026-08-13, second session (workstation), reviewing the cluster's
> manifests before greenlight. Anton's rulings + one review finding, all
> implemented; NOTHING sbatch'd yet.

## Rulings

1. **MATH500 replaces full MATH-5000** (benchmark key `math500`, task
   `minerva_math500`, dataset HuggingFaceH4/MATH-500, SAME Minerva
   protocol: 4-shot fixed prompt, imported extraction, strict +
   math_verify). Deciding argument: the column's direct-reproduction
   anchor was ALREADY spent by the 08-12 math_verify ruling — comparator
   published numbers are strict-metric full-set either way — so the
   marginal comparability cost is one clause in an existing disclosure,
   while the saving is the campaign's biggest line (~150 -> ~15 GPU-h).
   VERIFIED FIRST against primary sources: LLaDA EVAL.md and Dream's
   eval_dream_gen.sh both run full minerva_math (no --limit) — so
   LLaDA/Dream MATH cells become subset re-measurements, published
   numbers context-only; and n=500 stderr ~2.2pp means no narrating
   sub-2pp MATH deltas. Both caveats belong in the protocol appendix.
2. **Acceleration grid runs MBPP too** (HE + GSM8K + MBPP frontier).
3. **~250 docs per DLM generation cell** (was ~200): mbpp of2, gsm8k of6,
   math500 of2, mbpp± of2, he whole. Fatter shards, fewer tasks; worst
   shapes still ~2x under the (now 6h) wall.
4. **NO v1-mixture arm** — S4.6 stays workstation-stamped, provenance
   note in the manuscript.
5. **part2 += full @25k rows for 0.8B/4B/9B** (S4.2 budget-trade
   symmetry; over-run ethos). part1 += fence benchmarks (AR rows possible
   in the format table).
6. **AR: no sharding (already policy) + greedy confirmed at source**
   (every generative spec sets do_sample=False; minerva pins temp 0).

## The review finding, fixed: AR canvas wiring

lm-eval 0.4.8's HFLM hardcodes max_gen_toks=256 (property, no init
param); only humaneval's spec overrides it. The 1024-canvas ruling was
therefore silently skipping the AR side — fatal on MATH (8.5% of golds
exceed 512 tokens; truncation grades zero) and quietly unfair under the
S4.1 normalized fractions (DLM 1024 vs AR 256). run.py now has _ARLM, an
8-line HFLM subclass wiring BENCH gen into the cap; verified
max_gen_toks=1024 and smoked on a live AR math500 cell. Consequence:
cluster AR MATH/MBPP numbers will sit above the 256-cap workstation
batch — expected, directional, disclosed.

## Regenerated campaign (all three manifests, one generator)

part1 72 (8 AR x 9 benchmarks, unsharded) · part2 272 (14 rows x 19
cells + 6-row ceiling) · part3 810 (9 rows x 11 decodes x
{he,gsm8k-of6,mbpp-of2}, deduped vs part2) = **1,154 cells, ~420-450
GPU-h, under a day at %32**. Structural validation: no dupes, complete
disjoint shard groups, zero part2/part3 overlap, no stale `math` keys,
part1 bare-repos + unsharded. Both gates PASS (freeze gate registers
minerva_math500 as a variant: 500 docs, clean renders). Smokes: AR +
DLM math500 stripes ran end-to-end; sbatch wall bumped to 06:00:00;
--list tuples now show tau 0.95 and standard s1024.

Cluster Claude: `git pull`, re-run `run.py --prewarm` on each manifest
(picks up HuggingFaceH4/MATH-500), then part 1 fires first.
