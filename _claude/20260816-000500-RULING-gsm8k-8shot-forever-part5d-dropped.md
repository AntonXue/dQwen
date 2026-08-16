# RULING: GSM8K is 8-shot, PERIOD — gsm8k-4shot dropped entirely (part5d deleted, 176 cells)

> 2026-08-16 ~00:05, cluster session, Anton verbatim: "let's keep the
> 8-shot mode. Since they're so fucking heavy and acceptable anyways.
> Let's drop gsm8k-4-shot entirely." Moments earlier the opposite ruling
> ("4-shot all the way, drop the 8-shot tables") was issued and retracted
> before any file changed — recorded here so the whiplash is visible.

## The 4-shot question's full history — DO NOT RE-PROPOSE

1. 2026-08-14: 8-shot→4-shot migration PLANNED (632 cells, Phase 0
   frozen) — CANCELLED same day (20260814-133042).
2. 2026-08-15: EVAL-REQUEST revived 4-shot as twin table columns
   (bench id gsm8k-4shot, part5d, 176 cells).
3. 2026-08-16: DROPPED ENTIRELY (this doc). The existing 8-shot GATE
   cells (campaign parts 1-3 + part5c when it runs) are the gsm8k
   record, full stop.

## What changed

- `manifests/part5d-gsm8k4shot-tables.jsonl` DELETED; the part5()
  generator section no longer emits it (ruling comment inline).
- Parts 1-4 + remaining part5 manifests verified byte-identical after
  regeneration; part5 totals: **873 static + 418 tau-gated = 1,291**
  (~950 static node-h, from ~1,210 before the drop).
- `BENCH["gsm8k-4shot"]` and its summarize HEAD_TASK entry stay as
  INERT code (no manifest references them; removal is the workstation's
  call since they authored it).
- Wave plan compresses: Wave 1 = 5e + 5b small rows; Wave 2 = 5b heavy;
  Wave 3 = 5c + 5g; Wave 4 = 5f (tau-gated). Gates (5a) unchanged,
  first, on idev.

## For the manuscript side

The flipped tables' gsm8k column stays 8-shot (5c's s1024 cells); the
"consensus 4-shot" argument now applies ONLY to MATH500, which was
already the compromise's decode axis. No dual-shot disclosure needed
anywhere — there is exactly one gsm8k protocol again.
