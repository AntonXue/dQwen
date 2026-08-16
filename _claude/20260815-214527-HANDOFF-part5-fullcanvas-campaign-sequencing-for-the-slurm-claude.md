# HANDOFF to the SLURM Claude: part5, the full-canvas campaign — scope, sequencing, tiling

> ⚠ **AMENDED 2026-08-15 21:54 by the manuscript-side Claude** (the session in
> `~/foo/VistaCoder_Technical_Report`, not the author of this doc), at Anton's
> instruction. **See §3b** — it adds 84 MBPP-variant cells and one unowned
> regrade pass, both table-blocking. Everything else is untouched.

> 2026-08-15 ~21:45, workstation → cluster. Executes EVAL-REQUEST
> `20260815-211016` **as amended by Anton's compromise ruling** (~21:30,
> after digesting the cost): **MATH500 carries the decode axes; GSM8K's
> role shrinks to big-table cells.** Where this doc and the request
> disagree, THIS DOC WINS — the request's §1 (why) and §4 gates stand;
> its §3/§5 scope tables are superseded. Cost is pre-authorized by Anton
> ("I don't give a shit how much compute…"), but the compromise cut it to
> ~1/3 anyway. This is a MULTI-DAY campaign, not an overnight — plan in
> waves.

## 0. The compromise, in one table (and the data that motivated it)

| axis | benchmark | why |
|---|---|---|
| decode sweeps (retention, frontier, full-canvas ladder) | **MATH500** | consensus 4-shot (Qwen3 + LLaDA convention, no dual-shot needed); 500 docs not 1,319; and our family ALIGNS there — at block32-s32 the 9B **leads all comparators** (mv 37.40 vs Dream 35.00 / Dream-Coder 30.20 / LLaDA 27.20) where on gsm8k it trails all of them (70.58 vs 72.2–75.9) |
| big tables | GSM8K both shot counts + everything else | 4-shot twins are cheap at table decodes; 8-shot stays untouched |

GSM8K is deliberately ABSENT from the full-canvas ladder; it keeps one
full-canvas config (s1024, the expected headline) in both shot counts so
the flipped big table keeps its column.

## 1. What you get on `git pull` (parts 1–4 verified byte-identical)

| file | cells | contents |
|---|--:|---|
| `part5a-fullcanvas-gates.jsonl` | 6 | **run FIRST** — s32/s128/τ0.9 in standard mode, HE, for 2B@50k + LLaDA |
| `part5b-fullcanvas-ladder.jsonl` | 405 | 14 rows × `standard-static-s{32..1024}` × HE+MBPP+MATH500 (11 already-run s1024 HE ceilings + 4 gate cells deduped out) |
| `part5c-fullcanvas-gsm8k.jsonl` | 84 | 14 rows × `standard-static-s1024` × gsm8k (8-shot) |
| `part5d-gsm8k4shot-tables.jsonl` | 176 | **NEW bench id `gsm8k-4shot`**: 14 rows × {block32-s32, standard-s1024} × 6 stripes + 8 AR cells **UNSHARDED** (the request's "8×6 shards" violated the AR batch-composition policy; corrected) |
| `part5e-math500-block-backfill.jsonl` | 280 | 14 rows × the 10 non-s32 block decodes × math500 — the decode figures' math axis, block side |
| `part5f-fullcanvas-tau.jsonl` | 418 | `standard-tau{0.5..0.95}` × 14 rows × HE+MBPP+MATH500 — **GATED on 5a's τ cell** |

Static total 951; with the τ half 1,369. Code changes you also get:
`BENCH["gsm8k-4shot"]` (same task, `shots=4` — distinct id ⇒ distinct
filenames/merge-groups, so the MIXED-PROTOCOL guard never trips; the
4-exemplar rendering was verified byte-identical to stock-at-4 in the
20260814-123751 episode) and the `summarize.py` HEAD_TASK entry. No new
models, no new datasets — prewarm should be a full cache-hit pass.

## 2. Gates (5a, six cells, idev) — everything queues behind them

1. **s32/s128 (4 cells): does `standard-static-sN` respect N?** Pass =
   measured `n_forward` ≈ N exactly, accuracy degrades smoothly (no ~0),
   samples logged. If s32 reports ~1024, the ladder is one config run six
   times — STOP.
2. **τ0.9 (2 cells): does `standard-tau` run at all?** If the config
   errors, report and DROP part5f entirely — do not implement a sampler
   to satisfy the request. If it runs: same n_forward sanity as part4's
   gate (strictly below that model's s1024 = 1024, not equal to canvas).
3. Source-side pre-answer for the request's gate 4c, to confirm
   empirically: the stop-string check is **block-granular**
   (`models/samplers.py:218`), and standard mode is one block — so
   `standard-static-sN` spends **exactly N forwards structurally** (the
   1024.0s in the existing ceilings are measured AND nominal at once; the
   paper's block-vs-sequential cost denominators are safe).
   `standard-tau`, by contrast, CAN exit early inside the canvas via the
   all-committed break — its n_forward is real signal, record it.

## 3. Sequencing + tiling (your call on details; the shape that makes sense)

**The static cells are exactly arithmetically priced** — no early exit ⇒
forwards per shard-cell = docs × N, zero variance. You can tile 5b/5c/5d's
full-canvas half by pure arithmetic with per-model ms/forward anchors from
the part4 store (your 1.4×-underprice lesson was a τ-early-stop artifact;
it cannot recur on static cells). Forward counts per ROW:

- 5b ladder: HE 164×2016 + MBPP 500×2016 + MATH500 500×2016 ≈ **2.35M
  forwards/row** (s1024 alone is half of it).
- 5c: 1,319 × 1024 ≈ **1.35M/row** — as heavy as half a ladder row.
- 5d: the s1024-gsm8k-4shot half is another 1.35M/row; the block32-s32
  half and 5e are early-stopping block cells, cheap by comparison.

Suggested waves under the qgh QOS (20 running / 40 submitted, 8h walls):

1. **Wave 0**: gates on idev (minutes-to-an-hour).
2. **Wave 1** (cheap, immediate figure value): 5e backfill + 5d's block
   half + 5b's small rows (0.8B, 1.7B×3, 2B×2) — the block-side MATH500
   decode figures become drawable as soon as 5e lands.
3. **Wave 2** (the payload): 5b's heavy rows (4B/9B × 2 ckpts, the three
   7–8B comparators). Within a row, queue s32→s512 before s1024 so
   partial results are usable early; s1024 completes the row.
4. **Wave 3**: 5c + 5d's full-canvas half (the two gsm8k-s1024 blocks —
   together they rival 5b's heavy half; nothing draws from them until
   the table flip, so they must not displace Wave 2).
5. **Wave 4**: 5f if the τ gate passed. Spillover-tolerant like part4e
   was; cut it first if anything must give.

Requeue freely throughout — deterministic names + the summary sentinel
make every completed cell a fast-skip.

## 3b. AMENDMENT — the MBPP variant family under full canvas

> ⚠ **Edited into this doc by a DIFFERENT Claude**: the manuscript-side
> session working in `~/foo/VistaCoder_Technical_Report`, on **2026-08-15
> ~21:54**, at Anton's instruction after a paper-side validation pass of
> part5. Nothing above this section was changed. If this contradicts your own
> plan, the manuscript side is the one that needs the numbers — but say so
> rather than silently reconciling.
>
> Anton's rulings from that pass, for context: the deduped HumanEval s1024
> skips are **fine** (no archive-and-rerun needed); the MATH500 metric
> instability is a **known accepted risk**; the retention claim will be
> **recomputed** against full-canvas rather than carried over; and he wants
> **fence and non-fence MBPP and MBPP+ in standard decode too**, which is
> what this section specifies.

### The correction that makes this necessary

The manuscript side assumed the `+` columns came free by CPU regrade under
full canvas. **That is only true for HumanEval+.** Checked against
`benchmark_specs.py`:

| bench | task | shares stock prompt? | regradeable? |
|---|---|---|---|
| `humaneval-plus` | `humaneval_plus_sound` | **yes** — "same 164 problems and prompts as humaneval" | ✅ free regrade |
| `mbpp-plus` | `mbpp_plus_full` | **no** — `evalplus/mbppplus`, 378 problems, *edited prompts* | ❌ needs generations |
| `mbpp-fence` | `mbpp_ticks` | **no** — fence template replaces `[BEGIN]`/`[DONE]` | ❌ needs generations |
| `mbpp-plus-fence` | `mbpp_plus_ticks` | **no** — both of the above | ❌ needs generations |

So **`mbpp-plus` is not optional**: MBPP+ is a column in the main table, the
size-class table and the appendix grid. Without it those tables cannot flip to
full-canvas at all, which is the entire point of part5.

### What to add — 84 cells at the headline

At **`standard-static-s1024` only** (the flipped tables' headline), 14 rows,
existing shard counts (`mbpp*` = 2):

```
mbpp-plus         14 x 1 x 2 =  28   REQUIRED — main-table column
mbpp-fence        14 x 1 x 2 =  28   Anton: fence wanted in standard mode
mbpp-plus-fence   14 x 1 x 2 =  28   Anton: fence wanted in standard mode
                                ---
                                 84
```

Cost is ~500 docs × 1024 forwards per shard-pair per row, i.e. comparable to
one 5b MATH500 row per benchmark. Small next to 5b.

**Deliberately NOT the full ladder.** Extending these three across
`s{32..512}` too would be +420 cells and **no current figure needs it** — the
decode figures use HumanEval, MBPP and MATH500 only, and fence is an appendix
format-sensitivity aside that needs the headline alone. If someone later wants
fence decode curves, that is a separate ask.

### Also required, and currently unowned

**Run the HumanEval+ regrade** over 5b's saved full-canvas generations. It is
CPU-only and the mechanism is proven (`run.py:428` keeps response text; ten
`regrade` cells already exist in the store from the 2026-08-12 pass), but no
manifest or checklist item in part5 covers doing it. Two table columns
(HumanEval+ across the flipped tables) depend on it existing.

---

## 4. Protocol locks (unchanged; publication cells)

GATE path only (sdpa=math, canvas 1024, bs=1, greedy); same store
(`grid_v1`), one-hardware rule; ALWAYS log samples; `n_forward` on every
per-sample record (the cost axes are its mean); the n=80 stepsweep record
stays dead; **8-shot gsm8k cells are untouched** — nothing is archived,
nothing moves, the two shot counts coexist by benchmark id.

## 5. Report back

1. Gate verdicts (n_forward≈N table; τ existence + its measured
   n_forward; the empirical 4c confirmation).
2. Zero MIXED-PROTOCOL in `summarize.py --merge` with both gsm8k ids in
   the store — the coexistence check.
3. **Dream under full canvas is the one to watch** (request §8.5): if it
   needs any special handling, that is a protocol disclosure the paper
   must carry. Also flag qualitative full-canvas-vs-block surprises on
   any family.
4. Per-wave completion notes + the usual merge snapshot; expected new
   stripe-groups when everything lands: 5b 14×6×2 benches... concretely
   every group MERGED, and file counts per manifest matching `--pending`
   = 0.
5. Expectation-setting we already hold: LLaDA s1024 HE = 33.54 vs their
   published 33.5 (the anchor); Dream s512 HE should NARROW the 43.90 vs
   57.9 gap, not close it (their published numbers also use temp-0.2
   top-p sampling and their own alg variants — ours stays greedy).
6. **[added 2026-08-15 21:54 by the manuscript-side Claude, see §3b]** The 84
   MBPP-variant cells, and confirmation that the HumanEval+ regrade pass has
   been run over the full-canvas generations. Both are table-blocking.

— the workstation Claude. Gates, then waves; MATH500 is the axis, GSM8K
is table-only, and no existing cell moves.
