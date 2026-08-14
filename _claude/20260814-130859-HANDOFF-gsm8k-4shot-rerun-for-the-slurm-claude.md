# HANDOFF to the SLURM Claude: the GSM8K 4-shot rerun (632 cells + 16 gate cells)

> 2026-08-14 ~13:09, workstation → cluster. You closed the campaign clean
> (`20260814-041500-HANDOFF-…`, thank you); one protocol correction reopens
> exactly one benchmark. **GSM8K moves 8-shot → 4-shot and every gsm8k cell
> re-runs.** Anton ruled it (12:19 today, reaffirmed ~13:00 after a
> 4-vs-5-shot re-litigation); the audit that produced the ruling is
> `20260814-121922-PLAN-…` (§1–§5), the code freeze is `20260814-123751-…`,
> the primary-source re-verification is `20260814-125102-…`. This doc is
> your complete execution brief — everything you need is on origin/main as
> of `8944f12`.
>
> Nothing else changes. math500 / mbpp / humaneval / mmlu cells are FINAL;
> do not touch them, do not regenerate them, no other protocol edits.

---

## 1. Why (one paragraph, receipts in 20260814-125102)

We ran `gsm8k_cot` at lm-eval's default 8-shot. Both reference protocols
use 4: the Qwen3 tech report (§3.3: "GSM8K (4-shot, CoT)" — re-fetched
from arXiv 2505.09388 today) and the LLaDA paper (70.7 @ 4-shot). Dream is
the 8-shot house, so the OLD cells aren't wasted: they're every model
measured under Dream's convention, archived as a protocol-sensitivity
record. GSM8K is about to carry the manuscript's ONLY normalized math axis
(the MATH500 AR normalization was dropped — PLAN §5.2), which is why the
outlier choice had to go. 5-shot (stock `gsm8k`'s default) was considered
and rejected ~13:00: it changes prompt format AND extractor, and it breaks
the Phase-1 anchor logic below.

## 2. What you inherit on `git pull` (a873b91 + 8944f12)

| file | change |
|---|---|
| `run.py` | `BENCH["gsm8k"].shots` 8 → **4**. This is the AUTHORITATIVE value — run.py sets `num_fewshot` on the task from BENCH; the spec's field is informational. First 4 of the 8 canonical CoT exemplars (lm-eval `first_n`), prompt machinery otherwise byte-identical. |
| `benchmark_specs.py` | GSM8K `num_fewshot` synced to 4; `metadata.version` 3.0 → **4.0** (4.0 ≡ the 4-shot protocol). |
| `tests/task_freeze_gate.py` | `SHOTS["gsm8k_cot"]` → 4. Gate re-run on the workstation: **PASSED** (vendored == pinned stock at 4-shot, byte-identical renders). |
| `summarize.py` | `merge_groups` refuses stripes whose `meta.bench.shots` disagree: prints `MIXED-PROTOCOL … refusing to merge`. If you EVER see that line, a stray 8-shot cell leaked back — investigate, don't merge. |
| `manifests/make_manifests.py` + `gsm8k-rerun-phase{1,2,3}.jsonl` | The ordered work list (16 / 84 / 540 cells). part1/2/3.jsonl regenerate byte-identical — the rerun set is additive. |
| `_runs/README.md` | documents the archive-store convention. |

Every cell's meta already records `bench.shots`, so 8- and 4-shot cells
are discriminable forever (verified 8 in all archived cells).

## 3. ⚠ STEP ZERO, before ANY run: archive your 8-shot cells

Deterministic filenames + the summary sentinel mean a rerun of an existing
cell **fast-skips in seconds**. If you skip this step, all 632 cells will
"complete" instantly and you will have a 100%-cached non-rerun.

    cd ~/foo/dQwen/_runs
    mkdir -p grid_v1_gsm8k_8shot
    mv grid_v1/gsm8k grid_v1_gsm8k_8shot/gsm8k
    ls grid_v1_gsm8k_8shot/gsm8k | wc -l    # MUST print 632

(Workstation equivalent already done: `grid_v1_vista_gsm8k_8shot/`, 632
files verified. Keep your archive permanently — it is the 8-shot side of
the sensitivity record and the one-hardware rule applies to it too.)

Sanity after the move: `python summarize.py --merge` must print **zero**
gsm8k lines and zero MIXED-PROTOCOL.

## 4. Prep (network node, per your own bring-up playbook)

    git pull                                  # -> 8944f12
    PYTHONPATH=. HF_ALLOW_CODE_EVAL=1 python tests/task_freeze_gate.py
                                              # paranoia; datasets are cached
    python run.py --prewarm manifests/gsm8k-rerun-phase1.jsonl

Prewarm notes: the two `-Base` anchors (`Qwen/Qwen3-1.7B-Base`,
`Qwen/Qwen3-0.6B-Base`) are NEW downloads (~4.4GB total); every other
model in phases 1–3 was in the campaign, so cache-hit — **unless $SCRATCH
got purged** (your handoff flagged the exposure). If prewarm shows misses
beyond the two anchors, prewarm phase2+3 manifests too before queueing.

## 5. Phase 1 — the validation gate (16 cells, RUN FIRST, everything waits)

`manifests/gsm8k-rerun-phase1.jsonl`. All AR, all unsharded, HFLM bs=16 —
one lane, well under an hour. Contents: 2 anchors + 8 bare repos at gsm8k,
plus anchor extras (mmlu / mbpp / humaneval × 2).

**Acceptance gates (strict-match, the protocol column):**

| cell | published (Qwen3 Table 8) | gate |
|---|--:|---|
| Qwen3-1.7B-Base gsm8k | **75.44** | within 3pp, else **STOP** |
| Qwen3-0.6B-Base gsm8k | **59.59** | within 3pp, else **STOP** |
| Qwen3-1.7B-Base mmlu / mbpp | 62.63 / 55.40 | report; mmlu is the purest check (likelihood, no extraction) |
| Qwen3-0.6B-Base mmlu / mbpp | 52.81 / 36.60 | report |

On a STOP: write the doc, sync nothing, don't improvise (house rule). A
fail means the extraction path is broken and finding it here costs 16
cells, not 632. Their humaneval cells have no direct single-number
published counterpart (Qwen3's "EvalPlus" is a 4-way average) — record
them, no gate.

⚠ These `-Base` repos are **CALIBRATION ANCHORS, NEVER AR counterparts**.
The counterparts remain the bare post-trained repos (manuscript §3.1 names
them). Two model sets, two roles. Do not let the anchors near a
normalization.

Context for the 8 bare retakes (no hard gate, expected movement small):
8-shot strict values were 0.8B 35.03 · 2B 58.98 · 4B 85.29 · 9B 87.64 ·
Qwen3-1.7B 73.62 · Qwen3-0.6B (see store) · Qwen2.5-7B / -Coder-7B (see
store). Known quirk that should persist: bare 2B strict > flexible
(post-trained verbosity breaks lenient extraction; strict is the column).

## 6. Phase 2 — the manuscript blocker (84 cells)

`manifests/gsm8k-rerun-phase2.jsonl`: the 14 big-table rows ×
`block32-static-s32` × 6 stripes. **The paper harvest is blocked on
exactly these**; ship them (and the Phase-1 result) back as soon as they
merge clean — don't hold them for Phase 3.

Expected sanity vs the 8-shot record (your merged numbers): dq9B@50k
73.64 · LLaDA 72.48 · Dream 75.89 · Dream-Coder 72.18 · CoDA 3.18.
4-shot should move these little (LLaDA's own 5-vs-4 delta was 0.4pp).
CoDA stays near-zero — that's real, not a regression.

## 7. Phase 3 — acceleration (540 cells, not blocking)

`manifests/gsm8k-rerun-phase3.jsonl`: 9 accel rows × 10 decodes (s32 lives
in phase 2) × 6 stripes. Feeds the §4.3 frontier figures only.

## 8. Tiling / cost (your call, your cost model)

Reuse your measured GH200 rates (they landed within 3% on the campaign);
4-shot prefill is slightly shorter, generation length unchanged, so your
gsm8k per-stripe rates hold as a mild overestimate. Manifests keep 6
stripes everywhere. The PLAN suggested 12 for 4B/9B — still your call: if
taken, override `DLM_SHARDS["gsm8k"]` locally and regenerate ONLY the
rerun manifests (never part1/2/3), keep ONE consistent n per group, and
note it in your results doc. Same QOS discipline as always (20/40, ≤16
lanes/job, 8h walls). Requeue freely — post-archive, completed 4-shot
cells fast-skip like any others.

## 9. Done = this checklist

1. Phase 1 gates PASS (both anchors within 3pp) — recorded in a doc.
2. `python summarize.py --merge`: all **104** gsm8k stripe-groups MERGED
   (14 from phase 2 + 90 from phase 3), zero INCOMPLETE, zero
   MIXED-PROTOCOL.
3. `ls _runs/grid_v1/gsm8k/*.jsonl | wc -l` = **634** (84 + 540 + 8 bare
   AR + 2 anchors; the 6 anchor extras land under mmlu/ mbpp/ humaneval/,
   for 640 new cells total).
4. Spot-open one cell: `meta.bench.shots == 4`.
5. Fresh merge snapshot (your `merged-*.txt` pattern) + a results doc in
   `_claude/` with the gate table and any deviations.
6. Push, then tell Anton it's ready to rsync — workstation pull target is
   `_runs/grid_v1_vista/gsm8k/` (currently EMPTY, awaiting exactly these
   files; the anchor extras land in their benchmark dirs alongside
   existing cells — rsync without `--delete`, as always).

## 10. Standing cautions (all from your own docs, restated so this file is self-sufficient)

- Compute nodes offline (`HF_*_OFFLINE=1` via slurm_launch.sh); sbatch
  from a login node; prep needs network.
- Lustre wedge signature: fresh env-python startups hanging at zero CPU →
  get a fresh node (your 08-14 note).
- Your idev session 907357 ended 17:01 on 08-14; start fresh.
- Stale `*.mopup.tiles.json` are harmless; ignore.
- One-hardware rule: these 640 cells are publication cells; GH200 only.

— the workstation Claude. Phase 1 gate first; everything else is routine.
