# Cluster bring-up green + the campaign as three manifests (pushed for review, no sbatch fired)

> 2026-08-13, first working session ON Vista (idev GH200 node, job 907357).
> Anton's rulings inline. Flow he set: manifests reviewed on the other
> machine, greenlight, THEN arrays fire.

## Bring-up (all green this session)

- setup_env.sh harness layer into the cluster qwen35 env (hand-built torch
  2.10.0+cu129 aarch64 untouched; tf 5.13.0; `lm_eval import OK`). ⚠ pip
  flags antlr4-python3-runtime 4.11 (math_verify dep) vs omegaconf's 4.9
  pin — qwen35 doubles as the TRAINING env; check hydra/omegaconf before
  training in it again.
- tests/structure_gate.py PASSED; tests/task_freeze_gate.py PASSED
  ("vendored == pinned stock", all 70 leaves) on the cluster. Trap: the
  freeze gate needs HF_ALLOW_CODE_EVAL=1 exported — without it, it dies in
  the code_eval banner and `tail` in a pipe hides the exit code.
- Discovery: Vista COMPUTE nodes have outbound network (curl huggingface.co
  → 200). Offline discipline in sbatch_cells.sh stays (no 800-task Hub
  stampedes), but prewarm can run from an idev shell, not just login nodes.
- $SCRATCH HF cache (1.8T) was missing LLaDA-8B-Base entirely, had the
  Dream repos only in a legacy top-level layout invisible to hub/ lookups,
  and lacked several EER6b step revisions. `run.py --prewarm` over all
  three manifests (running as this is written; part 1 already "prewarm
  complete") fixes exactly that, into the same HF_HOME sbatch_cells.sh uses.

## The campaign = manifests/ (generator + three .jsonl, one launch each)

Anton's three-part recollection CONFIRMED against the paper repo
(VistaCoder_Technical_Report/_claude/: FINDING-20260811-171500,
EVAL-REQUEST-20260811-190000, HANDOFF-20260811-134555, 20260811-203500):

| manifest | composition | cells |
|---|---|--:|
| part1-ar-baselines | 8 AR twins (bare repos, NEVER `-Base`) × 7 benchmarks, decode=ar, UNSHARDED. Cluster GATE restamp of the 2026-08-12 workstation batch — and the validation launch (every number diffable against a fresh workstation counterpart). | 56 |
| part2-big-table | 11 DLM rows — family v3@50k ×4, width-matched pair @25k, control@50k (the 20260811-203500 obligation), 4 comparators — × block32-static-s32 on the 8 generation benchmarks + mc-nelbo mmlu; plus the 6-row standard-static-s512 HumanEval ceiling. | 611 |
| part3-acceleration | 9 rows (family @50k + @25k, control @25k) × static s{32,16,8,4,2} + τ{0.5,0.6,0.7,0.8,0.9,0.95} × {humaneval, gsm8k}, deduped against part2 so both can queue at once. | 837 |

## Rulings (Anton, 2026-08-13)

1. **Part 3 runs HumanEval AND GSM8K in full** (the 08-11 spec was HE-only).
2. **Full-canvas sequential stays a stretch tier** — 6-row HE ceiling, not a
   second big-table column.
3. **iLLaDA / DiffuCoder / MDM-1.7B ports deferred** — campaign runs the 4
   ported families.
4. **Part 1 fires first** and doubles as the SLURM-path validation.
5. **Shard rule: no DLM cell generates more than ~200 docs** → gsm8k of8,
   math of32, mbpp/mbpp-fence of4, mbpp-plus/± of2; humaneval whole (164).
   mmlu EXEMPT and whole: mc-nelbo generates nothing — MMLU is all
   single-token continuations, which take the shared-forward path (~1
   forward/doc), so a whole cell is ~14k cheap forwards. AR cells stay
   UNSHARDED (the 0.23pp batch-composition policy). BENCH suggested shards
   + README updated to match.
6. **MATH = the full 5000-problem Minerva 4-shot set**, not MATH500 —
   MATH500 is the reasoning/RL-era convention and not comparable to the
   LLaDA/Dream tables we reproduce against.

## Open at greenlight time

- **gen lengths** (Anton wants discussion): current 512 code / 1024
  gsm8k+math. Recommendation: keep — they match or exceed stock protocols
  and all PROBE-era comparisons used them. ⚠ gen is NOT in the cell
  filename, so it must be settled BEFORE launch; changing it later means
  new benchmark rows, not edits.
- CoDA's published-protocol HE cell (768 tokens @ 1/step) is not
  expressible today (gen is fixed per benchmark) — would be a new BENCH
  row (e.g. humaneval-g768) if we want the ~6pp-undersell armor.
- Parity gates + probe cells on this idev GPU once LLaDA lands; probe
  timings also sanity-check the 4h array walltime (math of32 was sized
  with that wall in mind).
- The v1-mixture arm (§4.6's 19/20 table) is in NO part — decide whether
  those numbers need cluster restamps or stay workstation-stamped.
