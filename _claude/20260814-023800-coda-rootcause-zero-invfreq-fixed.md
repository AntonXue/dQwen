# CoDA root cause: torch 2.10 materialises meta buffers as ZEROS — the repair's "is it broken?" trigger never fired. Fixed, goldens exact.

> 2026-08-14 ~02:40, cluster. Child of 20260814-011801-DELEGATION; the
> goldens did their job in one diff. Fix verified; 19-cell rerun staged.

## Root cause (subtly different from the brief's hypothesis)

`models/coda.py`'s post-load inv_freq repair was CONDITIONAL:

    if buf is None or torch.isfinite(buf).all(): continue

On torch 2.7/x86 (workstation), meta-materialised buffers hold garbage
(non-finite) -> trigger fires -> repair runs -> correct. On torch
2.10/aarch64 (GH200), materialisation yields **ZEROS — finite — so the
repair declined to repair.** All-zero inv_freq = identical rotary phase
at every position = the model is POSITION-BLIND: weights perfect
(lm_head checksum exact), logit stats superficially sane (within a few
percent — the "silent" part), chance MMLU, degenerate repetition
(" add ad add ad"), 0.00 on all generation, and canvas-filling (a
position-blind model never places a stop — the campaign's "CoDA is 17x
slower, fills 1024 forwards every doc" observation was the FAILURE
SIGNATURE, not a protocol property; 20260814-011500's CoDA section
carries this correction). `assert_finite_rope` passed by construction:
zeros are finite. Corroborated on TWO nodes (idev + batch job 911212)
before fixing — identical inv_freq sum -0.000000 on both.

## The two fixes (both in this commit)

1. **coda.py: recompute UNCONDITIONALLY.** What a meta buffer holds is
   torch's choice per version/arch; a brokenness predicate cannot be
   trusted. Recomputation is cheap pure config data.
2. **adapter.py assert_finite_rope hardened for every family:**
   inv_freq[0] = theta**0 = 1.0 EXACTLY for every rope variant — sharp
   invariant that catches all-zeros (and any future materialisation
   surprise) where finiteness cannot.

## Verification (idev GH200, post-fix)

- inv_freq sum **5.150445 = golden EXACT**; lm_head exact (unchanged).
- Logit stats + argmax match workstation goldens **digit-for-digit**
  ('def add' mean -4.6345 std 8.0565, argmax [11,279,11,11,279]) —
  cross-hardware agreement at display precision.
- Generation coherent: 'def add(a, b):' -> ' return add(a, b)'.
- `load_matrix` over all cached families under the HARDENED gate:
  9 passed, 0 failed (4 skipped uncached).

## Rerun staged (Anton fires from a login node)

19 poisoned cells DELETED from the cluster store (`--pending` = 19);
mop-up tiles regenerated (19 one-cell lanes -> 2 jobs, 8h walls):

    bash slurm_launch.sh manifests/part2-big-table.mopup.tiles.json

Post-fix CoDA should stop properly, so cells should run FAR cheaper
than the 2.5h the broken row cost. After it lands: re-sync the 19 cells
to the workstation grid_v1_vista, re-merge, restamp the CoDA row.
Expectations per the delegation brief: mmlu 30s-40s, HE near the
PROBE-era 23.17 (canvas 1024 now).

## UPDATE ~03:50: rerun LANDED — 19/19 in ~1h (vs 12+ GPU-h broken)

Final merge: **zero INCOMPLETE, 264 groups, campaign structurally
whole at 1,154/1,154.** The restamped CoDA row:

| mmlu | gsm8k | math500 (mv) | he | he+ | mbpp | mbpp+ | fence | fence+ |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 26.21 | 3.18 | 3.20 | **23.17** | 17.07 | 35.00 | 42.33 | 36.80 | 44.18 |

- **HE 23.17 = the PROBE-era head-to-head EXACTLY** — canvas-invariant
  like LLaDA (early stop makes trailing canvas unreachable). mbpp 35 in
  their published range. Nothing 0.00; generations coherent. Post-fix
  CoDA stops properly: cells ran ~10-15 min, not 2.5h.
- ⚠ **mmlu 26.21 sits BELOW the brief's expected 30s-40s** (chance=25;
  broken row was 24.15). Load is goldens-exact and the generative row
  proves the same loaded model works, so this is NOT the silent-load
  class — but workstation should check whether a PROBE-era CoDA mmlu
  exists to compare; else it stands as a re-measurement (a 2B
  code-specialist near chance on 5-shot MMLU is believable; their paper
  never evaluates knowledge) with a disclosure line.

Resync (workstation): the 19 refreshed cells overwrite their garbage
namesakes in grid_v1_vista — same rsync line as before; then re-merge
with --store vista and the row restamps.

## Sidebar: tonight's OTHER problem, resolved

The post-campaign stalls (merge pass, first diagnosis attempts) were
the idev node's $SCRATCH Lustre client intermittently wedging after
~33h of heavy IO — fresh env-python startups hung 3/3 (zero CPU,
blocked pre-import) while single-file reads passed. Batch nodes were
unaffected (the probe job ran clean), and the client healed on its own
~02:00. Cost two wrong theories (general slowness, stdin prompt) before
/proc forensics settled it. Nothing was ever at risk: all data on
$HOME, already mirrored to the workstation.
