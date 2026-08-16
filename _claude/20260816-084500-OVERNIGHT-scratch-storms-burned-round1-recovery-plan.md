# OVERNIGHT: $SCRATCH storms burned part5 round 1 — nothing lost, round 2 staged behind a weather gate

> 2026-08-16 ~08:45, the SLURM Claude. The design's promise held — zero
> data corrupted, every burned cell is cleanly pending — but the night's
> compute was largely taken by filesystem weather. Full forensics for
> Anton + the workstation + a TACC ticket.

## Timeline (log-extracted, per-minute failure histogram)

| window | event |
|---|---|
| ~23:10–04:10 | healthy: 16 static jobs running, 91 cells landed, heavy lanes mid-monster |
| 04:11–04:40 | $SCRATCH flapping begins: single-cell failures in import windows (~15 cells) |
| **05:56–06:11** | **hard outage**: ~180 cells burn in 15 min; in-flight generations die too (mmap SIGBUS class); lanes fast-fail their lists (pre-hardening scripts, no retry) |
| 06:11–08:15 | trickle + progressive job drain; monster-holders survive |
| ~08:20–08:40 | second storm: beats even retry-once — 5f (submitted ~08:20, hardened v1) burns ~400 cells through 2x attempts; login-side tooling stalls too |

Failure signatures (all one class): `ModuleNotFoundError:
multiprocessing.popen_fork` (stdlib unreadable), `PackageNotFoundError:
lm_eval` (site-packages metadata unreadable), Bus error (mmap fault on
running processes). All roads lead to the scratch Lustre client/served
side. THIRD incident in three days (08-14 idev wedge; 08-16 x2).

## Accounting (exact --pending sweep pending calm; file-count basis now)

- Landed round 1: **~91 static cells + 914339's 15 5g cells (16/16 lanes,
  its 4 last monsters completed ~08:30) + all 6 gate cells.** The gates'
  verdicts (ALL GREEN, incl. the standard-tau headline results) are
  unaffected and pushed.
- Burned/pending: everything else — order 1,150+ of part5's 1,291. All
  idempotently recoverable; zero store corruption (atomic writes held
  throughout).

## Hardening evolution (both pushed)

1. f6072ff retry-once +90s — beaten by sustained storms (measured).
2. **d47edda STORM-WAIT (current)**: pre-flight probe of the exact
   failure surface before EVERY cell; lanes stall in 180s loops until
   healthy. Walls (16h) absorb storms; lists cannot burn.

## Recovery plan (Anton, from a login node, WHEN the weather monitor says CLEAR)

A stability monitor probes the env every 4 min; it announces
"WEATHER CLEAR" after 5 consecutive healthy probes (~20 min). Then:

    cd ~/foo/dQwen && bash slurm_launch.sh \
      manifests/part5c-fullcanvas-gsm8k.tiles.json \
      manifests/part5g-headline-variants.tiles.json \
      manifests/part5b-fullcanvas-ladder.tiles.json \
      manifests/part5e-math500-block-backfill.tiles.json \
      manifests/part5f-fullcanvas-tau.tiles.json

One line, all five manifests (914339 has drained, so 5g is safe to
include): 21 jobs, every lane storm-hardened, completed cells fast-skip.
If storms recur mid-round-2, lanes STALL instead of burning — worst case
is SU-inefficiency, never list-burn.

## TACC ticket material

Dates/windows above; signatures; affected jobs 914329-914357 (round 1),
914937-914944 (5f); nodes in sacct NodeLists; the 08-14 precedent
(idev c611-081 client wedge, resolved ~02:00). Ask: scratch stability
status + whether nightly windows correlate with known maintenance/load.

## Lessons recorded

- Retry counts are calibrated to blips; storms need WAIT-not-retry.
- "Slow/failed probe" != outage (three false alarms this week); the
  failure HISTOGRAM over log timestamps is the trustworthy instrument.
- Future option if storms persist: stage env+model to node-local NVMe at
  lane start (removes scratch from the steady-state path entirely).
