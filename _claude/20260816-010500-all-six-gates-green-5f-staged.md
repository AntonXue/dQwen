# ALL SIX part5 gates GREEN — standard-tau is real and FREE; 5f tiled + staged

> 2026-08-16 ~01:05, the SLURM Claude. Completes the gate program of
> EVAL-REQUEST 20260815-211016 §4 / HANDOFF 20260815-214527 §2.

## The complete gate table

| cell | n_forward | pass@1 | vs anchor |
|---|--:|--:|---|
| 2B standard-s32 | =32 exact | 0.61 | real collapse (lexical), ladder's left edge |
| 2B standard-s128 | =128 exact | 7.93 | smooth recovery |
| llada standard-s32 | =32 exact | 1.83 | same cliff |
| llada standard-s128 | =128 exact | 20.12* | recovering |
| **2B standard-tau0.9** | mean 316.6, max 563 | **45.12** | **BEATS s1024's 43.29 at 3.2x fewer forwards** |
| **llada standard-tau0.9** | mean 129.6, max 357 | **33.54** | **IDENTICAL to s1024 anchor (= published 33.5) at 7.9x fewer** |

(*llada s128 read from its landed cell; all six: 164/164 samples.)

Verdicts: 4a PASS, 4b PASS (no special handling — the commit rule ran
verbatim in standard mode), 4c answered (structurally exact-N), 4d moot.
**Threshold decoding = free acceleration replicates in FULL CANVAS on a
comparator** — §4.3's flipped frontier has its headline before the sweep
even runs.

## 5f staged (NOT yet submitted)

Tiled with MEASURED standard-tau pricing (gate fwd fractions 0.31/0.127
of canvas; 0.35 conservative): **5 jobs / 75 lanes / est 660 node-h**,
worst lane 10.0h vs 16h walls, LPT, coverage-verified. Submit whenever
queue headroom allows (16 static jobs running now; 5f brings the fleet
to 21 > 20-running cap -- SLURM just queues the excess, or wait for the
first static drains):

    bash slurm_launch.sh manifests/part5f-fullcanvas-tau.tiles.json

Store math: 1,615 now; +866 static remaining -> 2,482; +418 tau -> 2,900.
