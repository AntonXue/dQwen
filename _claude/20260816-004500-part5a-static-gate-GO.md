# Part5a static gate: GO — standard-static-sN spends EXACTLY N forwards; s32 collapse is real, not wiring

> 2026-08-16 ~00:45 (cluster). 2B@50k verdicts; LLaDA statics running as
> anchors (same family-independent code path; its s1024 already matches
> published 33.5). τ cells follow — they gate ONLY 5f.

| cell | n_forward | pass@1 | reading |
|---|---|--:|---|
| standard-s32 | =32 exactly, all 164 | 0.61 | REAL collapse (outputs degraded-lexical, not garbage) — the ladder's left edge |
| standard-s128 | =128 exactly | 7.93 | smooth recovery toward s1024's 43.29 |

Confirms the handoff §2.3 structural claim empirically: stop check is
block-granular, standard = one block ⇒ zero early exit ⇒ static
full-canvas cost is arithmetic (docs × N). Waves 1–2 (5e + 5b, 16 jobs,
~780 node-h) launch on this verdict; 5c+5g tranche (13 jobs) staged
behind; 5f awaits its τ gate cells.
