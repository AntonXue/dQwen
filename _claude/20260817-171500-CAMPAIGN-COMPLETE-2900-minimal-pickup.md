# CAMPAIGN COMPLETE: 2,900/2,900, merge ZERO INCOMPLETE — minimal pickup for slow links

> 2026-08-17 17:15, the SLURM Claude. The 2,900th cell (control@25k
> math500 tau0.95 s1of2) landed 17:05; queue empty; tmp orphans
> scrubbed; store = exactly 2,900 jsonl files; final merge exit 0 with
> ZERO INCOMPLETE. This doc = the cheapest possible sync, tiered by what
> you already have.

## If you already ran the 2,899-sync (HANDOFF 20260817-162000 §2)

The delta is EXACTLY TWO FILES (~0.6 MB + snapshot). Fetch them
directly — no tree walk, minimal round-trips on a slow link:

    cd ~/foo/dQwen/_runs
    rsync -avz \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/grid_v1/math500/dqwen3-1.7b-base-v3@step25000-swa__math500__standard-tau0.95__s1of2.jsonl \
      grid_v1_vista/math500/
    rsync -avz \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/20260817-161500-part5-merge/merged-2900.txt \
      20260817-161500-part5-merge/

    # verify: 2,900 files; then your merge must reproduce merged-2900.txt
    find grid_v1_vista -name '*.jsonl' | wc -l          # -> 2900
    cd .. && python summarize.py --merge --store vista   # ZERO INCOMPLETE

## If you have NOT yet synced (or are unsure)

Run the full additive command from 20260817-162000 §2(a)+(b) verbatim —
it transfers only what you lack (rsync no-ops identical files; the
'*.tmp*' exclude is now moot cluster-side but harmless). Then the same
two verifications above.

## What changed since the 2,899 handoff

Only: +1 cell (completing the control@25k math500 tau0.95 pair — its
merge group and the tau0.95 curve point are now real), the tmp-orphan
scrub, and merged-2900.txt (final, zero-INCOMPLETE) added ALONGSIDE
merged-2899.txt in the same stamped dir. Everything in the 162000
handoff — content map, manuscript analysis, retention carve-out —
stands unchanged.

## Cluster end-state

Queue: empty (idev only). Store: FROZEN at 2,900 unless new requests
arrive — safe to treat as the release dataset for the flip. All code,
docs, and the campaign ledger are on origin/main.

— the SLURM Claude. 2,900 of 2,900. Go write the paper.
