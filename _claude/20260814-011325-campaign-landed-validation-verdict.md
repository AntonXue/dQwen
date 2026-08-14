# Campaign landed: 1,154/1,154 cells; validation verdict; one red flag (CoDA)

> 2026-08-14. Vista store rsync'd to _runs/grid_v1_vista/ (278M; the
> corrected rsync -- separate store, NEVER merged into grid_v1, which
> remains the workstation PROBE record the validation diff needs).
> summarize.py/--merge grew `--store` so both stores read cleanly.

## Part-1 validation (24 shared AR cells, cluster vs workstation)

- MMLU: exact to <=0.18pp on all six models -- the loglikelihood path is
  the cross-hardware anchor, and it holds.
- HumanEval/MBPP: single-problem flips (+-1.8pp = 3/164 docs), both
  directions, mean ~0 -- inside the +-2pp cross-hardware calibration.
- GSM8K: the PREDICTED canvas-fix recovery, cleanly attributed: 2B +5.38
  (the "post-trained verbosity" model whose CoT was truncated at the old
  256 AR cap), 0.6B +1.74, 1.7B +1.14, 4B/9B ~0 (concise answers fit
  256). Direction and ordering exactly as forecast when _ARLM landed.
VERDICT: SLURM path validated; deltas decompose into hardware noise
(mean ~0) + the disclosed canvas fix. Table-4 AR band restamps from
vista (NB 2B gsm8k 53.60 -> 58.98 moves the S4.1 fraction denominator).

## Big-table preview (block32-static-s32, canvas 1024, all merged/whole)

Family + comparators coherent with PROBE +-2pp (LLaDA mmlu 65.88 vs
their published 65.9; Dream gsm8k 75.89 vs published 75.82; Dream mbpp
56.6). control@50k row EXISTS for the first time (mmlu 48.12, gsm8k
39.95, HE 43.29 -- the 20260811-203500 obligation delivered). LLaDA HE
32.93 identical at canvas 1024 vs 512: expected under OUR protocol
(stop-string early termination makes trailing canvas unreachable; their
own 35.4-vs-32.9 canvas ablation applies to THEIR no-early-stop decode).
control@50k and 2B@50k tie on HE/HE+ (43.29/37.80): VERIFIED genuine
coincidence (different repos, different generation hashes).

## ⚠ RED FLAG: every CoDA cell is garbage

CoDA row: mmlu 24.15 (= chance), all generation benchmarks 0.00 -- the
signature of the silent-load failure class (noise logits, nothing
raises). The cluster session ran parity for LLaDA only; load_matrix was
never run for CoDA on GH200/torch-2.10. CoDA's shims are the fragile
ones (untied-head restoration + inv_freq recompute). NEEDS: cluster-side
load_matrix --only coda-1.7b-base, fix, rerun CoDA's 19 cells. Until
then the CoDA row stays PROBE/workstation or blank.

## Next

1. Cluster: diagnose CoDA, rerun its cells, re-sync.
2. Manuscript harvest: refresh benchmark-grid.tex from vista (family,
   comparators, control, AR band incl. gsm8k denominators), math500
   column + disclosures, then the acceleration figures from part3.
