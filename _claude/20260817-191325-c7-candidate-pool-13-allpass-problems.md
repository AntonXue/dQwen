# C.7 candidate pool: the 13 all-pass HumanEval problems (selection receipted BEFORE any trajectory exists)

> 2026-08-17 ~19:13, workstation. Anton's ruling this session tightened the
> C.7 selection rule (manuscript EVAL-REQUEST `20260817-151600` §4) from
> "solved by most models" to **"solved by ALL models under ALL decode
> configs"** — panels must show real decoding of correct solutions
> everywhere. This doc freezes the pool and its receipts NOW, before a
> single trajectory has been generated, so the selection can never be
> accused of chasing scatter. Anton picks finalists from screening plots
> later; the pool itself is closed.

## The funnel: 164 → 27 → 13

1. **All-pass intersection (164 → 27).** Per-doc pass/fail read from the
   GATE store (`_runs/grid_v1_vista/humaneval/`), 20 cells = 5 models ×
   4 decode configs — the canvas-1024 analogues of the C.7 grid:

   | C.7 scheme (canvas 128) | receipt config (canvas 1024) |
   |---|---|
   | block32-static-s32 | block32-static-s32 (identical per-block behavior) |
   | block32-tau⟨T⟩ | block32-tau0.9 |
   | standard-static-s128 (1 tok/forward) | standard-static-s1024 (1 tok/forward) |
   | standard-tau⟨T⟩ | standard-tau0.9 |

   Models: dqwen3.5-9B@50k, control-1.7B@50k, llada-8b, dream-7b,
   dream-coder-7b. Per-model pass counts across the four configs:
   9B 102–106, control 68–71, **llada 54–55 (the binding constraint)**,
   dream 70–73, dream-coder 106–110. Intersection of all 20 sets:
   **27 problems** — docs 0, 3, 4, 8, 12, 13, 21, 22, 23, 27, 28, 29, 30,
   31, 34, 35, 40, 45, 48, 50, 53, 55, 56, 58, 60, 61, 86.

2. **Solution-length band 30–100 tokens (27 → 13)** — the request's
   legibility criterion (fills a 128 canvas without running out; kills
   the trivial one-liners like `strlen`/`add`). Token counts via the
   Qwen3 tokenizer on `canonical_solution`.

## THE POOL (13, all receipts attached)

| doc | task | entry point | sol tok | structure |
|--:|---|---|--:|---|
| 0 | HumanEval/0 | has_close_elements | 50 | nested loop + cond |
| 3 | HumanEval/3 | below_zero | 32 | running-sum loop + cond |
| 8 | HumanEval/8 | sum_product | 40 | loop |
| 12 | HumanEval/12 | longest | 39 | loop + cond |
| 21 | HumanEval/21 | rescale_to_unit | 35 | comprehension w/ min–max |
| 31 | HumanEval/31 | is_prime | 41 | loop + cond |
| 40 | HumanEval/40 | triples_sum_to_zero | 59 | triple nested loop |
| 48 | HumanEval/48 | is_palindrome | 32 | loop + cond |
| 50 | HumanEval/50 | decode_shift | 33 | near-one-liner comprehension |
| 55 | HumanEval/55 | fib | 39 | recursion, branch-first |
| 56 | HumanEval/56 | correct_bracketing ⟨⟩ | 51 | counter loop + cond |
| 58 | HumanEval/58 | common | 41 | loop + cond |
| 61 | HumanEval/61 | correct_bracketing () | 51 | counter loop + cond |

⚠ 56 and 61 are the same program on different brackets — at most ONE may
appear in the final figure.

## Recommendation (Anton decides after screening; recorded for the record)

Final 3 hitting the request's variety criterion (loop / conditional /
near-one-liner): **/31 is_prime** (the loop panel), **/55 fib** (the
recursion-and-branch panel, structurally unlike everything else),
**/50 decode_shift** (the near-one-liner panel). Backups in order:
/0, /3, /48, /40.

## Caveats + next

- The receipts are canvas-1024 passes; C.7 runs at canvas 128. Solutions
  this short fit comfortably, but the final `passed` flags come from the
  actual runs — 13 candidates leaves deep slack if a couple flip.
- Per the request: trajectories are emitted for ALL 13 (a re-pick means
  re-rendering, never re-running).
- Next: `commit_step` instrumentation → staircase smoke on /31 → the
  5×4×13 grid (~33k forwards, local) → τ measured-then-picked
  (τ ∈ {0.8, 0.9, 0.95} on the 128 canvas, rule = nearest 32 forwards)
  → screening PNGs → Anton picks → hand-back doc with the JSONL path.
