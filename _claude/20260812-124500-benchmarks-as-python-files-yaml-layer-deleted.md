# Benchmarks are Python files now: dqeval/tasks/*.py, yaml layer deleted

> 2026-08-12, Anton's ruling: "we need this file structure to be easily
> skimmable by people — even if we have to redundantly have a bunch of
> (usefully named!) files." lm-eval stays the execution engine; its yaml
> format does not (get_task_dict accepts config dicts directly).

## The layout

One benchmark = one file exporting a complete config (`TASK`) or group
(`GROUP`): `humaneval.py`, `humaneval_plus.py`, `mbpp.py`, `mbpp_plus.py`,
`mbpp_fence.py`, `gsm8k.py`, `minerva_math.py`, `mmlu.py`,
`hellaswag_sub.py`, `race_sub.py`; `_grading.py` (code-exec pass@1 with the
workers/timeout knobs + the plus-comparator soundness patch);
`__init__.py` = the registry (`task_config(name)` returns a DEEP COPY —
lm-eval's loader mutates what it's given).

Frozen mechanically, not transcribed: fewshot literals (mbpp 3-shot, gsm8k
8 CoT) and the 57-subject MMLU table were extracted from the pinned
install; minerva_math freezes CONFIG but imports the Minerva
extraction/math_verify LOGIC from the pin (single source for the 200 lines
where a transcription slip silently changes numbers).

## The gate (permanent): tests/task_freeze_gate.py

Vendored == pinned stock, checked as doc fingerprints + byte-identical
rendered prompts/targets/fewshot-contexts. PASSED 2026-08-12 across all 70
leaf tasks (57 mmlu + 7 minerva + humaneval/mbpp/gsm8k_cot); local variants
construct clean. Re-run whenever tasks/ changes or the lm_eval pin bumps —
at a bump, the frozen copies PROTECT the protocol.

## What died

- All task yamls + tasks/utils.py re-export shim + TaskManager
  include_path injection.
- setup_lmeval.sh PATCH 3 (site-packages grading patch): workers/timeout
  are ordinary code in _grading.py now (LM_EVAL_CODE_WORKERS /
  LM_EVAL_CODE_TIMEOUT, defaults 32 / 10s, recorded per-cell in
  provenance).

## Traps hit (recorded for the next reader)

1. lm-eval's dict loader POPS "task" from the config and never writes the
   name back onto unregistered tasks → they run as [Task: None] and result
   aliasing crashes with a NoneType concat. grid._restore_names() is the
   repair (lm-eval's python-class branch does the same fixup itself,
   self-described in their source as "very scuffed"). Registered names
   (humaneval, mbpp, gsm8k_cot, minerva/mmlu leaves) never hit it because
   the loader merges our complete dict over the stock base — which the
   freeze gate proves is a no-op merge.
2. mbpp fewshot docs carry `text`, test docs carry `prompt` — mbpp_plus's
   doc_to_text must keep the `prompt if prompt is defined else text`
   conditional (bitten once via yaml already, kept in the .py).

Task names are UNCHANGED (humaneval_plus_sound, mbpp_plus_full, ...), so
the grid_v1 store and summarize.py needed no migration. Smoked end-to-end
via run.py: all 8 BENCH benchmarks ran stripe cells on the vendored
configs (gsm8k/math/mmlu/humaneval-plus/mbpp-plus/mbpp-fence today;
humaneval/mbpp are byte-identical registered-merge no-ops).

## UPDATE: the single-entry reshape (same day)

Anton's target: "call a single file and say run this model, this sampler
config, this dataset, this shard" + descriptive names everywhere. Final:

| was | is |
|---|---|
| grid.py + harness.py | **cell_runner.py** — THE entry: `run_cell(Cell(...))`; imports the full stack (root run.py is the login-node-safe CLI) |
| adapter.py | **models.py** (registry + contract + loading + pins) |
| sampler.py | **samplers.py** (decode + MC-NELBO; now owns GenOutput) |
| families/ | **model_families/** |
| tasks/ | **benchmarks/** ("benchmark" = our file, "task" = lm-eval's name string — vocabulary collision gone) |

Orthogonality rule enforced and AST-checked: models.py and samplers.py do
not import each other (GenOutput moved to samplers — it is the sampler's
return type; generate() duck-types the adapter). model_families/* import
both; cell_runner sits on top; benchmarks/ touch only _grading.

Gates: freeze gate PASSED post-rename; smoke matrix covered all four
execution paths (AR generate, AR gsm8k, DLM block-decode via RecordingLM,
DLM MC-NELBO via DQEvalLM.loglikelihood). hellaswag_sub/race_sub deleted
earlier the same day (retention-probe orphans; striping obsoleted the
subsample trick).

## UPDATE 2: comment/dead-code pass + imports-at-top rule (same day)

Anton's rules, all applied: no banner-style comments (dividers stripped,
labels kept as plain comments); ALL imports at module top so each file's
dependencies read at a glance (one documented exception: run.py defers its
cell_runner import so --list/manifest inspection works without the ML
stack); dead code deleted (PRESETS/get_preset/with_/full_ids/describe/chat
-- zero callers, verified by grep; the parity test hardcodes its preset
values); cell_runner's duplicate __main__ CLI deleted (run.py is THE CLI);
`from __future__ import annotations` dropped everywhere (env is pinned
py3.12, which needs none of it).

REPAIRED while in there: models.pins() read third_party/pins.json and its
errors said "run third_party/fetch.sh" -- NEITHER FILE EVER EXISTED.
Replaced with a hardcoded UPSTREAM_PINS dict (values from LOCKFILE.md) and
error messages carrying the actual clone+checkout command, so
generate_upstream/parity fixtures are now actually reachable.

Trap for the record: deleting chat() by index-slicing glued
_FAMILY_MODULES onto the ModelAdapter class body as a class attribute --
py_compile AND import smoke both passed (valid class attr!); only the
functional smoke cell caught the NameError in load(). Cell-level smokes
after every structural edit are not optional.

## UPDATE 3: dependency inversion + the closed models/ package (same day)

Two-step fix for "the dependency organization is really confusing":

1. INVERSION — models.py's hidden string-dispatch into the families
   (importlib + _FAMILY_MODULES, a disguised cycle broken only by
   laziness) was replaced by explicit dispatch: load() + BUILDERS live
   with the families, every family module imported by name.
2. CLOSURE — model_families/ became dqeval/models/, absorbing adapter.py
   (contract + HF loading + pins) and samplers.py (decode + MC-NELBO +
   GenOutput). The package is CLOSED: family files import only siblings;
   nothing imports upward; adapter ⊥ samplers stays AST-enforced.
   models/__init__.py = the MODELS catalog + BUILDERS + load() — ONE
   place to register a model. The `as hf` aliases died for direct named
   imports.

Final tree: dqeval/ = cell_runner.py + models/ + benchmarks/. Outsiders
touch exactly two surfaces: `from dqeval.models import load, MODELS` and
`from dqeval.models.samplers import DecodeConfig, generate`.
Verified: AST closure check, import smoke, three-path cell smoke
(DLM generate / DLM MC-NELBO / AR).

## UPDATE 4: dqeval/ dissolved — the repo IS the tool (same day)

Anton: not enough content at dqeval/'s top level to justify the wrapper.
Final flattening: models/ and benchmarks/ move to the repo root;
cell_runner.py merges into run.py (CLI up front, machinery below;
`from run import Cell, run_cell` for scripts). dqeval survives as the
PROJECT name only.

Riders that made it safe: (1) pyproject packaging stanza DELETED — the
generic top-level names models/benchmarks must never reach site-packages;
(2) structure_gate rule 5 asserts `import models`/`import benchmarks`
resolve to THIS repo, so a future dependency claiming either name fails
the gate instead of shadowing silently (both names verified unclaimed in
the pinned env); (3) login-node lightness consciously retired — run.py
--list now imports torch (~10s); a manifest is five string fields per
line, writable without Python.

Repo root: run.py, summarize.py, models/, benchmarks/, tests/, envs/,
third_party/, _runs/, _claude/. Both gates PASS; three-path cell smoke
clean.

## UPDATE 5: benchmarks/ collapsed into benchmark_specs.py (same day)

Anton's read was right: the package was ~85% config literals, and data
does not need file isolation the way code does. One file, one section per
benchmark (HUMANEVAL, MBPP_PLUS, GSM8K, MMLU, ...; constants prefixed --
MBPP_FEWSHOT_SAMPLES etc.), grading section at the top, TASKS registry +
task_config() at the bottom; every per-file docstring became its section
comment, nothing lost. This consciously reverses the morning's
one-file-per-benchmark ruling FOR DATA; models/ stays a package because
families are code. Name "benchmark_specs.py" chosen for the ModelSpec
symmetry (specs for models, specs for benchmarks). Bonus: the generic
top-level name `benchmarks` is gone, so the shadowing guard now only
needs to police `models`. Both gates PASS (freeze gate = configs still
byte-identical to pinned stock); three-path smoke clean.

Final repo shape: run.py + summarize.py + benchmark_specs.py + models/
+ tests/ (+ envs, third_party, _runs, _claude).

## UPDATE 6: env story simplified — pyproject deleted, envs/ flattened

pyproject.toml's sole purpose was `pip install -e .` for `import dqeval`
— a package that no longer exists — and it declared itself NOT a
dependency list. Deleted. envs/ collapsed to root requirements.txt (now
COMPLETE: adds lm_eval, evaluate, math_verify pins the old setup script
installed on the side; stale docs/ + dqeval/families/ paths fixed) +
setup_env.sh (pip install -r + the TWO irreducible source patches to
pinned lm-eval: the tf5 hf_vlms rename fix, and the gsm8k dataset-id fix
that keeps the STOCK side loadable for the freeze gate). The dead
hellaswag/winogrande dataset-id patch went with the deleted benchmarks;
the piqa provenance note went with it. Verified: setup_env.sh runs
idempotent on the live env, lm_eval imports clean.

Env story now: hand-built CUDA conda env + `pip install -r
requirements.txt` + `bash setup_env.sh`. One env, two files, both at root.

## UPDATE 7: tests/ flattened, third_party/ dissolved into adapter.py

tests/parity/ (a wrapper dir holding two files) flattened into tests/;
empty __init__.py files deleted (script-style tests, no pytest); parity
tests got the sys.path bootstrap they lost when dqeval/ dissolved. RUN
LIVE post-refactor: GATE 2 PASS — the unified sampler is still
token-identical to LLaDA's vendored native sampler after the entire
simplification day. tests/ = two gates + two parity fixtures +
load_matrix preflight, all current.

third_party/ deleted from git: LOCKFILE.md's pin table was already
duplicated by UPSTREAM_PINS, and its which-implementation-produced-which-
published-table analysis now lives IN the UPSTREAM_PINS comment block
(roles per repo, LLaDA reproducible-via-lm-eval, Dream's vendored
modified lm-eval never diffed, SDAR published via LMDeploy) — provenance
next to the code that consumes it. The path stays gitignored as the
designated clone target for parity checkouts (error messages print the
exact clone command).

## UPDATE 8: the pruning pass (Anton's rulings, all four executed)

Symbol-level audit first (dead code deleted in 13837b2: samplers'
pre-grid loglikelihood wrappers, sigma_scale, eos_id). Then the
rarely-used menu, ruled and executed:

1. SDAR PRUNED entirely (no SDAR evals planned; Table-4 row policy
   already excluded it as a boundary family): models/sdar.py, both
   registry entries, the BUILDERS line, and flash_attn + einops out of
   requirements.txt (they existed only for SDAR). UPSTREAM_PINS keeps
   the SDAR/JetEngine rows as provenance for the manuscript's
   adjacent-decode prose.
2. Registry rows: three 404'd repos (v1/v2 LR-sweep tests -- load()
   traps; traceability lives in _claude) and four instruct entries
   deleted. 24 -> 15 models, 4 builders.
3. dream.generate_native + ORDER_TO_ALG deleted (Anton's YAGNI call,
   endorsed): Dream's sampler ships INSIDE their pinned HF repo
   (model.diffusion_generate still works -- shims stay), and the
   undersell caveat is handled by citation, not by us re-running their
   decoder. Note added to dream.py's docstring.
4. Upstream clone-and-run hatch deleted (path_for/head_of/verify_pin/
   on_path + llada.generate_upstream): its flagship use case (bracketing
   SDAR's un-parity-testable greedy) died with SDAR, and vendored-vs-
   upstream equality was verified bitwise in July (receipts in the
   port-evidence doc). UPSTREAM_PINS survives as a pure data ledger.

KEPT, with the reasoning now settled: llada.generate_native is the
engine's ORACLE, not a convenience -- the parity gate (now single-gate)
diffs the unified engine against it token-for-token, and it fired twice
today proving refactors moved nothing. The distinction that resolved
Anton's confusion: we implement Dream's unmasking IDEAS (orders in
_score) but LLaDA's ALGORITHM; parity is only a testable claim for the
latter. Gate re-run post-prune: PASS.
