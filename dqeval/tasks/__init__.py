"""The benchmark registry: every benchmark is one skimmable Python file here.

lm-eval is the execution engine (few-shot assembly, filters, striping hooks,
metrics plumbing); these files are the protocol. Each module exports a
complete task config (`TASK`) or group config (`GROUP`) that grid.py hands
straight to lm_eval's get_task_dict — no yaml, no include-path injection, no
patched site-packages. Configs were frozen from the pinned lm-eval 0.4.8 and
fidelity-gated (doc fingerprints + rendered prompt/target bytes) 2026-08-12.

`task_config(name)` returns a DEEP COPY: lm-eval's loader mutates the config
it is given (it pops the task name), so the registry must never hand out the
originals.
"""

from copy import deepcopy

from dqeval.tasks import (
    gsm8k,
    hellaswag_sub,
    humaneval,
    humaneval_plus,
    mbpp,
    mbpp_fence,
    mbpp_plus,
    mbpp_plus_fence,
    minerva_math,
    mmlu,
    race_sub,
)

_REGISTRY = {
    "humaneval": humaneval.TASK,
    "humaneval_plus_sound": humaneval_plus.TASK,
    "mbpp": mbpp.TASK,
    "mbpp_plus_full": mbpp_plus.TASK,
    "mbpp_plus_ticks": mbpp_plus_fence.TASK,
    "mbpp_ticks": mbpp_fence.TASK,
    "gsm8k_cot": gsm8k.TASK,
    "minerva_math": minerva_math.GROUP,
    "mmlu": mmlu.GROUP,
    "hellaswag_sub": hellaswag_sub.TASK,
    "race_sub": race_sub.TASK,
}


def task_config(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; have {sorted(_REGISTRY)}")
    return deepcopy(_REGISTRY[name])
