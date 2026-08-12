"""Gate: the vendored task configs (dqeval/tasks/*.py) are byte-faithful to
pinned lm-eval for every benchmark that derives from a stock task.

Run whenever dqeval/tasks/ changes or the lm_eval pin is bumped:

    PYTHONPATH=. HF_ALLOW_CODE_EVAL=1 <env>/python tests/task_freeze_gate.py

Checks, per benchmark: identical doc-set fingerprint, byte-identical
rendered prompt/target on sampled docs, byte-identical few-shot context at
the BENCH shot count. Local variants with no stock counterpart (the plus
tasks, mbpp_ticks, the *_sub samples) are checked for construction +
fingerprint stability only — their deviations from stock are the point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STOCK_PAIRS = {  # vendored registry name -> stock lm-eval task name
    "humaneval": "humaneval",
    "mbpp": "mbpp",
    "gsm8k_cot": "gsm8k_cot",
    "minerva_math": "minerva_math",
    "mmlu": "mmlu",
}
SHOTS = {"humaneval": 0, "mbpp": 3, "gsm8k_cot": 8, "minerva_math": 4,
         "mmlu": 5, "humaneval_plus_sound": 0, "mbpp_plus_full": 3,
         "mbpp_ticks": 3, "mbpp_plus_ticks": 3}
VARIANTS = ["humaneval_plus_sound", "mbpp_plus_full", "mbpp_ticks",
            "mbpp_plus_ticks"]


def leaves(td):
    for v in td.values():
        if isinstance(v, dict):
            yield from leaves(v)
        else:
            yield v


def fingerprint(task):
    import hashlib, json
    h = hashlib.sha256()
    n = 0
    for doc in task.test_docs():
        h.update(json.dumps(doc, sort_keys=True, default=str).encode())
        n += 1
    return n, h.hexdigest()


def renders(task, shots, k=5):
    docs = list(task.test_docs())
    idx = [0, len(docs) // 2, len(docs) - 1][:k]
    out = []
    for i in idx:
        out.append(task.doc_to_text(docs[i]))
        out.append(str(task.doc_to_target(docs[i])))
    out.append(task.fewshot_context(docs[0], shots))
    return out


def main():
    from lm_eval.tasks import TaskManager, get_task_dict
    from dqeval.tasks import task_config
    tm = TaskManager()
    fails = 0

    for ours, stock in STOCK_PAIRS.items():
        new = {t.config.task: t for t in leaves(get_task_dict([task_config(ours)], tm))}
        old = {t.config.task: t for t in leaves(get_task_dict([stock], tm))}
        if set(new) != set(old):
            print(f"FAIL {ours}: leaf sets differ: {set(new) ^ set(old)}")
            fails += 1
            continue
        for name in sorted(new):
            fp_n, fp_o = fingerprint(new[name]), fingerprint(old[name])
            if fp_n != fp_o:
                print(f"FAIL {ours}/{name}: fingerprint {fp_n} != {fp_o}")
                fails += 1
                continue
            shots = SHOTS[ours]
            r_n, r_o = renders(new[name], shots), renders(old[name], shots)
            if r_n != r_o:
                bad = next(i for i, (a, b) in enumerate(zip(r_n, r_o)) if a != b)
                print(f"FAIL {ours}/{name}: render #{bad} differs")
                fails += 1
            else:
                print(f"ok   {ours}/{name}: {fp_n[0]} docs, renders byte-identical")

    for ours in VARIANTS:
        task = next(leaves(get_task_dict([task_config(ours)], tm)))
        n, h = fingerprint(task)
        renders(task, SHOTS[ours])   # constructs + renders without error
        print(f"ok   {ours}: {n} docs, fingerprint {h[:16]}, renders clean")

    print("\n" + ("GATE FAILED" if fails else "GATE PASSED: vendored == pinned stock"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
