"""Benchmark specs: the entire evaluation protocol, one section per benchmark.

Each spec is a complete lm-eval task config -- prompt template, frozen
few-shot examples, stops, metrics, grading -- frozen from the pinned
lm-eval and held byte-identical to stock by tests/task_freeze_gate.py.
lm-eval is the execution engine; this file is the protocol.

    from benchmark_specs import task_config
    td = get_task_dict([task_config("gsm8k_cot")], TaskManager())

`task_config` returns a DEEP COPY: lm-eval's loader mutates the config it
is given (it pops the task name).
"""

import os
from copy import deepcopy

from lm_eval.tasks.minerva_math.utils import (
    doc_to_text as minerva_doc_to_text,
    list_fewshot_samples as minerva_fewshot_samples,
    process_docs as minerva_process_docs,
    process_results as minerva_process_results,
)


# ---- code-execution grading (HF code_eval underneath)
# Knobs, recorded per-cell in the meta record's provenance:
#   LM_EVAL_CODE_WORKERS  parallel grading processes     (default: cores, <=32)
#   LM_EVAL_CODE_TIMEOUT  seconds per problem's WHOLE test script (default 10
#                         -- the plus suites run ~1000 cases per problem, and
#                         a tight budget makes verdicts machine-load-sensitive)
# Grader lineage + the EvalPlus agreement study:
# _claude/20260812-113000-lmeval-only-plus-columns-evalplus-removed.md

import os
import evaluate as hf_evaluate


_CODE_WORKERS = int(os.environ.get("LM_EVAL_CODE_WORKERS") or min(32, os.cpu_count() or 8))
_CODE_TIMEOUT = float(os.environ.get("LM_EVAL_CODE_TIMEOUT") or 10.0)

_code_eval = None


def _metric():
    """HF code_eval, loaded lazily (its import-time canary requires
    HF_ALLOW_CODE_EVAL=1, which run.py sets for unsafe benchmarks)."""
    global _code_eval
    if _code_eval is None:
        _code_eval = hf_evaluate.load("code_eval")
    return _code_eval


def pass_at_k(references: list, predictions: list, k: list = None):
    """HumanEval-style: predictions are full programs (prompt + completion,
    assembled by `build_predictions`). Behavior-identical to stock lm-eval."""
    assert k is not None
    if isinstance(k, int):
        k = [k]
    return _metric().compute(references=references, predictions=predictions,
                             k=k, num_workers=_CODE_WORKERS,
                             timeout=_CODE_TIMEOUT)[0]


def pass_at_1(references, predictions):
    """MBPP-style: the completion IS the program. Behavior-identical to stock."""
    return _metric().compute(references=references, predictions=[predictions],
                             k=[1], num_workers=_CODE_WORKERS,
                             timeout=_CODE_TIMEOUT)[0]["pass@1"]


def build_predictions(resps: list, docs: list) -> list:
    """HumanEval filter: completions grade as prompt + completion."""
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


# plus-suite comparator soundness
# The rendered test scripts in evalplus/humanevalplus + evalplus/mbppplus carry
# an inline comparator whose `is_floats([])` is vacuously True (all() on
# empty), routing empty-expected checks into np.allclose -- and numpy
# broadcasts shape (1,) against (0,) to an empty result, so
# np.allclose([wrong], []) == True: any WRONG non-empty output passes whenever
# the expected value is an empty list/tuple. Caught 2026-08-12 (HumanEval/62 +
# /96 grader A/B vs the evalplus package, which compares these correctly).

_UNSOUND = "        return all(isinstance(i, float) for i in x)"
_SOUND = "        return len(x) > 0 and all(isinstance(i, float) for i in x)"


def _sound_test(doc):
    # a few problems render a bespoke comparator instead (e.g. HumanEval/32
    # checks a residual tolerance); those don't carry this bug -- skip them
    return {**doc, "test": doc["test"].replace(_UNSOUND, _SOUND, 1)}


def plus_process_docs(dataset):
    n = sum(_UNSOUND in t for t in dataset["test"])
    if not n or n < 0.8 * len(dataset):
        raise ValueError(
            f"comparator soundness patch matched {n}/{len(dataset)} docs -- "
            "the rendered format drifted; re-derive the patch."
        )
    return dataset.map(_sound_test)


# ---- humaneval
# HumanEval (164 problems; completion-style: the prompt holds the def, the
# model writes the body). Frozen from pinned lm-eval 0.4.8's `humaneval` task;
# fidelity-gated byte-identical 2026-08-12. Graded by the section above.
HUMANEVAL = {
    "task": "humaneval",
    "dataset_path": "openai/openai_humaneval",
    "test_split": "test",
    "doc_to_text": "{{prompt}}",
    "doc_to_target": "{{test}}\ncheck({{entry_point}})",
    "unsafe_code": True,
    "target_delimiter": " ",
    "num_fewshot": 0,
    "metric_list": [{"metric": pass_at_k, "aggregation": "mean",
                     "higher_is_better": True, "k": [1]}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                          "max_gen_toks": 1024, "do_sample": False},
    "repeats": 1,
    "filter_list": [{"name": "create_test",
                     "filter": [{"function": "custom",
                                 "filter_fn": build_predictions}]}],
    "metadata": {"version": 1.0},
}


# ---- humaneval-plus
# HumanEval+ — same 164 problems and prompts as humaneval, EvalPlus's ~78x
# denser test suite (dataset field `test` carries the full suite). Task name
# stays `humaneval_plus_sound` for store continuity: "sound" = the rendered
# comparator's empty-expected bug is patched at load time (plus_process_docs
# above).
HUMANEVAL_PLUS = {
    "task": "humaneval_plus_sound",
    "dataset_path": "evalplus/humanevalplus",
    "test_split": "test",
    "process_docs": plus_process_docs,
    "doc_to_text": "{{prompt}}",
    "doc_to_target": "{{test}}\ncheck({{entry_point}})",
    "unsafe_code": True,
    "target_delimiter": " ",
    "num_fewshot": 0,
    "metric_list": [{"metric": pass_at_k, "aggregation": "mean",
                     "higher_is_better": True, "k": [1]}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                          "max_gen_toks": 1024, "do_sample": False},
    "repeats": 1,
    "filter_list": [{"name": "create_test",
                     "filter": [{"function": "custom",
                                 "filter_fn": build_predictions}]}],
    "metadata": {"version": 1.0},
}


# ---- mbpp
# MBPP (500 problems, [BEGIN]/[DONE] scaffold, 3-shot). Frozen from pinned
# lm-eval 0.4.8's `mbpp` task; fidelity-gated byte-identical 2026-08-12.
#
# MBPP_FEWSHOT_SAMPLES is the protocol invariant shared by every MBPP
# variant below: the 3 worked examples must be identical across variants or
# a prompt-format probe becomes a two-variable experiment.
MBPP_FEWSHOT_SAMPLES = [{'task_id': 2,
  'text': 'Write a function to find the similar elements from the given two tuple '
          'lists.',
  'code': 'def similar_elements(test_tup1, test_tup2):\r\n'
          '  res = tuple(set(test_tup1) & set(test_tup2))\r\n'
          '  return (res) ',
  'test_list': ['assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)',
                'assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)',
                'assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, '
                '14)'],
  'is_fewshot': True},
 {'task_id': 3,
  'text': 'Write a python function to identify non-prime numbers.',
  'code': 'import math\r\n'
          'def is_not_prime(n):\r\n'
          '    result = False\r\n'
          '    for i in range(2,int(math.sqrt(n)) + 1):\r\n'
          '        if n % i == 0:\r\n'
          '            result = True\r\n'
          '    return result',
  'test_list': ['assert is_not_prime(2) == False',
                'assert is_not_prime(10) == True',
                'assert is_not_prime(35) == True'],
  'is_fewshot': True},
 {'task_id': 4,
  'text': 'Write a function to find the largest integers from a given list of numbers '
          'using heap queue algorithm.',
  'code': 'import heapq as hq\r\n'
          'def heap_queue_largest(nums,n):\r\n'
          '  largest_nums = hq.nlargest(n, nums)\r\n'
          '  return largest_nums',
  'test_list': ['assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, '
                '58],3)==[85, 75, 65] ',
                'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, '
                '58],2)==[85, 75] ',
                'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, '
                '58],5)==[85, 75, 65, 58, 35]'],
  'is_fewshot': True}]

MBPP = {
    "task": "mbpp",
    "dataset_path": "google-research-datasets/mbpp",
    "dataset_name": "full",
    "test_split": "test",
    "doc_to_text": "You are an expert Python programmer, and here is your task: "
                   "{{text}} Your code should pass these tests:\n\n"
                   "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n[BEGIN]\n",
    "doc_to_target": "{% if is_fewshot is defined %}{{code}}\n[DONE]"
                     "{% else %}{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}{% endif %}",
    "unsafe_code": True,
    "target_delimiter": "",
    "fewshot_config": {"sampler": "first_n", "samples": MBPP_FEWSHOT_SAMPLES},
    "num_fewshot": 3,
    "metric_list": [{"metric": pass_at_1, "aggregation": "mean",
                     "higher_is_better": True}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["[DONE]"], "do_sample": False},
    "repeats": 1,
    "metadata": {"version": 1.0},
}


# ---- mbpp-plus
# MBPP+ — EvalPlus's MBPP-sanitized set (378 problems, edited prompts, ~34x
# denser tests) under OUR MBPP protocol: the same [BEGIN]/[DONE] scaffold
# and the same MBPP_FEWSHOT_SAMPLES.
#
# Task name stays `mbpp_plus_full` for store continuity, and as a reminder of
# WHY this spec exists: lm-eval's stock `mbpp_plus` grades against
# test_list[0..2] — the three ORIGINAL asserts — and never executes the plus
# suite in the dataset's `test` field. This config targets the real suite.
#
# COMPARABILITY: published MBPP+ numbers (EvalPlus's own pipeline) use a
# 0-shot docstring prompt, not this scaffold — this column is a re-measurement
# under the house protocol, not a reproduction (Dream-tripwire-style
# disclosure in the manuscript ledger).
MBPP_PLUS = {
    "task": "mbpp_plus_full",
    "dataset_path": "evalplus/mbppplus",
    "test_split": "test",
    "process_docs": plus_process_docs,
    "doc_to_text": "You are an expert Python programmer, and here is your task: "
                   "{{prompt if prompt is defined else text}} Your code should "
                   "pass these tests:\n\n"
                   "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n[BEGIN]\n",
    "doc_to_target": "{% if is_fewshot is defined %}{{code}}\n[DONE]"
                     "{% else %}{{test}}{% endif %}",
    "unsafe_code": True,
    "target_delimiter": "",
    "fewshot_config": {"sampler": "first_n", "samples": MBPP_FEWSHOT_SAMPLES},
    "num_fewshot": 3,
    "metric_list": [{"metric": pass_at_1, "aggregation": "mean",
                     "higher_is_better": True}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["[DONE]"], "do_sample": False},
    "repeats": 1,
    "metadata": {"version": 1.0},
}


# ---- mbpp-fence
# MBPP with a markdown code fence instead of the [BEGIN]/[DONE] scaffold
# (task name `mbpp_ticks`). A LABELLED format-sensitivity variant, never the
# Table-4 MBPP column: the fence's sign tracks code-instruction exposure
# (dQwen +4/+6pp, Dream-Coder +0.20, LLaDA -2.40) because the task->fenced-
# solution pattern is corpus-dependent.
MBPP_FENCE = {
    "task": "mbpp_ticks",
    "dataset_path": "google-research-datasets/mbpp",
    "dataset_name": "full",
    "test_split": "test",
    "doc_to_text": "You are an expert Python programmer, and here is your task: "
                   "{{text}} Your code should pass these tests:\n\n"
                   "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n```python\n",
    "doc_to_target": "{% if is_fewshot is defined %}{{code}}\n```"
                     "{% else %}{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}{% endif %}",
    "unsafe_code": True,
    "target_delimiter": "",
    "fewshot_config": {"sampler": "first_n", "samples": MBPP_FEWSHOT_SAMPLES},
    "num_fewshot": 3,
    "metric_list": [{"metric": pass_at_1, "aggregation": "mean",
                     "higher_is_better": True}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["```"], "do_sample": False},
    "repeats": 1,
    "metadata": {"version": 1.0},
}


# ---- mbpp-plus-fence
# MBPP+ with the markdown-fence scaffold (task name `mbpp_plus_ticks`) —
# the fourth quadrant of the prompt-format probe:
#
#                  [BEGIN]/[DONE]      ```python fence
#     MBPP-500     MBPP                MBPP_FENCE
#     MBPP+-378    MBPP_PLUS           MBPP_PLUS_FENCE
#
# Only the scaffold differs from MBPP_PLUS, so the fence effect above can be
# re-read under ~34x denser tests. LABELLED variant, never the Table-4
# MBPP+ column.
MBPP_PLUS_FENCE = {
    "task": "mbpp_plus_ticks",
    "dataset_path": "evalplus/mbppplus",
    "test_split": "test",
    "process_docs": plus_process_docs,
    "doc_to_text": "You are an expert Python programmer, and here is your task: "
                   "{{prompt if prompt is defined else text}} Your code should "
                   "pass these tests:\n\n"
                   "{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n```python\n",
    "doc_to_target": "{% if is_fewshot is defined %}{{code}}\n```"
                     "{% else %}{{test}}{% endif %}",
    "unsafe_code": True,
    "target_delimiter": "",
    "fewshot_config": {"sampler": "first_n", "samples": MBPP_FEWSHOT_SAMPLES},
    "num_fewshot": 3,
    "metric_list": [{"metric": pass_at_1, "aggregation": "mean",
                     "higher_is_better": True}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["```"], "do_sample": False},
    "repeats": 1,
    "metadata": {"version": 1.0},
}


# ---- gsm8k
# GSM8K, 8-shot chain-of-thought (`gsm8k_cot`). Frozen from pinned lm-eval
# 0.4.8; fidelity-gated byte-identical 2026-08-12. Two extraction filters:
# strict-match ("The answer is N.") is the headline metric; flexible-extract
# (last number) is recorded alongside.
GSM8K_FEWSHOT_SAMPLES = [{'question': 'There are 15 trees in the grove. Grove workers will plant trees in the '
              'grove today. After they are done, there will be 21 trees. How many '
              'trees did the grove workers plant today?',
  'target': 'There are 15 trees originally. Then there were 21 trees after some more '
            'were planted. So there must have been 21 - 15 = 6. The answer is 6.'},
 {'question': 'If there are 3 cars in the parking lot and 2 more cars arrive, how many '
              'cars are in the parking lot?',
  'target': 'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is '
            '5.'},
 {'question': 'Leah had 32 chocolates and her sister had 42. If they ate 35, how many '
              'pieces do they have left in total?',
  'target': 'Originally, Leah had 32 chocolates. Her sister had 42. So in total they '
            'had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is '
            '39.'},
 {'question': 'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 '
              'lollipops. How many lollipops did Jason give to Denny?',
  'target': 'Jason started with 20 lollipops. Then he had 12 after giving some to '
            'Denny. So he gave Denny 20 - 12 = 8. The answer is 8.'},
 {'question': 'Shawn has five toys. For Christmas, he got two toys each from his mom '
              'and dad. How many toys does he have now?',
  'target': 'Shawn started with 5 toys. If he got 2 toys each from his mom and dad, '
            'then that is 4 more toys. 5 + 4 = 9. The answer is 9.'},
 {'question': 'There were nine computers in the server room. Five more computers were '
              'installed each day, from monday to thursday. How many computers are now '
              'in the server room?',
  'target': 'There were originally 9 computers. For each of 4 days, 5 more computers '
            'were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer '
            'is 29.'},
 {'question': 'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On '
              'wednesday, he lost 2 more. How many golf balls did he have at the end '
              'of wednesday?',
  'target': 'Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 '
            '- 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer '
            'is 33.'},
 {'question': 'Olivia has $23. She bought five bagels for $3 each. How much money does '
              'she have left?',
  'target': 'Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 '
            'dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.'}]

GSM8K = {
    "task": "gsm8k_cot",
    "tag": ["chain_of_thought"],
    "dataset_path": "openai/gsm8k",
    "dataset_name": "main",
    "test_split": "test",
    "doc_to_text": "Q: {{question}}\nA:",
    "doc_to_target": "{{answer.split('####')[-1].strip() if answer is defined else target}}",
    "target_delimiter": " ",
    "fewshot_config": {"sampler": "first_n", "samples": GSM8K_FEWSHOT_SAMPLES},
    "num_fewshot": 8,
    "metric_list": [{"aggregation": "mean", "higher_is_better": True,
                     "ignore_case": True, "ignore_punctuation": False,
                     "metric": "exact_match",
                     "regexes_to_ignore": [",", "\\$", "(?s).*#### ", "\\.$"]}],
    "output_type": "generate_until",
    "generation_kwargs": {"do_sample": False, "until": ["Q:", "</s>", "<|im_end|>"]},
    "repeats": 1,
    "filter_list": [
        {"name": "strict-match",
         "filter": [{"function": "regex",
                     "regex_pattern": "The answer is (\\-?[0-9\\.\\,]+)."},
                    {"function": "take_first"}]},
        {"name": "flexible-extract",
         "filter": [{"function": "regex", "group_select": -1,
                     "regex_pattern": "(-?[$0-9.,]{2,})|(-?[0-9]+)"},
                    {"function": "take_first"}]},
    ],
    "metadata": {"version": 3.0},
}


# ---- math (minerva protocol)
# MATH (Hendrycks), Minerva 4-shot protocol (`minerva_math`, 7 subject
# subtasks + size-weighted group). Config frozen from pinned lm-eval 0.4.8;
# fidelity-gated 2026-08-12.
#
# CONFIG is frozen here; the EXTRACTION/GRADING LOGIC is imported from the
# pinned install (Minerva answer normalization is ~200 lines of regex, and
# math_verify's symbolic equivalence — the ruled MATH column metric — lives in
# process_results). Copying that logic would create a second source of truth
# for exactly the code where a silent transcription slip changes numbers;
# importing keeps it single-sourced while the pin holds.
#
# Metrics per doc: exact_match (strict Minerva) AND math_verify — the
# manuscript column is math_verify (Anton's 2026-08-12 ruling; strict
# collapses to a format test on big post-trained AR models).
MINERVA_SUBJECTS = [
    ("minerva_math_algebra", "algebra"),
    ("minerva_math_counting_and_prob", "counting_and_probability"),
    ("minerva_math_geometry", "geometry"),
    ("minerva_math_intermediate_algebra", "intermediate_algebra"),
    ("minerva_math_num_theory", "number_theory"),
    ("minerva_math_prealgebra", "prealgebra"),
    ("minerva_math_precalc", "precalculus"),
]


def _minerva_leaf(task_name, dataset_name):
    return {
        "task": task_name,
        "tag": ["math_word_problems"],
        "dataset_path": "EleutherAI/hendrycks_math",
        "dataset_name": dataset_name,
        "dataset_kwargs": {"trust_remote_code": True},
        "training_split": "train",
        "test_split": "test",
        "process_docs": minerva_process_docs,
        "doc_to_text": minerva_doc_to_text,
        "doc_to_target": "{{answer if few_shot is undefined else solution}}",
        "process_results": minerva_process_results,
        "target_delimiter": " ",
        "fewshot_config": {"sampler": "first_n", "samples": minerva_fewshot_samples},
        "num_fewshot": 4,
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True},
            {"metric": "math_verify", "aggregation": "mean", "higher_is_better": True},
        ],
        "output_type": "generate_until",
        "generation_kwargs": {"until": ["Problem:"], "do_sample": False,
                              "temperature": 0.0},
        "repeats": 1,
        "metadata": {"version": 2.0},
    }


MINERVA_MATH = {
    "group": "minerva_math",
    "task": [_minerva_leaf(*row) for row in MINERVA_SUBJECTS],
    "aggregate_metric_list": [
        {"metric": "exact_match", "weight_by_size": True},
        {"metric": "math_verify", "weight_by_size": True},
    ],
    "metadata": {"version": 1.0},
}


# ---- mmlu
# MMLU, 5-shot multiple choice (57 subjects). Frozen from pinned lm-eval
# 0.4.8; fidelity-gated 2026-08-12. Flat group: stock nests subjects under 4
# category groups, but the headline `mmlu` number is the size-weighted micro
# average over all docs either way (category subaggregates were never used by
# the manuscript). num_fewshot comes from the cell (BENCH pins 5).
# SUBJECTS rows: (task_name, dataset_name, description) -- extracted verbatim
# from the pinned per-subject yamls.
MMLU_SUBJECTS = [
    ('mmlu_abstract_algebra', 'abstract_algebra',
     'The following are multiple choice questions (with answers) about abstract algebra.\n\n'),
    ('mmlu_anatomy', 'anatomy',
     'The following are multiple choice questions (with answers) about anatomy.\n\n'),
    ('mmlu_astronomy', 'astronomy',
     'The following are multiple choice questions (with answers) about astronomy.\n\n'),
    ('mmlu_business_ethics', 'business_ethics',
     'The following are multiple choice questions (with answers) about business ethics.\n\n'),
    ('mmlu_clinical_knowledge', 'clinical_knowledge',
     'The following are multiple choice questions (with answers) about clinical knowledge.\n\n'),
    ('mmlu_college_biology', 'college_biology',
     'The following are multiple choice questions (with answers) about college biology.\n\n'),
    ('mmlu_college_chemistry', 'college_chemistry',
     'The following are multiple choice questions (with answers) about college chemistry.\n\n'),
    ('mmlu_college_computer_science', 'college_computer_science',
     'The following are multiple choice questions (with answers) about college computer science.\n\n'),
    ('mmlu_college_mathematics', 'college_mathematics',
     'The following are multiple choice questions (with answers) about college mathematics.\n\n'),
    ('mmlu_college_medicine', 'college_medicine',
     'The following are multiple choice questions (with answers) about college medicine.\n\n'),
    ('mmlu_college_physics', 'college_physics',
     'The following are multiple choice questions (with answers) about college physics.\n\n'),
    ('mmlu_computer_security', 'computer_security',
     'The following are multiple choice questions (with answers) about computer security.\n\n'),
    ('mmlu_conceptual_physics', 'conceptual_physics',
     'The following are multiple choice questions (with answers) about conceptual physics.\n\n'),
    ('mmlu_econometrics', 'econometrics',
     'The following are multiple choice questions (with answers) about econometrics.\n\n'),
    ('mmlu_electrical_engineering', 'electrical_engineering',
     'The following are multiple choice questions (with answers) about electrical engineering.\n\n'),
    ('mmlu_elementary_mathematics', 'elementary_mathematics',
     'The following are multiple choice questions (with answers) about elementary mathematics.\n\n'),
    ('mmlu_formal_logic', 'formal_logic',
     'The following are multiple choice questions (with answers) about formal logic.\n\n'),
    ('mmlu_global_facts', 'global_facts',
     'The following are multiple choice questions (with answers) about global facts.\n\n'),
    ('mmlu_high_school_biology', 'high_school_biology',
     'The following are multiple choice questions (with answers) about high school biology.\n\n'),
    ('mmlu_high_school_chemistry', 'high_school_chemistry',
     'The following are multiple choice questions (with answers) about high school chemistry.\n\n'),
    ('mmlu_high_school_computer_science', 'high_school_computer_science',
     'The following are multiple choice questions (with answers) about high school computer science.\n\n'),
    ('mmlu_high_school_european_history', 'high_school_european_history',
     'The following are multiple choice questions (with answers) about high school european history.\n\n'),
    ('mmlu_high_school_geography', 'high_school_geography',
     'The following are multiple choice questions (with answers) about high school geography.\n\n'),
    ('mmlu_high_school_government_and_politics', 'high_school_government_and_politics',
     'The following are multiple choice questions (with answers) about high school government and politics.\n\n'),
    ('mmlu_high_school_macroeconomics', 'high_school_macroeconomics',
     'The following are multiple choice questions (with answers) about high school macroeconomics.\n\n'),
    ('mmlu_high_school_mathematics', 'high_school_mathematics',
     'The following are multiple choice questions (with answers) about high school mathematics.\n\n'),
    ('mmlu_high_school_microeconomics', 'high_school_microeconomics',
     'The following are multiple choice questions (with answers) about high school microeconomics.\n\n'),
    ('mmlu_high_school_physics', 'high_school_physics',
     'The following are multiple choice questions (with answers) about high school physics.\n\n'),
    ('mmlu_high_school_psychology', 'high_school_psychology',
     'The following are multiple choice questions (with answers) about high school psychology.\n\n'),
    ('mmlu_high_school_statistics', 'high_school_statistics',
     'The following are multiple choice questions (with answers) about high school statistics.\n\n'),
    ('mmlu_high_school_us_history', 'high_school_us_history',
     'The following are multiple choice questions (with answers) about high school us history.\n\n'),
    ('mmlu_high_school_world_history', 'high_school_world_history',
     'The following are multiple choice questions (with answers) about high school world history.\n\n'),
    ('mmlu_human_aging', 'human_aging',
     'The following are multiple choice questions (with answers) about human aging.\n\n'),
    ('mmlu_human_sexuality', 'human_sexuality',
     'The following are multiple choice questions (with answers) about human sexuality.\n\n'),
    ('mmlu_international_law', 'international_law',
     'The following are multiple choice questions (with answers) about international law.\n\n'),
    ('mmlu_jurisprudence', 'jurisprudence',
     'The following are multiple choice questions (with answers) about jurisprudence.\n\n'),
    ('mmlu_logical_fallacies', 'logical_fallacies',
     'The following are multiple choice questions (with answers) about logical fallacies.\n\n'),
    ('mmlu_machine_learning', 'machine_learning',
     'The following are multiple choice questions (with answers) about machine learning.\n\n'),
    ('mmlu_management', 'management',
     'The following are multiple choice questions (with answers) about management.\n\n'),
    ('mmlu_marketing', 'marketing',
     'The following are multiple choice questions (with answers) about marketing.\n\n'),
    ('mmlu_medical_genetics', 'medical_genetics',
     'The following are multiple choice questions (with answers) about medical genetics.\n\n'),
    ('mmlu_miscellaneous', 'miscellaneous',
     'The following are multiple choice questions (with answers) about miscellaneous.\n\n'),
    ('mmlu_moral_disputes', 'moral_disputes',
     'The following are multiple choice questions (with answers) about moral disputes.\n\n'),
    ('mmlu_moral_scenarios', 'moral_scenarios',
     'The following are multiple choice questions (with answers) about moral scenarios.\n\n'),
    ('mmlu_nutrition', 'nutrition',
     'The following are multiple choice questions (with answers) about nutrition.\n\n'),
    ('mmlu_philosophy', 'philosophy',
     'The following are multiple choice questions (with answers) about philosophy.\n\n'),
    ('mmlu_prehistory', 'prehistory',
     'The following are multiple choice questions (with answers) about prehistory.\n\n'),
    ('mmlu_professional_accounting', 'professional_accounting',
     'The following are multiple choice questions (with answers) about professional accounting.\n\n'),
    ('mmlu_professional_law', 'professional_law',
     'The following are multiple choice questions (with answers) about professional law.\n\n'),
    ('mmlu_professional_medicine', 'professional_medicine',
     'The following are multiple choice questions (with answers) about professional medicine.\n\n'),
    ('mmlu_professional_psychology', 'professional_psychology',
     'The following are multiple choice questions (with answers) about professional psychology.\n\n'),
    ('mmlu_public_relations', 'public_relations',
     'The following are multiple choice questions (with answers) about public relations.\n\n'),
    ('mmlu_security_studies', 'security_studies',
     'The following are multiple choice questions (with answers) about security studies.\n\n'),
    ('mmlu_sociology', 'sociology',
     'The following are multiple choice questions (with answers) about sociology.\n\n'),
    ('mmlu_us_foreign_policy', 'us_foreign_policy',
     'The following are multiple choice questions (with answers) about us foreign policy.\n\n'),
    ('mmlu_virology', 'virology',
     'The following are multiple choice questions (with answers) about virology.\n\n'),
    ('mmlu_world_religions', 'world_religions',
     'The following are multiple choice questions (with answers) about world religions.\n\n'),
]


def _mmlu_leaf(task_name, dataset_name, description):
    return {
        "task": task_name,
        "dataset_path": "hails/mmlu_no_train",
        "dataset_name": dataset_name,
        "dataset_kwargs": {"trust_remote_code": True},
        "test_split": "test",
        "fewshot_split": "dev",
        "fewshot_config": {"sampler": "first_n"},
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}"
                       "\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_target": "answer",
        "doc_to_choice": ["A", "B", "C", "D"],
        "description": description,
        "target_delimiter": " ",
        "metric_list": [{"metric": "acc", "aggregation": "mean",
                         "higher_is_better": True}],
        "output_type": "multiple_choice",
        "repeats": 1,
        "metadata": {"version": 1.0},
    }


MMLU = {
    "group": "mmlu",
    "task": [_mmlu_leaf(*row) for row in MMLU_SUBJECTS],
    "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
    "metadata": {"version": 2},
}


# ---- the registry
TASKS = {
    "humaneval": HUMANEVAL,
    "humaneval_plus_sound": HUMANEVAL_PLUS,
    "mbpp": MBPP,
    "mbpp_plus_full": MBPP_PLUS,
    "mbpp_ticks": MBPP_FENCE,
    "mbpp_plus_ticks": MBPP_PLUS_FENCE,
    "gsm8k_cot": GSM8K,
    "minerva_math": MINERVA_MATH,
    "mmlu": MMLU,
}


def task_config(name: str):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; have {sorted(TASKS)}")
    return deepcopy(TASKS[name])
