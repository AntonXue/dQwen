"""MBPP (500 problems, [BEGIN]/[DONE] scaffold, 3-shot). Frozen from pinned
lm-eval 0.4.8's `mbpp` task; fidelity-gated byte-identical 2026-08-12.

FEWSHOT_SAMPLES is the protocol invariant shared by every MBPP variant
(mbpp_fence, mbpp_plus import it): the 3 worked examples must be identical
across variants or a prompt-format probe becomes a two-variable experiment.
"""

from dqeval.benchmarks._grading import pass_at_1

FEWSHOT_SAMPLES = [{'task_id': 2,
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

TASK = {
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
    "fewshot_config": {"sampler": "first_n", "samples": FEWSHOT_SAMPLES},
    "num_fewshot": 3,
    "metric_list": [{"metric": pass_at_1, "aggregation": "mean",
                     "higher_is_better": True}],
    "output_type": "generate_until",
    "generation_kwargs": {"until": ["[DONE]"], "do_sample": False},
    "repeats": 1,
    "metadata": {"version": 1.0},
}
