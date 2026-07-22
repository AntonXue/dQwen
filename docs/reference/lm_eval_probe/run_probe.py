#!/usr/bin/env python
"""Drive lm-eval simple_evaluate on the DLLM wrapper for a tiny slice."""
import sys, transformers, argparse
# transformers 5.x: lm_eval hf_vlms.py patched to AutoModelForImageTextToText;
# nothing else needed at runtime.
import dllm_lmeval  # registers "dllm"
from lm_eval import simple_evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)         # gsm8k | mmlu_abstract_algebra ...
ap.add_argument("--limit", type=int, default=20)
ap.add_argument("--model", default="EER6b/dQwen3.5-0.8B-Base")
ap.add_argument("--num_fewshot", type=int, default=None)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--max_length", type=int, default=1024)
a = ap.parse_args()

res = simple_evaluate(
    model="dllm",
    model_args=f"pretrained={a.model},batch_size={a.batch_size},max_length={a.max_length}",
    tasks=[a.task],
    limit=a.limit,
    num_fewshot=a.num_fewshot,
    bootstrap_iters=0,
)
print("\n===== RESULTS =====")
for task, metrics in res["results"].items():
    print(task, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
