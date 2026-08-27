"""flash-attn-2 vs torch-SDPA delta, Tier-1 pair, full mode, slice points.

Standalone ON PURPOSE: NOT part of the main dataset and not a harness path.
The campaign handoff (20260826-235257 SS6) excluded the flash_attn package as
"marginal over torch-flash" -- this measures that claim instead of asserting
it, now that flash_attn 2.8.3 is already installed. Both impls load through
the same plain from_pretrained call, so the delta is internally clean even
though absolute numbers are not comparable to the adapter-loaded dataset.
"""
import json
import socket
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

OUT = Path(__file__).parent
REPOS = {
    "dqwen3.5-2b-base-v3": ("EER6b/dQwen3.5-2B-Base-v3", "step50000-swa"),
    "dqwen3-1.7b-base-v3": ("EER6b/dQwen3-1.7B-Base-v3", "step50000-swa"),
}
PTS = [(1, 1024), (1, 4096), (16, 1024), (16, 4096)]
WARMUP, ITERS = 5, 50


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(ITERS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b))
    ms.sort()
    q = statistics.quantiles(ms, n=4)
    return dict(median_ms=round(q[1], 3), iqr_ms=round(q[2] - q[0], 3), iters=ITERS)


with open(OUT / "fa2_delta.jsonl", "a") as f, torch.inference_mode():
    for name, (repo, rev) in REPOS.items():
        for impl in ("sdpa", "flash_attention_2"):
            try:
                m = AutoModelForCausalLM.from_pretrained(
                    repo, revision=rev, dtype=torch.bfloat16,
                    attn_implementation=impl, trust_remote_code=True,
                ).cuda().eval()
            except Exception as e:
                rec = dict(model=name, attn_impl=impl, error=str(e)[:300])
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(name, impl, "LOAD FAILED:", str(e)[:120], flush=True)
                continue
            V = m.get_output_embeddings().weight.shape[0]
            for B, L in PTS:
                x = torch.randint(0, V, (B, L), device="cuda")
                try:
                    r = timed(lambda: m(input_ids=x).logits)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    r = dict(oom=True)
                rec = dict(model=name, attn_impl=impl,
                           node=socket.gethostname().split(".")[0],
                           batch=B, length=L, **r)
                if "median_ms" in rec:
                    rec["tokens_per_s"] = round(B * L / (rec["median_ms"] / 1e3))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(name, impl, B, L, rec.get("median_ms", "OOM"), flush=True)
            del m
            torch.cuda.empty_cache()
