"""Forward-pass throughput sweep -- the RNN-layer cost question, no decoding.

Boss ask (via Anton, 2026-08-21): do the hybrid (Gated DeltaNet) layers cost
throughput vs full attention? Design: models x {full, trunk, head} x B x L.

  full   input_ids -> logits, the deployment number (adapter.raw_logits).
  trunk  random (B, L, d) bf16 tensors through the INNER STACK only --
         isolates attention/GDN + MLPs from both the embedding lookup and
         the vocab-head GEMM (248k vs 151k vocab would otherwise confound
         the hybrid-vs-control pair). Timing is data-independent for these
         stacks (no value-dependent control flow), so random inputs time
         identically to real hidden states.
  head   the lone (B*L, d) @ (d, V) GEMM -- makes the vocab confound a
         quotable number, and full ~= embed + trunk + head is the harness's
         own sanity check.

Deployment kernel posture ON PURPOSE: flash/mem-efficient SDPA stay enabled
(the eval grid's math-sdpa pin answers a different question -- determinism).
Publication runs happen on EXCLUSIVE cluster nodes; co-tenant GPUs corrupt
timing. Local GPU use is prototyping only.

    CUDA_VISIBLE_DEVICES=2 python throughput.py --model dqwen3.5-2b-base-v3 \
        --lengths 256,1024 --batches 1,4        # prototype slice

One JSONL line per (mode, B, L) cell, OOM cells recorded not skipped.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import models

OUT = Path(__file__).parent / "_runs" / "20260821-throughput-sweep"

REVISIONS = {                       # registry name -> revision (Tier 1 + 2)
    "dqwen3.5-0.8b-base-v3": "step50000-swa",
    "dqwen3.5-2b-base-v3": "step50000-swa",     # Tier 1: the width-matched
    "dqwen3-1.7b-base-v3": "step50000-swa",     #   hybrid / full-attn pair
    "dqwen3.5-4b-base-v3": "step50000-swa",
    "dqwen3.5-9b-base-v3": "step50000-swa",
    "llada-8b-base": None,
    "dream-7b-base": None,
    "dream-coder-7b-base": None,
    "coda-1.7b-base": None,
}


def find_trunk(model, d: int, device):
    """Locate the inner layer stack + the embeds kwarg it accepts, verified
    by output shape (B, L, d). Fails to (None, None) -- callers record trunk
    mode as unsupported rather than guessing."""
    cands, seen = [], set()
    for a in ("model", "transformer", "backbone"):
        m = getattr(model, a, None)
        for mm in (m, getattr(m, "model", None), getattr(m, "transformer", None)):
            if mm is not None and id(mm) not in seen and mm is not model:
                seen.add(id(mm))
                cands.append(mm)
    probe = torch.randn(1, 8, d, device=device, dtype=torch.bfloat16)
    for m in cands:
        for kw in ("inputs_embeds", "input_embeddings"):
            try:
                out = m(**{kw: probe})
            except Exception:
                continue
            h = getattr(out, "last_hidden_state", None)
            if h is None:
                h = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(h) and h.shape == (1, 8, d):
                return m, kw
    return None, None


def timed(fn, warmup: int, iters: int, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    ms = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize(device)
        ms.append(a.elapsed_time(b))
    ms.sort()
    q = statistics.quantiles(ms, n=4)
    return dict(median_ms=round(q[1], 3), iqr_ms=round(q[2] - q[0], 3),
                min_ms=round(ms[0], 3), iters=iters,
                peak_mem_mb=round(torch.cuda.max_memory_allocated(device) / 2**20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REVISIONS))
    ap.add_argument("--modes", default="full,trunk,head")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--lengths", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--sdpa", choices=("fast", "math"), default="fast",
                    help="attention axis: fast = flash/mem-efficient "
                         "(deployment), math = unfused reference. The GDN "
                         "axis is env-level: FLA installed = fast path, "
                         "absent = torch fallback.")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.sdpa == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    a.out.mkdir(parents=True, exist_ok=True)
    modes = a.modes.split(",")
    batches = [int(x) for x in a.batches.split(",")]
    lengths = [int(x) for x in a.lengths.split(",")]

    m = models.load(a.model, revision=REVISIONS[a.model])
    if a.sdpa == "math":
        # LLaDA's remote __init__ re-enables flash sdp (modeling_llada.py:1056)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    dev = m.device
    cfg = m.model.config
    d = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
    vocab = m.model.get_output_embeddings().weight.shape[0]
    trunk, embeds_kw = (find_trunk(m.model, d, dev)
                        if "trunk" in modes else (None, None))
    head_w = m.model.get_output_embeddings().weight
    import importlib.util
    import socket
    prov = dict(model=a.model, revision=REVISIONS[a.model], d=d, vocab=vocab,
                node=socket.gethostname().split(".")[0],
                device=torch.cuda.get_device_name(dev),
                torch=torch.__version__, sdpa_arm=a.sdpa,
                flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                mem_eff_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                fla=importlib.util.find_spec("fla") is not None,
                causal_conv1d=importlib.util.find_spec("causal_conv1d") is not None,
                trunk_module=type(trunk).__name__ if trunk else None,
                embeds_kw=embeds_kw, warmup=a.warmup)
    path = a.out / f"throughput_{a.model}.jsonl"
    print(f"[{a.model}] {prov}", flush=True)

    with open(path, "a") as f, torch.inference_mode():
        for mode in modes:
            if mode == "trunk" and trunk is None:
                f.write(json.dumps(dict(prov, mode="trunk",
                                        unsupported=True)) + "\n")
                print(f"[{a.model}] trunk: UNSUPPORTED (no accessor found)",
                      flush=True)
                continue
            for L in lengths:
                for B in batches:
                    torch.cuda.empty_cache()
                    try:
                        if mode == "full":
                            x = torch.randint(0, vocab, (B, L), device=dev)
                            fn = lambda: m.raw_logits(x)
                        elif mode == "trunk":
                            x = torch.randn(B, L, d, device=dev,
                                            dtype=torch.bfloat16)
                            fn = lambda: trunk(**{embeds_kw: x})
                        else:  # head
                            x = torch.randn(B, L, d, device=dev,
                                            dtype=torch.bfloat16)
                            fn = lambda: F.linear(x, head_w)
                        r = timed(fn, a.warmup, a.iters, dev)
                        r["tokens_per_s"] = round(B * L / (r["median_ms"] / 1e3))
                        rec = dict(prov, mode=mode, batch=B, length=L, **r)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        rec = dict(prov, mode=mode, batch=B, length=L, oom=True)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    tag = ("OOM" if rec.get("oom") else
                           f"{rec['median_ms']}ms {rec['tokens_per_s']} tok/s "
                           f"{rec['peak_mem_mb']}MB")
                    print(f"[{a.model}] {mode} B={B} L={L}: {tag}", flush=True)
                    if rec.get("oom"):
                        break        # larger B at this L will OOM too


if __name__ == "__main__":
    main()
