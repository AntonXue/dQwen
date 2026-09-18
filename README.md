# dQwen

Evaluation suite for the dQwen3.5 technical report, plus the modeling code the
released checkpoints run on. Every number in the paper, ours and the comparators',
comes from one entrypoint, one environment, and one decoding protocol.

**Paper:** [dQwen3.5: Hybrid-Attention Diffusion Language Models](https://arxiv.org/abs/2609.20751)  
**Models:** https://huggingface.co/UT-IFML/dQwen3.5-9B-Base (family table and quickstart)

| model | trunk | total | Hugging Face |
|---|---|---|---|
| dQwen3.5-9B-Base | 6.92B | 8.95B | [UT-IFML/dQwen3.5-9B-Base](https://huggingface.co/UT-IFML/dQwen3.5-9B-Base) |
| dQwen3.5-4B-Base | 3.57B | 4.21B | [UT-IFML/dQwen3.5-4B-Base](https://huggingface.co/UT-IFML/dQwen3.5-4B-Base) |
| dQwen3.5-2B-Base | 1.37B | 1.88B | [UT-IFML/dQwen3.5-2B-Base](https://huggingface.co/UT-IFML/dQwen3.5-2B-Base) |
| dQwen3.5-0.8B-Base | 0.50B | 0.75B | [UT-IFML/dQwen3.5-0.8B-Base](https://huggingface.co/UT-IFML/dQwen3.5-0.8B-Base) |
| dQwen3-1.7B-Base (full-attention control) | 1.41B | 1.72B | [UT-IFML/dQwen3-1.7B-Base](https://huggingface.co/UT-IFML/dQwen3-1.7B-Base) |

## Install

Pinned environment: `torch 2.7.1+cu128`, `transformers 5.13.0`, `flash-linear-attention 0.5.1`.
The Gated DeltaNet layers run Triton kernels, so a CUDA GPU is required.

```bash
pip install -r requirements.txt
bash setup_env.sh        # lm_eval 0.4.8 and the graders; patches lm_eval for transformers 5
```

## Generate

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/dQwen3.5-9B-Base", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
print(model.generate("def fibonacci(n):", gen_length=512, stop_strings=["\ndef "]).text)
```

`generate` is the block-diffusion sampler from `models/samplers.py`, shipped inside each
checkpoint's modeling file. `models/modeling_dqwen3_5.py` and `models/modeling_dqwen3.py`
are the canonical copies; `tests/test_dqwen_release_sampler.py` holds them token-identical
to the harness engine.

## Evaluate

One cell is (model, revision, decode, benchmark, shard):

```bash
python run.py dqwen3.5-9b-base main standard-static-s1024 humaneval   # the paper's protocol
python run.py dqwen3.5-9b-base main block32-tau0.9 gsm8k 2/8          # parallel decoding, shard 2 of 8
python run.py Qwen/Qwen3.5-9B - ar gsm8k                               # an AR counterpart
python run.py --list                                                   # models, decodes, benchmarks
```

Decodes: `standard-static-sK` (full canvas, K steps), `block32-static-sK`, `standard-tauT` and
`block32-tauT` (confidence threshold T), `ar`, and `mc-nelbo` (MMLU). Benchmarks: `humaneval`,
`humaneval-plus`, `mbpp`, `mbpp-plus`, `gsm8k`, `math500`, `mmlu`. Shards are stripes, never
chunks; merge every shard before quoting a number. `benchmark_specs.py` is the complete
protocol, `summarize.py` aggregates finished cells, and `manifests/` holds the cell lists
behind each table in the paper.

## Citation

```bibtex
@article{xue2026dqwen,
  title   = {dQwen3.5: Hybrid-Attention Diffusion Language Models},
  author  = {Xue, Anton and Rout, Litu and Akella, Aditya and Klivans, Adam and Sanghavi, Sujay and Shakkottai, Sanjay},
  journal = {arXiv preprint arXiv:2609.20751},
  year    = {2026}
}
```

## License

Apache-2.0.
