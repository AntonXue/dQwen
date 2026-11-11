---
license: apache-2.0
base_model: {parent}
pipeline_tag: text-generation
library_name: transformers
inference: false
tags: [diffusion-language-model, masked-diffusion, {arch_tag}, {family_tag}, arxiv:2609.20751]
---

# {name}

{description} This is a base model, with no instruction tuning.

**Paper:** [dQwen3.5: Hybrid-Attention Diffusion Language Models](https://arxiv.org/abs/2609.20751)  
**Code:** https://github.com/AntonXue/dQwen

## The dQwen3.5 family

| model | total params | backbone |
|---|---|---|
{family_rows}

## Quickstart

Needs a CUDA GPU and `transformers>=5.13` (tested with `torch 2.7.1+cu128`, `flash-linear-attention 0.5.1`).

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/{name}", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
stop_strings = ["\n\n", "\ndef", "\nclass", "<|endoftext|>"]
print(model.generate("def fibonacci(n):\n", gen_length=512, stop_strings=stop_strings).text)
```

`generate` decodes the whole canvas at once, committing positions above a confidence
threshold (`conf_threshold=0.9`, the paper's tau); pass `block_length=32` for
left-to-right block decoding, or `conf_threshold=None, steps_per_block=k` for a fixed
budget. A position's confidence is its top-1 probability (`conf_source="top1"`);
`conf_source="sampled"` reads the sampled token's probability instead, the default
before 2026-11-11. The two differ only when sampling (`temperature>0`): greedy output,
and every result in the paper, is unchanged. `tau=` and `order=` still work as
deprecated names for `conf_threshold=` and `remask_order=`. The 50B-token checkpoint
from the paper is `revision="step25000-swa"`.

## Citation

```bibtex
@article{{xue2026dqwen,
  title   = {{dQwen3.5: Hybrid-Attention Diffusion Language Models}},
  author  = {{Xue, Anton and Rout, Litu and Akella, Aditya and Klivans, Adam and Sanghavi, Sujay and Shakkottai, Sanjay}},
  journal = {{arXiv preprint arXiv:2609.20751}},
  year    = {{2026}}
}}
```
