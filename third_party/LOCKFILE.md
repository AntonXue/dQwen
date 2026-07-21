# Upstream reference pins

These repos are **reference material and test fixtures, never runtime dependencies**.
We re-implement their samplers; we do not import their code at eval time.

They are pinned because their numbers move. JetEngine carries `opt27`/`opt28`-numbered
research comments and its `dynamic_threshold` default has already drifted 0.9 -> 0.75,
so an unpinned citation is meaningless.

| repo | commit | role |
|---|---|---|
| `ML-GSAI/LLaDA` | `96441d4` | `generate.py` (native sampler), `EVAL.md` (published decode + task configs), `get_log_likelihood.py` (MC-NELBO reference) |
| `DreamLM/Dream` | `31f94a6` | `eval/eval.py` lm-eval wrapper (logit shift at :354) |
| `DreamLM/Dream-Coder` | `79d4387` | Dream-Coder variant |
| `JetAstra/SDAR` | `6a12cdb` | `generate.py` (`block_diffusion_generate`), OpenCompass eval configs |
| `Labman42/JetEngine` | `bf8cb31` | SDAR inference engine: ~15 remasking strategies, has a greedy path |

## Which implementation produced which published table

Not all of these are equal, and getting this wrong silently changes results.

- **LLaDA** — lm-eval, via their own `@register_model("llada_dist")`. Reproducible.
- **Dream** — lm-eval, but they VENDOR a modified copy (`eval_instruct/lm_eval`, version
  0.4.8). Same version we use; the modifications still need diffing.
- **SDAR** — **neither** of the two samplers they advertise. Their published table comes
  from **LMDeploy** (`evaluation/opencompass/configs/eval_sdar_lmdeploy.py`), with
  `block_length=4, denoising_steps=4, low_confidence_dynamic, confidence_threshold=0.9,
  do_sample=False`.

  SDAR ships four disagreeing implementations with three different thresholds:

  | implementation | strategies | greedy? | threshold |
  |---|---|---|---|
  | `SDAR/generate.py` | 4 | **no** | 0.85 |
  | JetEngine | ~15 | yes | 0.75 |
  | LMDeploy (published) | — | yes | **0.9** |

  Their own `generate.py` **cannot reproduce their published numbers**: the published
  config is greedy, and that script calls `torch.multinomial` unconditionally with
  `logits / temperature`, so `temperature=0` divides by zero. Any SDAR number we
  produce is a **re-measurement**, not a reproduction, unless LMDeploy is stood up.
