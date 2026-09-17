"""Development catalog: private dev repos and local checkpoint runs.

Loaded into MODELS only when DQEVAL_DEV=1 is set, so the public `run.py --list`
shows only models anyone can load. Nothing here is referenced by the paper's
released numbers; the v1 scouts and the v3 releases differ only in mixture.
"""

from .adapter import ModelSpec

MODELS: dict[str, ModelSpec] = {
    # ours (private org)
    # step checkpoints are HF revisions: load(..., revision="step30000-swa")
    "dqwen3.5-0.8b-base-v1": ModelSpec("EER6b/dQwen3.5-0.8B-Base", "dqwen", 248061, 248044),
    "dqwen3.5-2b-base-v1":   ModelSpec("EER6b/dQwen3.5-2B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-4b-base-v1":   ModelSpec("EER6b/dQwen3.5-4B-Base",   "dqwen", 248061, 248044),
    "dqwen3.5-9b-base-v1":   ModelSpec("EER6b/dQwen3.5-9B-Base",   "dqwen", 248061, 248044),
    "dqwen3-0.6b-base":   ModelSpec("EER6b/dQwen3-0.6B-Base",   "dqwen", 151660, 151643),
    "dqwen3-1.7b-base-v1":   ModelSpec("EER6b/dQwen3-1.7B-Base",   "dqwen", 151660, 151643),
    "qwen3sym-cdlm5k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_20260826_122649",
        "qwen3sym", 151660, 151643,
        notes="CDLM shared stage: causal attn + frozen-aug masking, 5k steps"),
    "qwen3sym-cdlm-bdlm5k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_free5k_20260826_123637",
        "qwen3sym", 151660, 151643,
        notes="CDLM->BDLM arm ('free'): bidir from cdlm ck-5000, 5k steps"),
    "qwen3sym-mix01": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_mix01_20260827_121855",
        "qwen3sym", 151660, 151643,
        notes="lambda-sweep arm: mix lambda=0.1 from cdlm ck-5000, 5k steps"),
    "qwen3sym-mix05": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_mix05_20260827_121951",
        "qwen3sym", 151660, 151643,
        notes="lambda-sweep arm: mix lambda=0.5 from cdlm ck-5000, 5k steps"),
    "qwen3sym-mix09": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_mix09_20260827_122037",
        "qwen3sym", 151660, 151643,
        notes="lambda-sweep arm: mix lambda=0.9 from cdlm ck-5000, 5k steps"),
    "qwen3sym06b-cdlm5k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_20260828_021121",
        "qwen3sym", 151660, 151643, notes="0.6B CDLM stage: causal + frozen-aug, 5k"),
    "qwen3sym06b-free10k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_free10k_20260828_021121",
        "qwen3sym", 151660, 151643, notes="0.6B lambda=0 arm: bidir from 0.6B cdlm ck-5000, 10k"),
    "qwen3sym06b-mix0110k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_mix0110k_20260828_021121",
        "qwen3sym", 151660, 151643, notes="0.6B mix lambda=0.1 arm, 10k"),
    "qwen3sym06b-mix0510k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_mix0510k_20260828_021121",
        "qwen3sym", 151660, 151643, notes="0.6B mix lambda=0.5 arm, 10k"),
    "qwen3sym06b-mix0910k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_mix0910k_20260828_021121",
        "qwen3sym", 151660, 151643, notes="0.6B mix lambda=0.9 arm, 10k"),
    "qwen3sym06b-sym10k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_sym10k_20260828_021121",
        "qwen3sym", 151660, 151643, notes="0.6B lambda=1 sym arm, 10k"),
    "qwen3sym06b-llbidir10k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_llbidir10k_20260828_094429",
        "qwen3sym", 151660, 151643, notes="0.6B lastlayer arm: causal trunk + free-bidir last layer readout, 10k"),
    "qwen3sym06b-llsym10k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_llsym10k_20260828_094429",
        "qwen3sym", 151660, 151643, notes="0.6B lastlayer arm: causal trunk + sym last layer readout, 10k"),
    "qwen3sym06b-llmix0510k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym06b_cdlm5k_llmix0510k_20260828_094429",
        "qwen3sym", 151660, 151643, notes="0.6B lastlayer arm: causal trunk + mix-0.5 last layer readout, 10k"),
    "qwen3sym-cdlm-sdlm5k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_cdlm5k_sym5k_20260826_123637",
        "qwen3sym", 151660, 151643,
        notes="CDLM->SDLM arm ('sym'): symmetrized from cdlm ck-5000, 5k steps"),
    # DLM1B project (sibling repo ~/foo/DLM1B): AR->DLM adaptation of Qwen3-0.6B via
    # on-the-fly-tokenized OWT continued pretraining. `repo` is the local training-run
    # dir (NOT an HF hub id); `revision` selects the milestone checkpoint subdirectory
    # (380/760/1530/3050/5340/7600, or "main" for the highest step present). Same ids
    # as dqwen3-0.6b-base (same Qwen3-0.6B backbone, same MASK/PAD tokens).
    "dlm1b-owt-baseline-1b": ModelSpec(
        "/home/ayx98/foo/DLM1B/_runs/20260814-044535-owt-baseline-1b", "dlm1b",
        151660, 151643,
        notes="cross-repo: weights+modeling code live in ~/foo/DLM1B, tokenizer "
              "sourced from Qwen/Qwen3-0.6B (checkpoint dir ships no tokenizer)"),
    # DLM1B qwen3_sym conversion arms (Vista scratch; 2026-08-26 campaign).
    # Matched total compute: bdlm ck-10000 vs the two staged arms at ck-5000.
    # Each checkpoint runs through ITS OWN attn_mode (adapter-enforced; toy
    # Exp 4: an SDLM ckpt through free attention collapses).
    "qwen3sym-bdlm10k": ModelSpec(
        "/scratch/11079/antonxue/dlm1b_runs/qwen3sym_bdlm10k_20260826_122624",
        "qwen3sym", 151660, 151643,
        notes="BDLM direct arm: bidir from stock Qwen3-1.7B, 10k steps"),
}
