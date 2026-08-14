#!/bin/bash
# One-node diagnostic job for the CoDA GH200 silent failure
# (_claude/20260814-011801-DELEGATION): load_matrix + the golden probe.
# Output: _slurm_out/coda-probe_<jobid>.out
#   sbatch sbatch_coda_probe.sh          # from a login node
#SBATCH -J coda-probe
#SBATCH -p gh
#SBATCH -A ASC25023
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --output=_slurm_out/%x_%j.out
set -o pipefail
source ~/.bashrc
conda activate qwen35
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export HF_HOME=${HF_HOME:-/scratch/11079/antonxue/cache/huggingface}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=$PWD

echo "== load_matrix coda ($(date +%H:%M:%S))"
python -u tests/load_matrix.py coda < /dev/null

echo "== golden probe ($(date +%H:%M:%S))"
python -u - < /dev/null <<'EOF'
import torch
from models import load
from models.samplers import DecodeConfig, generate
a = load("coda-1.7b-base")
print("shims:", a.shims)
sums = [float(b.sum()) for n, b in a.model.named_buffers() if n.endswith("inv_freq")]
print("inv_freq:", len(sums), f"{sums[0]:.6f}")
h = a.model.lm_head.weight
print(f"lm_head sum={float(h.float().sum()):.4f} std={float(h.float().std()):.6f}")
for prompt in ("def add(a, b):\n    ", "The capital of France is"):
    ids = a.encode(prompt)
    raw = a.raw_logits(ids)
    print(prompt[:12], f"mean={float(raw.float().mean()):.4f}",
          f"std={float(raw.float().std()):.4f}",
          "argmax[-5:]:", raw[0].argmax(-1)[-5:].tolist())
    out = generate(a, ids, DecodeConfig(gen_length=32, block_length=32,
                                        steps_per_block=32))
    print("  gen:", out.text[:70])
EOF
echo "== done ($(date +%H:%M:%S))"
