#!/bin/bash
#SBATCH --job-name=microbiome_jepa
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
# No hardcoded --account: SLURM uses your DEFAULT account (your team allocation).
# Override with: sbatch --account=<acct> train.slurm. Drop --reservation after the
# Vivatech reservation ends (sbatch --reservation= ...).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=microbiome_jepa_%j.out
#SBATCH --error=microbiome_jepa_%j.err

# Full Microbiome-JEPA training on a GPU node, followed by the controlled
# representation baselines (raw / random-encoder / trained JEPA).
#
# Submit one seed (default 0):
#   sbatch examples/microbiome/train.slurm
# Submit a specific seed (3-seed sweep = submit three times):
#   for s in 1 1000 10000; do sbatch examples/microbiome/train.slurm $s; done
# Override epochs:
#   MICROBIOME_EPOCHS=50 sbatch examples/microbiome/train.slurm 1
# Ablation condition (checkpoint subdir + extra dotted overrides):
#   MICROBIOME_TAG=baseline \
#     MICROBIOME_EXTRA="loss.div_coeff=0 loss.phylo_coeff=0 loss.tvar_coeff=0" \
#     sbatch examples/microbiome/train.slurm 1
# Or launch the whole 4-condition x 3-seed sweep with run_ablation.sh.

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"   # so the relative cache_path in train.yaml resolves

export PYTHONUNBUFFERED=1   # stream epoch/baseline logs live into the .out file

SEED="${1:-${MICROBIOME_SEED:-0}}"
EPOCHS="${MICROBIOME_EPOCHS:-50}"
TAG="${MICROBIOME_TAG:-full}"          # ablation condition name (checkpoint subdir)
EXTRA="${MICROBIOME_EXTRA:-}"          # extra dotted overrides, e.g. "loss.tvar_coeff=0"

echo "=== Host: $(hostname) | Arch: $ARCH | tag=$TAG seed=$SEED epochs=$EPOCHS | $(date) ==="
echo "=== extra overrides: ${EXTRA:-<none>} ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a) ==="

module load python312
if ! uv --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
echo "=== uv: $(uv --version) ==="

echo ">>> uv sync..."
uv sync --project "$REPO"

# Per-condition / per-seed checkpoint dir so concurrent jobs don't clash.
export EBJEPA_CKPTS="${EBJEPA_CKPTS:-$EBJEPA_WORK/checkpoints}/microbiome/${TAG}/seed${SEED}"
mkdir -p "$EBJEPA_CKPTS"
echo "=== checkpoints -> $EBJEPA_CKPTS ==="

echo ">>> Training (tag=$TAG seed=$SEED, epochs=$EPOCHS)..."
uv run --project "$REPO" python -m examples.microbiome.main \
    --fname "$REPO/examples/microbiome/cfgs/train.yaml" \
    meta.seed="$SEED" optim.epochs="$EPOCHS" logging.log_wandb=false $EXTRA

echo ">>> Baselines (raw / random-encoder / trained JEPA)..."
uv run --project "$REPO" python -m examples.microbiome.baselines \
    --ckpt "$EBJEPA_CKPTS/microbiome_jepa.pt"

echo "=== Done (tag=$TAG seed=$SEED) | $(date) ==="
