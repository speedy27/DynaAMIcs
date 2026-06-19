#!/bin/bash
# Tahoe perturbation world model: 250 epochs, frozen MosaicFM + drug-conditioned predictor.
#
# TIME ESTIMATE (DALIA GB200, 189 GB HBM3):
#   RNNPredictor (GRU hidden=2560), batch_size=1024, paired (ctrl, pert) cells.
#   ~5-20 s/epoch (faster than main.py: no BCS, no 2 views) + ~15 s per val eval.
#   250 epochs + 25 val evals -> 30-120 min training. Request 4h with 2x safety margin.
#   With early stopping (patience=5 probes) -> likely stops at 80-150 epochs (~30-60 min).
#
# Submit:
#   sbatch examples/tahoe/perturb.slurm
# Override epochs / seed:
#   EPOCHS=200 SEED=1 sbatch examples/tahoe/perturb.slurm
# 3-seed sweep:
#   for s in 0 1 2; do SEED=$s sbatch examples/tahoe/perturb.slurm; done
#
#SBATCH --job-name=tahoe_pert
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=tahoe_pert_%j.out
#SBATCH --error=tahoe_pert_%j.err

export PYTHONUNBUFFERED=1
set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-250}"

echo "=== DALIA Tahoe-Perturb | host=$(hostname) | $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a) ==="
echo "=== seed=$SEED epochs=$EPOCHS ==="

module load python312 2>/dev/null || true

if ! uv --version &>/dev/null; then
    echo ">>> installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
echo ">>> uv: $(uv --version)"
echo ">>> uv sync..."
uv sync --project "$REPO"

export EBJEPA_CKPTS="${EBJEPA_CKPTS:-$EBJEPA_WORK/checkpoints}/tahoe_pert/seed${SEED}"
mkdir -p "$EBJEPA_CKPTS"
echo "=== checkpoints -> $EBJEPA_CKPTS ==="

# Build perturbation cache if missing (pairs control/perturbed cells + Morgan fingerprints)
CACHE_PERT="$EBJEPA_WORK/tahoe/cache_pert.pt"
if [ ! -f "$CACHE_PERT" ]; then
    echo ">>> precomputing perturbation cache from parquet shards in $EBJEPA_WORK/tahoe_data"
    uv run python -u -m eb_jepa.datasets.tahoe.precompute_pert \
        --shards "$EBJEPA_WORK/tahoe_data" --out "$CACHE_PERT"
fi

echo ">>> training perturbation world model ($EPOCHS epochs, early stop patience=5 probes)"
uv run python -u -m examples.tahoe.perturb \
    --fname "$REPO/examples/tahoe/cfgs/perturb.yaml" \
    data.cache_path="$CACHE_PERT" \
    meta.seed="$SEED" \
    optim.epochs="$EPOCHS" \
    ${EXTRA:-}

echo "=== done | $(date) ==="
