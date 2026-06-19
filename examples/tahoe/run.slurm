#!/bin/bash
# Tahoe-JEPA (two-view SSL): 250 epochs, SIGReg + PathwayCoherenceLoss, early stopping.
#
# TIME ESTIMATE (DALIA GB200, 189 GB HBM3):
#   MLP 2000->512->512->128, BCS(256 slices), batch_size=2048, ~100-200k cells in cache.
#   ~10-40 s/epoch (data-size dependent) + ~20 s per probe eval (scikit-learn LogReg).
#   250 epochs + 25 probe evals -> 45-180 min training. Request 6h with 2x safety margin.
#   With early stopping (patience=5 probes) -> likely stops at 100-150 epochs (~1-2h).
#
# Submit:
#   sbatch examples/tahoe/run.slurm
# Override epochs / batch size / seed:
#   EPOCHS=100 BS=2048 SEED=1 sbatch examples/tahoe/run.slurm
# Submit 3-seed sweep:
#   for s in 0 1 2; do SEED=$s sbatch examples/tahoe/run.slurm; done
#
#SBATCH --job-name=tahoe_jepa
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=tahoe_jepa_%j.out
#SBATCH --error=tahoe_jepa_%j.err

export PYTHONUNBUFFERED=1
set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-250}"
BS="${BS:-2048}"

echo "=== DALIA Tahoe-JEPA | host=$(hostname) | $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a) ==="
echo "=== seed=$SEED epochs=$EPOCHS batch_size=$BS ==="

module load python312 2>/dev/null || true

# Install uv if missing (arch-aware: DALIA nodes are aarch64)
if ! uv --version &>/dev/null; then
    echo ">>> installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
echo ">>> uv: $(uv --version)"
echo ">>> uv sync..."
uv sync --project "$REPO"

# Per-seed checkpoint directory
export EBJEPA_CKPTS="${EBJEPA_CKPTS:-$EBJEPA_WORK/checkpoints}/tahoe/seed${SEED}"
mkdir -p "$EBJEPA_CKPTS"
echo "=== checkpoints -> $EBJEPA_CKPTS ==="

# Build the gene-panel cache on the compute node if missing (lots of RAM needed)
CACHE="$EBJEPA_WORK/tahoe/cache.pt"
if [ ! -f "$CACHE" ]; then
    echo ">>> precomputing gene-panel cache from parquet shards in $EBJEPA_WORK/tahoe_data"
    uv run python -u -m eb_jepa.datasets.tahoe.precompute \
        --shards "$EBJEPA_WORK/tahoe_data" --out "$CACHE" --panel 2000 --modules 32
fi

echo ">>> training Tahoe-JEPA ($EPOCHS epochs, early stop patience=5 probes)"
uv run python -u -m examples.tahoe.main \
    --fname "$REPO/examples/tahoe/cfgs/train.yaml" \
    data.cache_path="$CACHE" \
    meta.seed="$SEED" \
    optim.epochs="$EPOCHS" \
    data.batch_size="$BS" \
    ${EXTRA:-}

echo "=== done | $(date) ==="
