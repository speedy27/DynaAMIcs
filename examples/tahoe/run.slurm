#!/bin/bash
#SBATCH --job-name=tahoe
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=tahoe_%j.out
#SBATCH --error=tahoe_%j.err
export PYTHONUNBUFFERED=1
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"
echo "=== host $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Build the cache on the compute node (lots of RAM) if missing.
if [ ! -f "$WORK/tahoe/cache.pt" ]; then
  echo ">>> precompute cache from shards in $WORK/tahoe_data"
  uv run python -u -m eb_jepa.datasets.tahoe.precompute \
    --shards "$WORK/tahoe_data" --out "$WORK/tahoe/cache.pt" --panel 2000 --modules 32
fi
echo ">>> training (${EPOCHS:-250} epochs)"
uv run python -u -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml \
  data.cache_path="$WORK/tahoe/cache.pt" \
  optim.epochs="${EPOCHS:-250}" data.batch_size="${BS:-2048}" ${EXTRA:-}
echo "=== done | $(date) ==="
