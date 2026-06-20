#!/bin/bash
#SBATCH --job-name=tahoe_ultra
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=tahoe_ultra_%j.out
#SBATCH --error=tahoe_ultra_%j.err

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

SEED="${SEED:-1}"

echo "=== Host: $(hostname) | Seed: $SEED | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

module load python312
if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
uv sync --dev --project "$REPO"

TAG="${TAG:-$(date +%Y%m%d_%H%M)_$SLURM_JOB_ID}"
RUN_DIR="${EBJEPA_CKPTS:-$REPO/checkpoints}/tahoe_ultra_${TAG}"
mkdir -p "$RUN_DIR"
echo "=== output: $RUN_DIR ==="

CACHE_PATH="${TAHOE_CACHE:-$EBJEPA_WORK/tahoe/cache.pt}"
echo "=== cache: $CACHE_PATH ==="

# Background GPU-util sampler (proof of saturation)
(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,noheader >> "$RUN_DIR/gpu_util.csv"
    sleep 5
  done
) &
UTIL_PID=$!
trap "kill $UTIL_PID 2>/dev/null || true" EXIT

EBJEPA_CKPTS="$RUN_DIR" uv run --project "$REPO" python -u -m examples.tahoe.main_fast \
    --fname examples/tahoe/cfgs/train_ultra.yaml \
    meta.seed=$SEED \
    data.cache_path=$CACHE_PATH \
    logging.log_wandb=false 2>&1 | tee "$RUN_DIR/train.log"

kill $UTIL_PID 2>/dev/null || true
echo "=== seed $SEED done ==="
