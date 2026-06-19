#!/bin/bash
#SBATCH --job-name=tperturb
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=tperturb_%j.out
#SBATCH --error=tperturb_%j.err
export PYTHONUNBUFFERED=1
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"
echo "=== host $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo ">>> training perturbation world model (${EPOCHS:-250} epochs)"
uv run python -u -m examples.tahoe.perturb --fname examples/tahoe/cfgs/perturb.yaml \
  data.cache_path="$WORK/tahoe/cache_pert.pt" optim.epochs="${EPOCHS:-250}" ${EXTRA:-}
echo "=== done | $(date) ==="
