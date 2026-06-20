#!/bin/bash
# gLV JEPA Temporal Benchmark
# Trains SetEncoder+RNNPredictor JEPA on synthetic gLV, then runs MDSINE2-style
# hold-one-subject-out CLR-RMSE benchmark vs persistence / gLV-L2 / gLV-net.
#
# Submit (from repo root on DALIA):
#   sbatch examples/microbiome/glv_benchmark.slurm
#
#SBATCH --job-name=glv_jepa_bench
#SBATCH --partition=defq
#SBATCH --account=vivatech-dynamics
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=glv_jepa_bench_%j.out
#SBATCH --error=glv_jepa_bench_%j.err

export PYTHONUNBUFFERED=1
set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

echo "========================================"
echo " gLV JEPA Benchmark"
echo " Job:   $SLURM_JOB_ID"
echo " Node:  $(hostname)"
echo " GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo " Start: $(date)"
echo "========================================"

# Pull latest code from amine branch
git pull origin amine || echo "git pull failed (offline?), continuing with current code"

# Install uv if missing
if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
echo "uv: $(uv --version)"

uv sync --project "$REPO"

RESULTS="$EBJEPA_WORK/glv_benchmark_results"
FIGS="$EBJEPA_WORK/glv_benchmark_figs"
CKPT="$EBJEPA_WORK/checkpoints/glv_jepa.pt"
mkdir -p "$RESULTS" "$FIGS" "$(dirname $CKPT)"

echo ""
echo "--- Training JEPA + benchmark ---"
uv run python -u -m examples.microbiome.glv_benchmark \
    --n_species 32 \
    --n_candidate 8 \
    --n_traj 512 \
    --T_train 50 \
    --n_window 8 \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-3 \
    --d_model 128 \
    --h_enc 256 \
    --ckpt_out "$CKPT" \
    --n_subjects 30 \
    --T 60 \
    --seeds 0 1 2 \
    --horizons 1 3 5 10 \
    --out "$RESULTS/glv_benchmark_jepa.json" \
    --figs "$FIGS" \
    --device cuda

echo ""
echo "========================================"
echo " Done: $(date)"
echo " Results: $RESULTS/glv_benchmark_jepa.json"
echo " Figures: $FIGS/"
echo " Checkpoint: $CKPT"
echo "========================================"
