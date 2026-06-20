#!/bin/bash
#SBATCH --job-name=glv_jepa_bench
#SBATCH --partition=defq
#SBATCH --account=vivatech-dynamics
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

WORK=/lustre/work/vivatech-dynamics/aouldhoci
REPO=$WORK/DynaAMIcs
RESULTS=$WORK/glv_benchmark_results
FIGS=$WORK/glv_benchmark_figs
CKPT=$WORK/glv_jepa.pt

echo "========================================"
echo " gLV JEPA Benchmark"
echo " Job: $SLURM_JOB_ID"
echo " Node: $(hostname)"
echo " Started: $(date)"
echo "========================================"

cd $REPO
git pull origin amine || true    # best-effort pull; don't fail if offline

# Activate environment
source $WORK/.venv/bin/activate 2>/dev/null || \
  (cd $REPO && uv sync --quiet && source .venv/bin/activate)

mkdir -p $RESULTS $FIGS

echo ""
echo "--- Training JEPA + running benchmark ---"
python -m examples.microbiome.glv_benchmark \
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
    --ckpt_out $CKPT \
    --n_subjects 30 \
    --T 60 \
    --seeds 0 1 2 \
    --horizons 1 3 5 10 \
    --out $RESULTS/glv_benchmark_jepa.json \
    --figs $FIGS \
    --device cuda

echo ""
echo "--- Done ---"
echo "Results: $RESULTS/glv_benchmark_jepa.json"
echo "Figures: $FIGS/"
echo "Checkpoint: $CKPT"
echo "Finished: $(date)"
