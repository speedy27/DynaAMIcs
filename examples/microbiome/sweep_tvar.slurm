#!/bin/bash
#SBATCH --job-name=mb_tvar_sweep
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=mb_tvar_sweep_%j.out
#SBATCH --error=mb_tvar_sweep_%j.err

# OPTIMIZED "best of both" sweep.
# With per-feature normalization ON (which fixes the temporal collapse), scan
# tvar_margin to find the operating point that KEEPS the temporal dynamics
# (tvar>0, skill>1) while RECOVERING the static age-clock (the no-norm full hit
# age 0.526; the norm + tvar_margin=4 run was 0.47).
# The microbiome JEPA is tiny (~0.7M params) -> ALL runs share ONE GB200 in
# parallel: 1 of the team's 3 GPU slots, ~1x single-run wall-time.
#
#   sbatch examples/microbiome/sweep_tvar.slurm
#   MARGINS="1 2 3 4" SEEDS="1" EPOCHS=50 sbatch examples/microbiome/sweep_tvar.slurm
set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"
export PYTHONUNBUFFERED=1
module load python312
uv --version &>/dev/null || export PATH="$UV_INSTALL_DIR:$PATH"
echo "=== Host: $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) | $(date) ==="
uv sync --project "$REPO"

EPOCHS="${EPOCHS:-50}"
SEEDS="${SEEDS:-1}"
MARGINS="${MARGINS:-1 2 3 4}"     # norm ON; sweep the temporal-variance hinge margin
BASE="$EBJEPA_WORK/checkpoints/microbiome"

run() {  # $1=tvar_margin  $2=seed
  local ck="$BASE/norm_tvar$1/seed$2"
  mkdir -p "$ck"
  EBJEPA_CKPTS="$ck" uv run --no-sync --project "$REPO" python -m examples.microbiome.main \
    --fname "$REPO/examples/microbiome/cfgs/train.yaml" \
    meta.seed="$2" optim.epochs="$EPOCHS" logging.log_wandb=false \
    model.normalize_features=true loss.tvar_margin="$1" \
    > "$ck/train.log" 2>&1 &
}

echo ">>> norm ON | tvar_margin sweep={$MARGINS} | seeds={$SEEDS} | epochs=$EPOCHS (parallel on 1 GPU)"
for m in $MARGINS; do
  for s in $SEEDS; do
    run "$m" "$s"
  done
done
set +e   # one failing run must not abort the summary
wait
echo "=== sweep finished | $(date) ==="

echo ""
echo "=== SWEEP SUMMARY (norm ON, tvar_margin -> age vs dynamics trade-off) ==="
printf "%-12s %-6s %-8s %-7s %-8s %-8s %-7s\n" tvar_margin seed age_r2 skill effrank tvar t1d
for m in $MARGINS; do
  for s in $SEEDS; do
    M="$BASE/norm_tvar$m/seed$s/metrics.json"
    if [ -f "$M" ]; then
      python3 - "$M" "$m" "$s" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["metrics"]
g = lambda k: d.get(k, float("nan"))
print("%-12s %-6s %-8.3f %-7.3f %-8.1f %-8.4f %-7.3f" % (
    sys.argv[2], sys.argv[3], g("age_r2"), g("skill_vs_identity"),
    g("effrank"), g("tvar"), g("t1d_auroc")))
PY
    else
      printf "%-12s %-6s MISSING\n" "$m" "$s"
    fi
  done
done
echo "=== Done (tvar sweep) | $(date) ==="
