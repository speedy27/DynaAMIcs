#!/bin/bash
#SBATCH --job-name=mb_eval_suite
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=mb_eval_suite_%j.out
#SBATCH --error=mb_eval_suite_%j.err

# FULL evaluation suite for the Microbiome-JEPA, OPTIMIZED onto a single GB200.
# The model is tiny (~0.7M params) so we pack the whole ablation matrix on ONE GPU
# (1 of the team's 3 slots), capping concurrency at MAXJ to keep the CPU dataloaders
# happy. Per-feature normalization is ON for every run (the new default).
#
# Runs:
#   (A) bio-loss ablation x 3 seeds: baseline / div+phylo / tvar / full   (12 runs)
#   (B) collapse demo (no VICReg: std=cov=0)                              (1 run)
# Each -> its own checkpoint dir + metrics.json -> aggregate.py for the figure.
#
#   sbatch examples/microbiome/eval_suite.slurm
#   EPOCHS=50 SEEDS="1 1000 10000" MAXJ=6 sbatch examples/microbiome/eval_suite.slurm
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
SEEDS="${SEEDS:-1 1000 10000}"
MAXJ="${MAXJ:-6}"                 # max concurrent runs on the shared GPU
BASE="$EBJEPA_WORK/checkpoints/microbiome"

# condition name -> loss overrides (normalization stays ON via the default config)
NAMES=( "baseline" "div+phylo" "tvar" "full" )
OVERR=(
  "loss.div_coeff=0 loss.phylo_coeff=0 loss.tvar_coeff=0"
  "loss.div_coeff=1 loss.phylo_coeff=1 loss.tvar_coeff=0"
  "loss.div_coeff=0 loss.phylo_coeff=0 loss.tvar_coeff=1"
  "loss.div_coeff=1 loss.phylo_coeff=1 loss.tvar_coeff=1"
)

running=0
launch() {  # tag seed extra
  local ck="$BASE/$1/seed$2"
  mkdir -p "$ck"
  EBJEPA_CKPTS="$ck" uv run --no-sync --project "$REPO" python -m examples.microbiome.main \
    --fname "$REPO/examples/microbiome/cfgs/train.yaml" \
    meta.seed="$2" optim.epochs="$EPOCHS" logging.log_wandb=false $3 \
    > "$ck/train.log" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$MAXJ" ]; then wait -n; running=$((running - 1)); fi
}

echo ">>> (A) bio-loss ablation x seeds={$SEEDS} | (B) collapse demo | epochs=$EPOCHS | maxj=$MAXJ"
for i in "${!NAMES[@]}"; do
  for s in $SEEDS; do
    launch "${NAMES[$i]}" "$s" "${OVERR[$i]}"
  done
done
# (B) collapse demo: kill the VICReg variance+covariance -> watch effrank/std crash
launch "collapse_noreg" "1" "loss.std_coeff=0 loss.cov_coeff=0"
set +e
wait
echo "=== eval suite finished | $(date) ==="

echo ""
echo "=== ABLATION SUMMARY (norm ON; bio-loss conditions + collapse demo) ==="
printf "%-16s %-6s %-8s %-7s %-8s %-8s %-7s\n" condition seed age_r2 skill effrank tvar t1d

show() {  # $1=metrics.json  $2=tag  $3=seed
  [ -f "$1" ] || { printf "%-16s %-6s MISSING\n" "$2" "$3"; return; }
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["metrics"]; g = lambda k: d.get(k, float("nan"))
print("%-16s %-6s %-8.3f %-7.3f %-8.1f %-8.4f %-7.3f" % (
    sys.argv[2], sys.argv[3], g("age_r2"), g("skill_vs_identity"),
    g("effrank"), g("tvar"), g("t1d_auroc")))
PY
}

for tag in baseline div+phylo tvar full; do
  for s in $SEEDS; do show "$BASE/$tag/seed$s/metrics.json" "$tag" "$s"; done
done
show "$BASE/collapse_noreg/seed1/metrics.json" "collapse_noreg" "1"
echo ""
echo ">>> next: aggregate -> figure :"
echo "    python -m examples.microbiome.aggregate --root $BASE"
echo "=== Done (eval suite) | $(date) ==="
