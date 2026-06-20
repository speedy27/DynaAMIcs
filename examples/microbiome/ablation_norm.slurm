#!/bin/bash
#SBATCH --job-name=mb_norm_abl
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=mb_norm_abl_%j.out
#SBATCH --error=mb_norm_abl_%j.err

# OPTIMIZED per-feature-normalization ablation.
# The microbiome JEPA is tiny (~0.7M params), so a single GB200 (189 GB) runs ALL
# 6 jobs in parallel on ONE GPU: {norm_on, norm_off} x seeds {1, 1000, 10000}.
#   -> uses 1 of the team's 3 GPU slots (leaves 2 free for the Tahoe runs)
#   -> ~1x single-run wall-time instead of 6x (or 2 waves of an array)
# Each run writes its own checkpoint dir + train.log; metrics.json feeds aggregate.py.
#
#   sbatch examples/microbiome/ablation_norm.slurm
#   EPOCHS=50 sbatch examples/microbiome/ablation_norm.slurm   # override epochs
set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"
export PYTHONUNBUFFERED=1
module load python312
uv --version &>/dev/null || export PATH="$UV_INSTALL_DIR:$PATH"

echo "=== Host: $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) | $(date) ==="
echo ">>> uv sync (once, up front) ..."
uv sync --project "$REPO"

EPOCHS="${EPOCHS:-30}"          # norm-ON plateaus early (~ep10) -> 30 is plenty & fast
BASE="$EBJEPA_WORK/checkpoints/microbiome"

# Launch one training in the background on the shared GPU. --no-sync avoids the
# 6 processes fighting over the uv env lock (env already synced above).
run() {  # $1=tag  $2=seed  $3=extra-overrides
  local ck="$BASE/$1/seed$2"
  mkdir -p "$ck"
  EBJEPA_CKPTS="$ck" uv run --no-sync --project "$REPO" python -m examples.microbiome.main \
    --fname "$REPO/examples/microbiome/cfgs/train.yaml" \
    meta.seed="$2" optim.epochs="$EPOCHS" logging.log_wandb=false $3 \
    > "$ck/train.log" 2>&1 &
}

echo ">>> launching 6 parallel runs on 1 GPU (epochs=$EPOCHS) ..."
for s in 1 1000 10000; do
  run norm_on  "$s" ""
  run norm_off "$s" "model.normalize_features=false"
done
set +e   # one failing run must not abort the summary below
wait
echo "=== all 6 runs finished | $(date) ==="

echo ""
echo "=== ABLATION SUMMARY (per-feature normalization on vs off) ==="
printf "%-9s %-7s %-9s %-8s %-9s %-7s %-7s\n" condition seed age_r2 skill effrank tvar t1d
for tag in norm_on norm_off; do
  for s in 1 1000 10000; do
    m="$BASE/$tag/seed$s/metrics.json"
    if [ -f "$m" ]; then
      python3 - "$m" "$tag" "$s" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
g = lambda k: m.get(k, float("nan"))
print("%-9s %-7s %-9.3f %-8.3f %-9.1f %-7.4f %-7.3f" % (
    sys.argv[2], sys.argv[3], g("age_r2"), g("skill_vs_identity"),
    g("effrank"), g("tvar"), g("t1d_auroc")))
PY
    else
      printf "%-9s %-7s MISSING\n" "$tag" "$s"
    fi
  done
done

echo ""
echo ">>> dynamics + representation tables (norm_on seed1, norm_off seed1):"
for tag in norm_on norm_off; do
  echo "----- $tag seed1 -----"
  uv run --no-sync --project "$REPO" python -m examples.microbiome.baselines \
    --ckpt "$BASE/$tag/seed1/microbiome_jepa.pt" --rand-seeds 1 2>&1 \
    | sed -n "/representation baselines/,/error growth/p"
done
echo "=== Done (norm ablation) | $(date) ==="
