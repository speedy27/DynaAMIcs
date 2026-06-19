#!/bin/bash
# run_ablation.sh -- the controlled comparison: 4 conditions x N seeds.
#
# Conditions (one microbiome-specific change at a time, incl. the collapse fix):
#   baseline        no bio terms                       (generic VICReg JEPA)
#   div+phylo       diversity + phylo, NO temporal var (the pre-fix collapse)
#   tvar            temporal-variance term ONLY        (the fix in isolation)
#   div+phylo+tvar  everything                          (full model)
#
# Each run writes <ckpt>/microbiome/<condition>/seed<seed>/metrics.json;
# aggregate.py then turns the whole tree into one table + bar chart.
#
# Cluster (default): one SLURM job per (condition, seed) via train.slurm:
#   bash examples/microbiome/run_ablation.sh
# Local quick smoke (no SLURM, tiny epochs, one seed):
#   LOCAL=1 MICROBIOME_EPOCHS=2 SEEDS="1" bash examples/microbiome/run_ablation.sh
# After everything finishes:
#   python -m examples.microbiome.aggregate --root <base>/microbiome
set -e

SEEDS="${SEEDS:-1 1000 10000}"
EPOCHS="${MICROBIOME_EPOCHS:-50}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="${EBJEPA_CKPTS_BASE:-checkpoints}"   # local-mode checkpoint root

NAMES=( "baseline" "div+phylo" "tvar" "div+phylo+tvar" )
OVERR=(
  "loss.div_coeff=0 loss.phylo_coeff=0 loss.tvar_coeff=0"
  "loss.div_coeff=1 loss.phylo_coeff=1 loss.tvar_coeff=0"
  "loss.div_coeff=0 loss.phylo_coeff=0 loss.tvar_coeff=1"
  "loss.div_coeff=1 loss.phylo_coeff=1 loss.tvar_coeff=1"
)

for i in "${!NAMES[@]}"; do
  tag="${NAMES[$i]}"
  extra="${OVERR[$i]}"
  for s in $SEEDS; do
    if [ "${LOCAL:-0}" = "1" ]; then
      echo ">>> [local] tag=$tag seed=$s epochs=$EPOCHS"
      ckpt="$BASE/microbiome/${tag}/seed${s}"
      mkdir -p "$ckpt"
      EBJEPA_CKPTS="$ckpt" uv run python -m examples.microbiome.main \
        --fname "$REPO/examples/microbiome/cfgs/train.yaml" \
        meta.seed="$s" optim.epochs="$EPOCHS" logging.log_wandb=false $extra
    else
      echo ">>> [sbatch] tag=$tag seed=$s epochs=$EPOCHS"
      MICROBIOME_TAG="$tag" MICROBIOME_EXTRA="$extra" MICROBIOME_EPOCHS="$EPOCHS" \
        sbatch "$REPO/examples/microbiome/train.slurm" "$s"
    fi
  done
done

echo "=== all runs launched. When done, aggregate with:"
echo "    python -m examples.microbiome.aggregate --root $BASE/microbiome --out $BASE/microbiome/ablation.png"
