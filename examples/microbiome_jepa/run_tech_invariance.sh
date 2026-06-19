#!/usr/bin/env bash
# B — sequencing-tech invariance (CPU). Label corpus samples amplicon-vs-wgs via the RunID->Terms join,
# encode with the FROZEN corpus-pretrained encoder, and probe whether tech is recoverable from the rep
# (lower = more invariant = better) vs raw-meanpool + random-encoder baselines; biome probe as control.
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_tech_invariance.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --job-name=mb_tech
#SBATCH --output=mb_tech_%j.out
#SBATCH --error=mb_tech_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerA_real.yaml
ENC=$WORK/checkpoints/microbiome_jepa/realenc
DATA=$EBJEPA_DSETS/susagi/data
$PY -m examples.microbiome_jepa.tech_invariance --checkpoint $ENC/latest.pth.tar --fname $CFG \
  --data_dir $DATA --d_model 128 --n_max 256 --per_class_cap 2500 --device cpu \
  --susagi_repo $WORK/Microbiome-Modelling \
  --susagi_ckpt $DATA/model/checkpoint_epoch_0_final_newblack_2epoch.pt \
  --out $WORK/checkpoints/microbiome_jepa/tech_invariance
echo "MB_TECH_DONE"
