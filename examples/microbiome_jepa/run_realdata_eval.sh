#!/usr/bin/env bash
# M2 FAIRNESS eval (CPU, no pretraining) on the EXISTING corpus-pretrained encoder (realenc, 30ep).
# Adds: (1) an MLP probe matching Susagi's MLPClassifier(128) on the SAME frozen embeddings (apples-to-
# apples with the supervised baseline), reported alongside the linear probe; (2) a CORPUS z-score
# (consistent with pretraining) instead of the infant-token z-score. Encoder stays FROZEN. Fast CPU job.
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_realdata_eval.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --job-name=mb_eval
#SBATCH --output=mb_eval_%j.out
#SBATCH --error=mb_eval_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerA_real.yaml
ENC=$WORK/checkpoints/microbiome_jepa/realenc
DATA=$EBJEPA_DSETS/susagi/data
$PY -m examples.microbiome_jepa.realdata --checkpoint $ENC/latest.pth.tar --fname $CFG \
  --data_dir $DATA --d_model 128 --n_max 256 --device cpu --corpus_zscore_n 5000 \
  --out $WORK/checkpoints/microbiome_jepa/realdata_infants_fair
echo "MB_EVAL_DONE"
