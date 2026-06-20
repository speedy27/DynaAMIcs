#!/usr/bin/env bash
# SIGReg BIG BET (EXP 1) — swap VICReg -> SIGReg (eb_jepa's BCS) for Layer-A real-corpus pretraining,
# changing NOTHING else, at the SAME budget as the expanded VICReg #3 run (100ep / 50k samples / d256)
# so it is directly comparable. Thesis: M2 tie + M3 not-closed + tech-invariance all bottleneck on the
# VICReg representation; SIGReg's isotropic-Gaussian latent may make distances meaningful (fix M3
# geometry) and improve probes. Encoder stays FROZEN for all downstream evals. Checkpointed each epoch.
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_realdata_sigreg.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --job-name=mb_sigreg
#SBATCH --output=mb_sigreg_%j.out
#SBATCH --error=mb_sigreg_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerA_real.yaml
DATA=$EBJEPA_DSETS/susagi/data
EP=${EP:-100}; NS=${NS:-50000}; DM=${DM:-256}
ENC=${ENC:-$WORK/checkpoints/microbiome_jepa/realenc_sigreg_d${DM}}   # d_model-specific (no clobber)
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ pretrain Layer A with SIGReg (BCS), $NS samples, $EP ep, d_model=$DM ############"
$PY -m examples.microbiome_jepa.main --fname $CFG --folder $ENC \
  --loss.type bcs \
  --data.data_dir $DATA --data.synth_n_samples $NS --model.d_model $DM \
  --optim.epochs $EP --logging.tqdm_silent True

echo "############ probe infant-env: frozen linear + frozen MLP + corpus z-score + finetune upper bound ############"
$PY -m examples.microbiome_jepa.realdata --checkpoint $ENC/latest.pth.tar --fname $CFG \
  --data_dir $DATA --d_model $DM --n_max 256 --device cuda --corpus_zscore_n 5000 \
  --finetune True --ft_epochs 50 \
  --out $WORK/checkpoints/microbiome_jepa/realdata_infants_sigreg_d${DM}
echo "SIGREG_DONE"
