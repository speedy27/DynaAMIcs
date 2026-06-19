#!/usr/bin/env bash
# M2 #3 — LONGER + BIGGER Layer-A pretraining on the real corpus, then re-probe (fair: linear + MLP +
# corpus z-score + a labelled fine-tuned upper bound). Scales up the 30ep/20k/d128 M2 run to
# 100ep / 50k samples / d_model=256. Encoder stays FROZEN for the probe (the on-thesis claim). GPU lever;
# may help or plateau, so we measure. Checkpointed each epoch (main.py) so it can be stopped to ship.
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_realdata_big.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --job-name=mb_realbig
#SBATCH --output=mb_realbig_%j.out
#SBATCH --error=mb_realbig_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerA_real.yaml
ENC=$WORK/checkpoints/microbiome_jepa/realenc_big
DATA=$EBJEPA_DSETS/susagi/data
EP=${EP:-100}; NS=${NS:-50000}; DM=${DM:-256}
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ pretrain Layer A (two-view VICReg, $NS samples, $EP ep, d_model=$DM) ############"
$PY -m examples.microbiome_jepa.main --fname $CFG --folder $ENC \
  --data.data_dir $DATA --data.synth_n_samples $NS --model.d_model $DM \
  --optim.epochs $EP --logging.tqdm_silent True

echo "############ probe infant-env: frozen linear + frozen MLP + corpus z-score + finetune upper bound ############"
$PY -m examples.microbiome_jepa.realdata --checkpoint $ENC/latest.pth.tar --fname $CFG \
  --data_dir $DATA --d_model $DM --n_max 256 --device cuda --corpus_zscore_n 5000 \
  --finetune True --ft_epochs 50 \
  --out $WORK/checkpoints/microbiome_jepa/realdata_infants_big
echo "REALDATA_BIG_DONE"
