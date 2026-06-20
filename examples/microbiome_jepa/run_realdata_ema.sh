#!/usr/bin/env bash
# EXP2 step (GeneJepa, incremental) — SIGReg + EMA TEACHER. One attributable change on top of the SIGReg
# encoder (view2 target = EMA copy of the model, stop-grad; BYOL/DINO/I-JEPA style). Same budget
# (100ep/50k/d256) so it is directly comparable to the SIGReg-no-EMA result (M2 linear 0.514 / MLP 0.526).
# Frozen encoder for the probe. Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_realdata_ema.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --job-name=mb_ema
#SBATCH --output=mb_ema_%j.out
#SBATCH --error=mb_ema_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerA_real.yaml
ENC=$WORK/checkpoints/microbiome_jepa/realenc_sigreg_ema
DATA=$EBJEPA_DSETS/susagi/data
EP=${EP:-100}; NS=${NS:-50000}; DM=${DM:-256}
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"
echo "############ pretrain Layer A SIGReg + EMA teacher ($NS samples, $EP ep, d$DM) ############"
$PY -m examples.microbiome_jepa.main --fname $CFG --folder $ENC \
  --loss.type bcs --model.use_ema true --model.ema_decay 0.996 \
  --data.data_dir $DATA --data.synth_n_samples $NS --model.d_model $DM \
  --optim.epochs $EP --logging.tqdm_silent True
echo "############ probe infant-env (frozen linear + MLP + corpus z-score + finetune) ############"
$PY -m examples.microbiome_jepa.realdata --checkpoint $ENC/latest.pth.tar --fname $CFG \
  --data_dir $DATA --d_model $DM --n_max 256 --device cuda --corpus_zscore_n 5000 \
  --finetune True --ft_epochs 50 \
  --out $WORK/checkpoints/microbiome_jepa/realdata_infants_sigreg_ema
echo "EMA_DONE"
