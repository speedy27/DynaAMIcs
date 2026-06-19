#!/usr/bin/env bash
# M2: pretrain Layer-A set-JEPA on the REAL MicrobeAtlas corpus, then probe infant-env vs Susagi baseline.
# Checkpointed each epoch (main.py saves latest.pth.tar) so it can be stopped and the locked submission shipped.
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_realdata.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=mb_realdata
#SBATCH --output=mb_realdata_%j.out
#SBATCH --error=mb_realdata_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerA_real.yaml
ENC=$WORK/checkpoints/microbiome_jepa/realenc
DATA=$EBJEPA_DSETS/susagi/data
EP=${EP:-30}; NS=${NS:-20000}
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ pretrain Layer A on real corpus (two-view VICReg, $NS samples, $EP ep) ############"
$PY -m examples.microbiome_jepa.main --fname $CFG --folder $ENC \
  --data.data_dir $DATA --data.synth_n_samples $NS --model.d_model 128 \
  --optim.epochs $EP --logging.tqdm_silent True

echo "############ probe: infant-env (JEPA linear probe vs Susagi MLP on true abundance matrix) ############"
$PY -m examples.microbiome_jepa.realdata --checkpoint $ENC/latest.pth.tar --fname $CFG \
  --data_dir $DATA --d_model 128 --n_max 256 --device cuda \
  --out $WORK/checkpoints/microbiome_jepa/realdata_infants
echo "REALDATA_DONE"
