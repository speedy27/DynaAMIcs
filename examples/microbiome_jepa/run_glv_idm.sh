#!/usr/bin/env bash
# EXP2 IDM-REWEIGHT SELF-SUPERVISED CLOSURE — pure-JEPA (metric_coeff=0, NO true-state supervision),
# weak-reg K=24 d128 80ep, sweeping the inverse-dynamics loss weight idm_coeff. Question: does a
# STRONGER IDM objective induce enough latent METRIC to plan WITHOUT the isometry/metric loss, staying
# fully self-supervised? The IDM predicts the action from (z_t, z_{t+1}) only (model's own latents) —
# no true-state distance anywhere => self-supervised. Reference bars: idm_coeff=1.0 (default = the
# existing plan_model_k24_lowreg, which FAILS planning) and the hybrid mc=0.3 (upper bar).
#
# Env var IDMC (idm_coeff). Submit (<=2 concurrent for the budget):
#   cd $WORK/eb_jepa
#   IDMC=2  sbatch examples/microbiome_jepa/run_glv_idm.sh
#   IDMC=5  sbatch examples/microbiome_jepa/run_glv_idm.sh
#   IDMC=10 sbatch examples/microbiome_jepa/run_glv_idm.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --job-name=glv_idm
#SBATCH --output=glv_idm_%j.out
#SBATCH --error=glv_idm_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
IDMC="${IDMC:-1.0}"
IDMTAG="$(echo "$IDMC" | tr -d '.')"
MD=$WORK/checkpoints/microbiome_jepa/plan_model_idm_${IDMTAG}
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ EXP2 train pure-JEPA (metric_coeff=0) weak-reg K=24 d128 80ep, idm_coeff=$IDMC ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model 128 --data.n_candidate 24 \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --model.regularizer.idm_coeff $IDMC --model.regularizer.metric_coeff 0 \
  --logging.tqdm_silent True
echo "IDM_DONE idm_coeff=$IDMC dir=$MD"
