#!/usr/bin/env bash
# EXP3 BOTTLENECK SHRINK — pure-JEPA (metric_coeff=0), weak-reg K=24 80ep, idm default(1.0), sweeping
# the LATENT dim d_model toward the TRUE gLV state dim (S=24). Question: does shrinking the latent
# toward the true-state dimension improve PURE planning (no metric loss)? Reference: d128 = the existing
# plan_model_k24_lowreg (fails). We try d below/at/above the true dim.
#
# Env var DM (d_model). Submit (<=2 concurrent):
#   cd $WORK/eb_jepa
#   DM=16 sbatch examples/microbiome_jepa/run_glv_dim.sh
#   DM=24 sbatch examples/microbiome_jepa/run_glv_dim.sh
#   DM=32 sbatch examples/microbiome_jepa/run_glv_dim.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --job-name=glv_dim
#SBATCH --output=glv_dim_%j.out
#SBATCH --error=glv_dim_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
DM="${DM:-128}"
MD=$WORK/checkpoints/microbiome_jepa/plan_model_dim_${DM}
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ EXP3 train pure-JEPA (metric_coeff=0) weak-reg K=24 80ep, d_model=$DM ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model $DM --data.n_candidate 24 \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --model.regularizer.metric_coeff 0 \
  --logging.tqdm_silent True
echo "DIM_DONE d_model=$DM dir=$MD"
