#!/usr/bin/env bash
# M3: train ONE collapse-regime idm_on world model to a known folder, then plan interventions with it.
# (Self-contained + reproducible — avoids the timestamp-named ablation checkpoint dirs.)
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_plan.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --job-name=glv_plan
#SBATCH --output=glv_plan_%j.out
#SBATCH --error=glv_plan_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
MD=$WORK/checkpoints/microbiome_jepa/plan_model
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ train collapse-regime idm_on world model (d_model=128, 80ep) ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model 128 \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --logging.tqdm_silent True

echo "############ plan interventions (random / greedy / final_only / mppi) ############"
$PY -m examples.microbiome_jepa.plan_glv --fname $CFG --checkpoint $MD/latest.pth.tar \
  --device cuda --seeds 0,1,2 --n_episodes 12 --mpc_steps 20 --horizon 6 --n_samples 128 --n_iters 3 \
  --out $WORK/checkpoints/microbiome_jepa/planning \
  --overrides '{"model.d_model": 128, "model.regularizer.sim_coeff_t": 4, "model.regularizer.cov_coeff": 1, "model.regularizer.std_coeff": 0.25}'
echo "PLAN_DONE"
