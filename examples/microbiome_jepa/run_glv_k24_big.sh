#!/usr/bin/env bash
# BIG BET, step 3 — does a HIGHER-CAPACITY weak-reg K=24 world model convert the partial planning
# success into a convincing one? MEASURED so far (weak-reg d_model=128): dynamics are faithful (rollout
# ~0.8% over 20 steps) and the ONLY cap is state-DECODABILITY of the latent (decoder R^2 0.89), which
# limits decoded-state MPPI to 2.8% success / final 2.78 (best of all methods, first non-zero). The
# readout-fidelity trend (R^2 0.76->0.89 => success 0%->2.8%) predicts that a more state-decodable
# encoder plans better. Test it: bigger encoder (d_model=256), more data (n_traj=512), more epochs
# (150), same weak-reg regime. Decoded-state planning eval runs on CPU after fetch (plan_glv_decoded).
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_k24_big.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=glv_k24_big
#SBATCH --output=glv_k24_big_%j.out
#SBATCH --error=glv_k24_big_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
MD=$WORK/checkpoints/microbiome_jepa/plan_model_k24_big
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ train K=24 BIG weak-reg world model (d_model=256, n_traj=512, 150ep) ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 150 --model.d_model 256 --data.n_candidate 24 --data.n_traj 512 \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --logging.tqdm_silent True

echo "############ (reference) latent-MPPI on the big weak-reg K=24 model ############"
$PY -m examples.microbiome_jepa.plan_glv --fname $CFG --checkpoint $MD/latest.pth.tar \
  --device cuda --seeds 0,1,2 --n_episodes 12 --mpc_steps 20 --horizon 6 --n_samples 128 --n_iters 3 \
  --out $WORK/checkpoints/microbiome_jepa/planning_k24_big \
  --overrides '{"data.n_candidate": 24, "model.d_model": 256, "data.n_traj": 512, "model.regularizer.sim_coeff_t": 4, "model.regularizer.cov_coeff": 1, "model.regularizer.std_coeff": 0.25}'
echo "K24_BIG_DONE"
