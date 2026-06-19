#!/usr/bin/env bash
# BIG BET, step 2 — train a K=24 world model in the WEAK-variance-reg regime (sim=4, cov=1, std=0.25),
# the SAME setting the IDM headline used to induce collapse. Rationale (all MEASURED): the K=24
# default-reg encoder is UNPLANNABLE — its latent only decodes state at R^2~0.76 and latent-distance is
# uninformative (corr ~0) — so every learned-model planner fails despite the task being controllable
# (oracle 100%) and the dynamics faithful (rollout ~2%). The headline showed the WEAK-reg encoder keeps
# state LINEARLY decodable at R^2~0.97 (only the ACTION features collapse, which DECODED planning does
# not need since the action is fed to the GRU explicitly). So a weak-reg encoder should yield a
# high-fidelity z->x readout and let DECODED-state MPPI close the loop. We TRAIN here; the decoded-state
# planning eval + diagnostics run on CPU after fetch (plan_glv_decoded.py / diagnose_planning.py).
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_k24_lowreg.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --job-name=glv_k24_lowreg
#SBATCH --output=glv_k24_lowreg_%j.out
#SBATCH --error=glv_k24_lowreg_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
MD=$WORK/checkpoints/microbiome_jepa/plan_model_k24_lowreg
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ train K=24 world model (WEAK reg sim4/cov1/std0.25, idm on, d_model=128, 80ep) ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model 128 --data.n_candidate 24 \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --logging.tqdm_silent True

echo "############ (reference) latent-MPPI on the weak-reg K=24 model ############"
$PY -m examples.microbiome_jepa.plan_glv --fname $CFG --checkpoint $MD/latest.pth.tar \
  --device cuda --seeds 0,1,2 --n_episodes 12 --mpc_steps 20 --horizon 6 --n_samples 128 --n_iters 3 \
  --out $WORK/checkpoints/microbiome_jepa/planning_k24_lowreg \
  --overrides '{"data.n_candidate": 24, "model.d_model": 128, "model.regularizer.sim_coeff_t": 4, "model.regularizer.cov_coeff": 1, "model.regularizer.std_coeff": 0.25}'
echo "K24_LOWREG_DONE"
