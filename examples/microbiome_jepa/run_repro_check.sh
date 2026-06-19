#!/usr/bin/env bash
# GPU step 2: LOCK the guaranteed submission — confirm the one-command repro runs clean from a fresh
# bnz checkout and the headline DIRECTION reproduces (idm_on > idm_off on intervention decodability),
# plus the planning harness runs. Fast/tiny by design (the full measured numbers are already committed).
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_repro_check.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=repro_check
#SBATCH --output=repro_check_%j.out
#SBATCH --error=repro_check_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
echo "== repo HEAD $(git rev-parse --short HEAD) on $(git branch --show-current) =="

echo "############ repro: collapse-regime IDM ablation (1 seed, 20ep) — expect idm_on > idm_off ############"
$PY -m examples.microbiome_jepa.run_ablation --seeds 0 --epochs 20 --n_traj 128 --eval_n_traj 64 \
  --d_model 128 --sim_coeff_t 4 --cov_coeff 1 --std_coeff 0.25 --use_amp False \
  --out $WORK/checkpoints/microbiome_jepa/repro_check

echo "############ repro: planning harness smoke (random model — confirms plan_glv runs) ############"
$PY examples/microbiome_jepa/_smoke_plan.py
echo "REPRO_CHECK_DONE"
