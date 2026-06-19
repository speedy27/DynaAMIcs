#!/usr/bin/env bash
# GPU sweep for the IDM-ablation headline (gLV, fully synthetic — no data needed).
# Tests whether idm-on vs idm-off diverge, across regularizer regimes. The Sobal mechanism:
# weak variance-reg + strong temporal-smoothness tempts slow-feature collapse that IDM rescues.
# Submit from the repo root on the login node:
#   cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_sweep.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=glv_ablation
#SBATCH --output=glv_ablation_%j.out
#SBATCH --error=glv_ablation_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
echo "== node $(hostname) arch $(uname -m) =="
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

# Shrunk for fast iteration (launch-overhead-bound: tiny model on GB200). d_model 128, 60 ep, 256 traj.
EP=${EP:-60}; NT=${NT:-256}; EV=${EV:-96}; SEEDS=${SEEDS:-0}; DM=${DM:-128}
OUT=$WORK/checkpoints/microbiome_jepa
RA="$PY -m examples.microbiome_jepa.run_ablation --seeds $SEEDS --epochs $EP --n_traj $NT --eval_n_traj $EV --d_model $DM --use_amp False"

echo "############ SWEEP A: default (sim=1, cov=25, std=1) ############"
$RA --out $OUT/sweep_A
echo "############ SWEEP B: induce-collapse (sim=4, cov=1, std=0.25) ############"
$RA --sim_coeff_t 4 --cov_coeff 1 --std_coeff 0.25 --out $OUT/sweep_B
echo "############ SWEEP C: mid (sim=2, cov=5, std=0.5) ############"
$RA --sim_coeff_t 2 --cov_coeff 5 --std_coeff 0.5 --out $OUT/sweep_C
echo "ALL_SWEEPS_DONE"
