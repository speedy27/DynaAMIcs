#!/usr/bin/env bash
# FINAL 3-seed IDM-ablation confirmation (the headline figure). Two regimes:
#   default      : VICReg as-is (sim=1,cov=25,std=1) — cleanest contrast (IDM ~2x action decodability)
#   collapse     : induce slow-feature collapse (sim=4,cov=1,std=0.25) — robustness check
# Submit: cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_final.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=glv_final
#SBATCH --output=glv_final_%j.out
#SBATCH --error=glv_final_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"
EP=${EP:-80}; NT=${NT:-256}; EV=${EV:-128}; SEEDS=${SEEDS:-0,1,2}; DM=${DM:-128}
OUT=$WORK/checkpoints/microbiome_jepa
RA="$PY -m examples.microbiome_jepa.run_ablation --seeds $SEEDS --epochs $EP --n_traj $NT --eval_n_traj $EV --d_model $DM --use_amp False"

echo "############ FINAL default regime (3 seeds) ############"
$RA --out $OUT/final_default
echo "############ FINAL collapse regime (3 seeds) ############"
$RA --sim_coeff_t 4 --cov_coeff 1 --std_coeff 0.25 --out $OUT/final_collapse
echo "FINAL_DONE"
