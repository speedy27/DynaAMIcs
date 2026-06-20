#!/usr/bin/env bash
# EXP1 GENERALIZATION — train the M3 HYBRID closure (metric_coeff=0.3, weak-reg, d128, 80ep, idm on,
# K=n_species full actuation) on a DIFFERENT gLV INSTANCE. The instance is defined by STRUCTURAL knobs
# (n_guilds / comp strengths / species count) read from env vars; the gLV builds A + attractors
# deterministically from them, so each is a genuinely different system. Config identical to the headline
# closure EXCEPT the gLV instance => a clean generalization test.
#
# Env vars (defaults = the headline g3/S24 baseline):
#   NAME (checkpoint tag), S (n_species), K (n_candidate=full), G (n_guilds), CS (comp_strong), CW (comp_weak)
# Submit (one per instance, keep <=2 concurrent for the 1-2 GPU budget):
#   cd $WORK/eb_jepa
#   NAME=g4_s24  S=24 K=24 G=4 CS=-2.5 CW=-0.4  sbatch examples/microbiome_jepa/run_glv_gen.sh
#   NAME=g3_s18  S=18 K=18 G=3 CS=-2.5 CW=-0.4  sbatch examples/microbiome_jepa/run_glv_gen.sh
#   NAME=g5_s30  S=30 K=30 G=5 CS=-2.5 CW=-0.4  sbatch examples/microbiome_jepa/run_glv_gen.sh
#   NAME=g3_s24_strongcomp S=24 K=24 G=3 CS=-3.5 CW=-0.25 sbatch examples/microbiome_jepa/run_glv_gen.sh
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --job-name=glv_gen
#SBATCH --output=glv_gen_%j.out
#SBATCH --error=glv_gen_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
NAME="${NAME:-baseline_g3_s24}"; S="${S:-24}"; K="${K:-24}"; G="${G:-3}"; CS="${CS:--2.5}"; CW="${CW:--0.4}"
MD=$WORK/checkpoints/microbiome_jepa/plan_model_gen_${NAME}
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ EXP1 train HYBRID metric (mc=0.3) on instance $NAME (S=$S K=$K guilds=$G cs=$CS cw=$CW) ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model 128 \
  --data.n_species $S --data.n_candidate $K --data.n_guilds $G \
  --data.comp_strong $CS --data.comp_weak $CW \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --model.regularizer.metric_coeff 0.3 \
  --logging.tqdm_silent True
echo "GEN_DONE name=$NAME dir=$MD"
