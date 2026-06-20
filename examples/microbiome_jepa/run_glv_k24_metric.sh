#!/usr/bin/env bash
# HYBRID M3 — train a K=24 world model in the WEAK-reg regime (sim4/cov1/std0.25, idm on, d128, 80ep),
# the EXACT pure-JEPA substrate config, with ONE thing added: the metric-preserving isometry auxiliary
# (model.regularizer.metric_coeff>0). This bakes the TRUE gLV state metric into the latent => HYBRID,
# NOT pure JEPA (it uses ground-truth-state supervision). Goal: test whether a metric latent closes the
# planning loop the pure-JEPA negative could not, and measure the recognition/rollout tradeoff.
#
# metric_coeff is read from env var MC (default 1.0); the checkpoint dir encodes it so a sweep does not
# clobber. Eval (gate + planning + recognition) runs on CPU after fetch.
# Submit (sweep on the 2 GPUs):
#   cd $WORK/eb_jepa && for mc in 0.5 3.0; do MC=$mc sbatch examples/microbiome_jepa/run_glv_k24_metric.sh; done
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --job-name=glv_k24_metric
#SBATCH --output=glv_k24_metric_%j.out
#SBATCH --error=glv_k24_metric_%j.out
set -e
source "${SLURM_SUBMIT_DIR}/env.sh"
cd "$EBJEPA_REPO"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CFG=examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml
MC="${MC:-1.0}"
MCTAG="$(echo "$MC" | tr -d '.' )"
MD=$WORK/checkpoints/microbiome_jepa/plan_model_k24_metric_mc${MCTAG}
$PY -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"

echo "############ train K=24 HYBRID metric world model (weak reg + metric_coeff=$MC, idm on, d128, 80ep) ############"
$PY -m examples.microbiome_jepa.train_worldmodel --fname $CFG --folder $MD \
  --optim.epochs 80 --model.d_model 128 --data.n_candidate 24 \
  --model.regularizer.sim_coeff_t 4 --model.regularizer.cov_coeff 1 --model.regularizer.std_coeff 0.25 \
  --model.regularizer.metric_coeff $MC \
  --logging.tqdm_silent True
echo "K24_METRIC_DONE mc=$MC dir=$MD"
