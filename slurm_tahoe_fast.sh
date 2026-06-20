#!/bin/bash
#SBATCH --job-name=tahoe_fast
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=tahoe_fast_seed%a_%j.out
#SBATCH --error=tahoe_fast_seed%a_%j.err
#SBATCH --array=0-2

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

SEEDS=(1 1000 10000)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Host: $(hostname) | Seed: $SEED | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

module load python312
if ! uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
uv sync --dev --project "$REPO"

SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d)_$SLURM_ARRAY_JOB_ID}"
OUT_ROOT="${EBJEPA_CKPTS:-$REPO/checkpoints}/tahoe_fast_${SWEEP_TAG}"
RUN_DIR="$OUT_ROOT/seed${SEED}_fast"
mkdir -p "$RUN_DIR"
echo "=== output: $RUN_DIR ==="

CACHE_PATH="${TAHOE_CACHE:-$EBJEPA_WORK/tahoe/cache.pt}"
echo "=== cache: $CACHE_PATH ==="
ls -la "$CACHE_PATH"

# Background GPU-util sampler (proof of saturation)
(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,noheader >> "$RUN_DIR/gpu_util.csv"
    sleep 15
  done
) &
UTIL_PID=$!
trap "kill $UTIL_PID 2>/dev/null || true" EXIT

EBJEPA_CKPTS="$RUN_DIR" uv run --project "$REPO" python -u -m examples.tahoe.main_fast \
    --fname examples/tahoe/cfgs/train_fast.yaml \
    meta.seed=$SEED \
    data.cache_path=$CACHE_PATH \
    logging.log_wandb=false 2>&1 | tee "$RUN_DIR/train.log"

kill $UTIL_PID 2>/dev/null || true
echo "=== seed $SEED done ==="

if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "=== waiting 30s for other tasks to flush, then aggregating ==="
    sleep 30
    uv run --project "$REPO" python -c "
import json, glob, statistics as S
runs = sorted(glob.glob('$OUT_ROOT/seed*/metrics.json'))
print(f'found {len(runs)} runs')
agg = {}
for r in runs:
    with open(r) as f: m = json.load(f)
    for task, baselines in m.get('metrics', {}).items():
        if not isinstance(baselines, dict): continue
        for bname, vals in baselines.items():
            key = f'{task}.{bname}'
            agg.setdefault(key, []).append(vals['macro_f1'])
print()
print(f'{\"task.baseline\":35s}  mean   std    n')
print('-'*60)
for k, vs in sorted(agg.items()):
    m = sum(vs)/len(vs); sd = S.pstdev(vs) if len(vs)>1 else 0.0
    print(f'{k:35s}  {m:.3f}  {sd:.3f}  {len(vs)}')
" | tee "$OUT_ROOT/seed_stability.txt"
fi
