---
name: dalia-training
description: >-
  How to launch and monitor training/eval jobs on the Dalia GB200 cluster (IDRIS) for the
  HackTheWorld(s) / EB-JEPA hackathon. Use this whenever the task involves running training,
  evaluation, sweeps, or any GPU job on Dalia — connecting over SSH, setting up the two-arch
  uv venvs, submitting SLURM jobs (sbatch / launch_sbatch.py), the mandatory Vivatech
  reservation, the /lustre/work layout, and job monitoring. Verified live on 2026-06-19.
---

# Running training on Dalia (GB200 cluster, HackTheWorld(s))

Everything here was verified by connecting to the live cluster on 2026-06-19. Cluster facts
are ground truth; the repo workflow (`env.sh`, `setup.sh`, `launch_sbatch.py`) is the
canonical way to run things — prefer it over hand-rolled scripts.

## 0. You operate from the user's Mac — prefix every cluster command with SSH

Claude runs locally; Dalia is remote. The private key is already on disk. Run cluster
commands non-interactively like this:

```bash
ssh -i ~/.ssh/dynamics@bbenziada -o BatchMode=yes bbenziada@dalia.idris.fr '<remote command>'
```

- Host: `dalia.idris.fr` · user: `bbenziada` · key: `~/.ssh/dynamics@bbenziada` (perms `600`).
- Access only works from the competition network (Wi-Fi/wired). Off-network → port 22 times out.
- The Wi-Fi/network-auth password is **separate** from SSH and is not needed for any cluster
  command. Never put secrets in this file or in the repo.
- Login lands on `dalia1`/`dalia2` and prints a gcc module + a post-quantum SSH warning on
  every connection — that noise is normal, ignore it.
- For interactive/long work, prefer a single `srun ... bash` or an `sbatch` job over many
  separate `ssh` calls.

## 1. Hardware & scheduler (verified)

| Thing | Value |
|---|---|
| Login nodes | `dalia1` / `dalia2`, **x86_64**, RHEL 9.5 |
| Compute nodes | `dalianvl[01-18]` — 18 nodes, **aarch64 (ARM Grace)** |
| GPUs | **4× NVIDIA GB200** per node (Grace-Blackwell, ~185 GB HBM3e each), driver 595 |
| CPU / RAM | 144 cores, ~1.5 TB RAM per node |
| Scheduler | SLURM 24.11, partition **`defq`** (only/default), max wall **2 days**, QOS `normal` |
| Account | `vivatech-dynamics` (this user's team allocation) |
| CUDA | login modules: `cuda12.8/toolkit`; compute nodes ship CUDA 13.1/13.2 in `/usr/local` |
| Containers | `enroot`, `singularity`, `apptainer` available (no docker/podman/pyxis) |

### ⚠️ Two architectures = two venvs (the #1 gotcha)
Login is **x86_64**, compute GB200 is **aarch64**. Compiled wheels (torch, …) are NOT
portable across them. The repo keeps two arch-specific venvs and `env.sh` auto-selects by
`$(uname -m)`:
- `venvs/eb_jepa_x86_64` → runs the **launcher** on the login node (builds config + submits).
- `venvs/eb_jepa_aarch64` → runs the **actual training** on compute nodes (torch + CUDA).

Building on the login node and running on a compute node → `Exec format error`. Always build
the aarch64 venv on a compute node (this is what `setup.sh` does for you).

### ⚠️ The Vivatech reservation is MANDATORY
The 18 GB200 nodes are held under reservation **`Vivatech`** (active until
**2026-06-21T00:00**). Without `--reservation=Vivatech` your job is queued then **revoked**
(`Required node not available (down, drained or reserved)`). The repo defaults already pass
it (`EBJEPA_SLURM_RESERVATION=Vivatech`). For ad-hoc `srun`/`sbatch`, add it yourself.
(There's also a power-outage maintenance reservation `Coupure_Electrique_23/06` on 06-23 —
after the event, ignore.)

### ⚠️ Do NOT request memory
The Dalia scheduler **rejects** jobs that pass `--mem` / `--mem-per-gpu`; memory is allocated
proportional to requested cores. Leave it unset (the repo's `EBJEPA_SLURM_MEM` is empty).

## 2. Everything lives on /lustre/work, not $HOME

- `$HOME` = `/lustre/home/extusers/bbenziada` — **small quota**; git/venvs/downloads fill it
  and fail. Don't work here.
- Work dir = `/lustre/work/vivatech-dynamics/$USER` (`$WORK` after sourcing `env.sh`).
- `env.sh` redirects every cache (uv, HF, torch, triton, pip, wandb) under `$WORK/.cache`.
- Clone + run the repo from `$WORK`; `setup.sh` relocates a home-cloned repo there automatically.

## 3. First-time setup (per arch venv)

```bash
# on the login node, from the work copy of the repo
cd /lustre/work/vivatech-dynamics/$USER/eb_jepa
bash setup.sh          # builds eb_jepa_x86_64 here, THEN submits a job that builds eb_jepa_aarch64
echo "source $(pwd)/env.sh" >> ~/.bashrc   # make env persistent
source env.sh
```

`setup.sh` submits a short SLURM job named `eb_jepa_setup` that builds the **aarch64** venv on
a compute node. **Wait for it to finish** (`sq`) before launching training — until then the
aarch64 venv has no torch and jobs fail to import it.

Sanity check the whole environment on a GPU node:
```bash
sbatch slurm_test.sh   # runs pytest on a GPU node; check with: log -f
```

## 4. Launch training — use launch_sbatch.py

`env.sh` must be sourced first (sets `$WORK`, SLURM defaults, auto-detects account/QOS, adds
`cluster/` to PATH). Then:

```bash
# Single run (no seed averaging), W&B off (see W&B gotcha below):
python -m examples.launch_sbatch --example ac_video_jepa --single --logging.log_wandb false

# 3-seed run with a named sweep:
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment

# Full hyperparameter sweep:
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment --full-sweep

# Override a config value or SLURM resources:
python -m examples.launch_sbatch --example ac_video_jepa --optim.lr 0.0005
python -m examples.launch_sbatch --example ac_video_jepa --cpus-per-task 16 --time-min 240 --gpus-per-node 4
```

Examples available: `image_jepa`, `video_jepa`, `ac_video_jepa` (see `EXAMPLE_CONFIGS` in
`examples/launch_sbatch.py`). The launcher uses **submitit**; `COMPUTE_PYTHON` points at the
aarch64 venv so the pickled job runs with the right interpreter.

### ⚠️ W&B gotcha
Training configs default to `logging.log_wandb: true`. A shell env var does NOT propagate
through submitit, so without W&B creds the compute job crashes
(`You must call wandb.init() before wandb.log()`). Disable it **via the config override**
(lowercase `false`): `--logging.log_wandb false`. Or run `wandb login` on the login node first.

## 5. Ad-hoc GPU work (quick interactive test)

```bash
ssh -i ~/.ssh/dynamics@bbenziada -o BatchMode=yes bbenziada@dalia.idris.fr \
  'srun --account=vivatech-dynamics --partition=defq --reservation=Vivatech \
        --gres=gpu:1 --cpus-per-task=8 --time=00:10:00 \
        bash -lc "nvidia-smi; uname -m"'
```

Minimal `#SBATCH` header for a hand-written job:
```bash
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1            # up to gpu:4 per node
#SBATCH --time=02:00:00         # max 2-00:00:00
#SBATCH --output=%x_%j.out --error=%x_%j.err
# NO --mem / --mem-per-gpu (scheduler rejects it)
# --account omitted → uses your default team account
```

## 6. Monitoring (read-only helpers in cluster/, on PATH after `source env.sh`)

| Command | Shows |
|---|---|
| `sq` | your running/pending jobs (color-coded) |
| `qall` | every job on `defq` + per-user summary |
| `gpus` | per-node GPU used/free across all 18 nodes |
| `users` | GPU/CPU/node usage per user |
| `log [JOBID] [-f]` | a job's stdout (latest if no id; `-f` to tail; falls back to stderr) |

Plain SLURM also works: `squeue -u bbenziada`, `scontrol show job <id>`, `sacct`,
`scancel <id>`, `sinfo`, `scontrol show res`.

## 7. Quick decision guide for Claude

1. Confirm on-network (a probe `ssh ... 'hostname'` should return `dalia1`/`dalia2`, not time out).
2. Ensure the repo is on `/lustre/work` and both venvs exist (`setup.sh` done, `eb_jepa_setup` finished).
3. To run training: `source env.sh` → `python -m examples.launch_sbatch ...` (add `--logging.log_wandb false` unless W&B is configured).
4. Always carry `--reservation=Vivatech`, never pass `--mem`, keep wall time ≤ 2 days.
5. Monitor with `sq` / `log -f`; surface failures (stderr) to the user honestly.
