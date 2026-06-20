# Tahoe-JEPA FAST — seed 1 (SLURM 75863_0)

Run on Dalia `defq`, node `dalianvl06`, NVIDIA GB200 (184 GiB).
Submitted 2026-06-20 05:32 CEST, completed ~06:25 CEST.

## Setup
- Encoder: MLP 2000 -> 2048 -> 2048 -> 2048 -> 256 (BN + GELU)
- Projector: 256 -> 1024 -> 1024
- Regularizer: SIGReg (BCS, 256 slices, lmbd=10)
- Pathway loss coeff = 1.0
- Batch size: 16384  (32x default)
- LR: 3e-3 (scaled from 1e-3 at BS=512), cosine warmup
- 70 epochs, bf16 autocast, GPU-resident dataset, on-device augmentation

## Throughput
~1.6M cells/s steady-state (vs ~11k cells/s old DataLoader). 70 epochs of
real training = ~21 s. The rest of the wallclock (~50 min) is sklearn
LogReg probes on CPU.

## Final linear-probe macro-F1

| task       | raw   | pca50 | random-enc | **JEPA(ours)** |
|------------|-------|-------|------------|----------------|
| cell_line  | 0.887 | 0.929 | 0.705      | **0.915**      |
| drug       | 0.344 | 0.043 | 0.027      | **0.012**      |
| moa        | 0.521 | 0.128 | 0.067      | **0.036**      |

## Story
- `cell_line`: PCA50 slightly beats JEPA; both well above random-enc baseline.
  The cell-line signal is highly linear (housekeeping / identity genes).
- `drug` and `moa`: **JEPA collapses to worse than random-enc**. Raw expression
  destroys every learned representation — the SSL objective discarded the
  drug/MoA signal entirely.

This is the textbook **slow-feature collapse** (Sobal et al. 2022): with no
temporal axis and no IDM-style action prediction, the encoder locks onto the
single dominant slow factor (cell-line identity) and throws away everything
else. The drug perturbation is a *fast*, small-effect signal that gets
regularized out.

Implication for the microbiome track: this is exactly why we need
action-conditioned IDM / two-views with intervention-aware augmentations for
the microbiome world model.

## Files in this folder
- `metrics.json` — final probe scores
- `train.log` — full stdout (epochs + probe blocks + sklearn warnings)
- `gpu_util.csv` — 15s sampling of GPU utilization during the run
- `tahoe_jepa.pt` — encoder + config snapshot (50 MB)
- `config_used.yaml` — `train_fast.yaml` snapshot
- `main_fast.py` — training script snapshot
- `slurm_tahoe_fast.sh` — launcher snapshot
