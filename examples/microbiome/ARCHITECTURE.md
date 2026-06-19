# Microbiome-JEPA — Architecture, Data & Pipeline

An energy-based **JEPA world model** for the gut microbiome, built on the
`eb_jepa` library. It carries the JEPA recipe to a new, noisy, high-dimensional
**biological** modality (longitudinal bacterial communities), with two
**microbiome-specific losses**. Pure JEPA: prediction in representation space,
**no reconstruction, no imposter/contrastive target**.

---

## 1. TL;DR

- **State** = a bacterial **community** = an unordered *set* of OTUs, each an
  OTU carrying a fixed **ProkBERT** DNA embedding (384-d) + its relative abundance.
- **World model** = predict the *next* community's latent from the current one,
  **conditioned on an action** (infant feeding + Δt).
- **Core** = the library's `JEPA.unroll()` (encoder → regularizer → action-conditioned
  predictor rollout → latent prediction cost).
- **Our contribution** = a permutation-invariant `SetEncoder` + two biological
  regularizers (`AlphaDiversityLoss`, `PhyloDispersionLoss`).

---

## 2. Datasets

| Source | Provides | Used for |
|---|---|---|
| **MicrobeAtlas** `samples-otus.97.mapped` (19 GB) | per-OTU **counts** per sample (97 % OTU clusters) | community composition + **abundances** |
| **ProkBERT embeddings** `prokbert_embeddings.h5` (640 MB) | **384-d** DNA embedding per OTU (16 811 used) | per-bacterium representation (no learned IDs → generalizes) |
| **DIABIMMUNE** tables (csv) | `samples.csv` (subject, **age**, country, cohort), `milk.csv` (**feeding**), `diabetes.csv` (**T1D**) | longitudinal **time axis**, **actions**, **probe labels** |

Provenance: downloaded from Zenodo (`10.5281/zenodo.18679373`). DIABIMMUNE is a
longitudinal infant gut cohort; MicrobeAtlas provides the reference OTUs + their
sequence embeddings.

**Scale after preprocessing:** 293 subjects · 3348 timepoints · 16 811 unique
OTU embeddings · Shannon α-diversity range 0.04–5.09.

---

## 3. Data pipeline — two stages

```
 STAGE 1 — precompute.py   (run ONCE, offline; needs the 19GB raw data)
 ───────────────────────────────────────────────────────────────────────
   MicrobeAtlas counts ─┐
   ProkBERT h5          ├─► stream + resolve OTU→embedding (98.5%) ─► cache.pt (24 MB)
   DIABIMMUNE metadata ─┘        + Shannon α-diversity per timepoint
                                 + phylo descriptor (abundance-weighted mean emb)

 STAGE 2 — dataset.py      (runs at TRAINING time; only needs torch + cache.pt)
 ───────────────────────────────────────────────────────────────────────
   cache.pt ─► MicrobiomeDataset ─► fixed-length windows ─► DataLoader ─► JEPA
```

**Key point:** training is **self-contained** — `dataset.py` imports only
`numpy`/`torch` and does `torch.load(cache_path)`. The 19 GB raw data is **never
touched on the cluster**; only the 24 MB `cache.pt` (which ships inside the repo
at `eb_jepa/datasets/microbiome/cache.pt`) is read. Rebuilding the cache
(`precompute.py`) requires the raw data + the sibling Microbiome-Modelling
loaders, so it is a local/offline step.

**Normalization (honest note):** abundances enter as
`log1p(scale · relative_abundance)` (compresses the long-tailed counts) and the
encoder pools tokens by abundance weight. We do **not** yet apply CLR or
per-dimension z-scoring of the ProkBERT channels — a known lever (VICReg variance
can be dominated by the single abundance channel) and a cheap ablation worth running.

---

## 4. Architecture

```
   Community at time t  =  SET of OTUs (top-64 by abundance)
        token_i = [ ProkBERT emb (384)  |  log-abundance (1) ]
                                   │   tensor  [B, 385, T, N=64, 1]
                                   ▼
   ┌───────────────────────────────────────────────────────────┐
   │  SetEncoder   (DeepSets — PERMUTATION-INVARIANT)            │
   │   per-token 1×1-conv MLP  →  abundance-weighted sum-pool    │   ← f_θ
   └───────────────────────────────────────────────────────────┘
                                   │  z_t   [B, 128, T, 1, 1]
                                   ▼
        action a_t = [ feeding one-hot | Δt ]   [B, A=5, T]
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────┐
   │  RNNPredictor (GRU)   ẑ_{t+1} = g_φ(z_t , a_t)             │   ← g_φ (world model)
   └───────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ENERGY = ‖ ẑ_{t+1} − z_{t+1} ‖²      (prediction in LATENT space)
```

Padded OTU slots carry abundance 0 → zero pooling weight → ignored for free,
so variable community sizes need no explicit mask.

### It is genuinely the library's JEPA

`examples/microbiome/main.py` builds and trains via the official class:

```python
from eb_jepa.jepa import JEPA
jepa = JEPA(encoder, action_encoder, predictor, regularizer, predcost)
preds, (loss, rloss, _, _, ploss) = jepa.unroll(
    obs, act, nsteps=K, unroll_mode="autoregressive", compute_loss=True)
```

Mapping to the three JEPA components (sujet.pdf §2):

| JEPA component | Our choice | Source |
|---|---|---|
| Encoder `f_θ` | `SetEncoder` (abundance-weighted DeepSets) | **new** (`eb_jepa/architectures.py`) |
| Predictor `g_φ` | `RNNPredictor` (action-conditioned GRU) | library |
| Regularizer `R` | `VC_IDM_Sim_Regularizer` (var+cov+temporal+inverse-dynamics) | library |
| Prediction cost | `SquareLossSeq` (latent MSE) | library |

This is the **(c) action-conditioned video-JEPA** setting of the guide, applied
to microbiome time series. **Size:** ~0.7 M params total (~0.2 M in the
`SetEncoder`), printed at startup; no custom CUDA, trains in minutes on one GPU.

---

## 5. The loss (what is optimized)

```
 L =  L_pred                       ‖ g_φ(z_t,a_t) − z_{t+1} ‖²        (JEPA energy)
   +  R_{VC+IDM+sim}               variance + covariance (anti-collapse)
                                   + temporal-similarity + inverse-dynamics
   +  λ_d · AlphaDiversityLoss     a head must recover Shannon α-diversity   ◄ bio, option 2
   +  λ_p · PhyloDispersionLoss    latent dist ≈ soft-UniFrac phylo dist      ◄ bio, option 3
   +  λ_t · TemporalVarianceLoss   per-dim std ALONG TIME ≥ γ                 ◄ the collapse FIX
```

The first two lines are the standard eb-JEPA objective (inside `unroll`). The
last three are **our microbiome-specific terms** (added as auxiliary losses in
`main.py`):

- **`AlphaDiversityLoss`** — keeps ecological diversity *decodable* from the
  latent, so imagined futures don't wash out community diversity.
- **`PhyloDispersionLoss`** — makes latent geometry respect microbial
  **phylogeny**, using each community's abundance-weighted mean ProkBERT
  embedding as a **tree-free soft-UniFrac** descriptor.
- **`TemporalVarianceLoss`** — the **collapse fix**: VICReg's variance is measured
  *across the batch*, which a slow encoder satisfies by storing host identity while
  letting z_t ≈ z_{t+1}. This applies the same hinge to the std **along the time
  axis**, forcing each trajectory to move and the predictor to model real dynamics
  (cf. Sobal et al. 2022, slow-feature collapse).

---

## 6. Evaluation

| Metric | Meaning |
|---|---|
| `age_r2` | linear probe predicting host **age** from frozen latents — the "microbiome clock" |
| `t1d_auroc` | linear probe for **T1D** host phenotype (harder, imbalanced) |
| `skill_vs_identity` | does the predictor beat the "no-change" baseline in latent space? |
| `tvar` | temporal variance of the latent — a **temporal**-collapse monitor |
| `effrank` | effective rank of the latent covariance — a **feature**-collapse monitor (dims actually used, not just std) |

Probes are fit on **train** subjects and scored on **val** subjects
(subject-disjoint, no leakage).

---

## 7. Results so far (honest)

- ✅ **Positive:** the representation recovers the **microbiome aging clock**
  (`age_r2 ≈ 0.50` on held-out subjects). `baselines.py` checks this against raw
  mean-ProkBERT and an **untrained** random encoder — report the three side by side
  so the gain is attributable to *training*, not just the set-pooling inductive bias.
- 🔬 **Diagnosed collapse (the insight):** with only VICReg/IDM the
  action-conditioned model **collapses *temporally*** (`tvar → 0`, `skill ≤ 1`):
  VICReg keeps *feature/batch* variance (high `effrank`) but **not temporal
  variance**, so the encoder stores host identity while making consecutive
  timepoints near-identical — the slow-feature collapse of Sobal et al. (2022).
- 🛠️ **Collapse fought (the fix):** `TemporalVarianceLoss` applies the VICReg
  hinge **along time**. The `tvar=0` vs full conditions (`run_ablation.sh` →
  `aggregate.py`) are the controlled before/after: does the fix lift
  `skill_vs_identity` above 1 *without* hurting `effrank` / `age_r2`?
- 🔬 **Hard:** the T1D probe stays near chance (few positives, subtle signal) — an
  honest negative we keep in the report.

---

## 8. How to run

```bash
# (once, local) build the cache from raw data
python -m eb_jepa.datasets.microbiome.precompute

# single training run
python -m examples.microbiome.main --fname examples/microbiome/cfgs/train.yaml \
    optim.epochs=50

# evaluate (metrics + latent PCA figure)
python -m examples.microbiome.eval --ckpt $EBJEPA_CKPTS/microbiome/microbiome_jepa.pt

# on the cluster (GPU)
sbatch examples/microbiome/train.slurm
```

### Controlled comparison (the deliverable) — one change at a time, 3 seeds

```bash
# launches 4 conditions x 3 seeds (cluster sbatch, or LOCAL=1 for a smoke test):
#   baseline · div+phylo (pre-fix) · tvar (fix only) · div+phylo+tvar (full)
bash examples/microbiome/run_ablation.sh

# each run writes <ckpt>/<condition>/seed<seed>/metrics.json; collapse them into
# one table + bar chart (skill / effrank / age_r2 / t1d_auroc, mean +/- std):
python -m examples.microbiome.aggregate --root <ckpt>/microbiome
```

---

## 9. File map

| File | Role |
|---|---|
| `eb_jepa/architectures.py` → `SetEncoder` | abundance-weighted DeepSets encoder |
| `eb_jepa/losses.py` → `AlphaDiversityLoss`, `PhyloDispersionLoss`, `TemporalVarianceLoss`, `effective_rank` | bio losses + collapse fix + collapse metric |
| `eb_jepa/datasets/microbiome/precompute.py` | raw data → 24 MB cache (offline) |
| `eb_jepa/datasets/microbiome/dataset.py` | cache → windowed batches (training) |
| `examples/microbiome/main.py` | wires `JEPA` + bio losses + probes; trains; dumps `metrics.json` |
| `examples/microbiome/eval.py` | metrics + latent-space + collapse (corr / eff-rank) figure |
| `examples/microbiome/baselines.py` | raw / random-encoder / trained probe comparison |
| `examples/microbiome/aggregate.py` | ablation runs → table + bar chart |
| `examples/microbiome/cfgs/train.yaml` | config (model dims, loss coeffs, optim) |
| `examples/microbiome/train.slurm` · `run_ablation.sh` | SLURM launcher · 4×3 ablation sweep |
| `tests/test_microbiome.py` | encoder / losses / JEPA wiring tests |

---

## 10. Related work (cross-links in `references/paper/`)

| Theme | Papers | How it connects |
|---|---|---|
| **Slow-feature / temporal collapse** | `jepa-slow-features` (Sobal 2022), `temporal-straightening` | the theory behind our finding + the `TemporalVarianceLoss` fix |
| **Anti-collapse regularizers** | `lejepa` (SIGReg/BCS, also in `losses.py`), `reconstruction-or-semantics` | VICReg-vs-SIGReg ablation; why no reconstruction |
| **Set / permutation-invariant JEPA** | `point-jepa`, `stem-jepa`, `s-jepa` | precedent for the `SetEncoder` (DeepSets over OTUs) |
| **Biological / sequence JEPA** | `protein-jepa`, `jepa-dna`, `polymer-jepa`, `graph-jepa` | JEPA on bio / sequence modalities (ProkBERT = DNA-LM) |
| **Time-series JEPA** | `ts-jepa`, `mts-jepa`, `t-jepa` | comparable temporal recipes; "scales to other TS" bonus |

External: *Abundance-Aware Set Transformer for Microbiome Sample Embedding*
(arXiv 2508.11075) — direct precedent for an abundance-weighted set encoder.
