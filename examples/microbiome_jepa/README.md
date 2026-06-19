# Microbiome JEPA — an action-conditioned world model with intervention planning

A new modality for EB-JEPA: the human microbiome. We reframe the Susagi "imposter" idea as a proper
JEPA (predict in representation space + anti-collapse regularizer), add an **action-conditioned
temporal world model** of community dynamics, and **plan interventions** in latent space. See
[REPORT.md](REPORT.md) for the writeup and the headline result.

> Static masked-set JEPAs for omics already exist (GeneJepa, Cell-JEPA, JEPA-DNA). Our white space is
> the **dynamics + planning**, not the encoder.

## Headline result (measured, 3 seeds, GPU job 74610)
IDM ablation on the gLV simulator: in a collapse-prone regime the world model discards the applied
intervention without the inverse-dynamics term; the IDM term robustly recovers it.
**Intervention decodability `fast_r2_action`: 0.748 ± 0.051 (IDM on) vs 0.520 ± 0.021 (IDM off)** — a
+0.229 gap, positive in all 3 seeds, non-overlapping error bars. Figure:
[results/ablation_collapse.png](results/ablation_collapse.png). The effect is regime-dependent (strong
VICReg partially substitutes for IDM; default regime +0.073, seed-noisy) — that contrast is the point.

## Reproduce
All GPU runs are on the Dalia GB200 cluster (see the `dalia-training` skill); the gLV experiments are
fully synthetic (no data download). From the repo root on the cluster (`cd $WORK/eb_jepa`):

```bash
# Headline IDM ablation (3 seeds x {default, induce-collapse}) -> figure + JSON
sbatch examples/microbiome_jepa/run_glv_final.sh

# Intervention planning (train a world model, then latent-MPPI vs random/greedy/final-only)
sbatch examples/microbiome_jepa/run_glv_plan.sh
```

Fast local CPU smokes (no GPU, no data — minutes; use the `.venv-cpu` interpreter):
```bash
PY=.venv-cpu/bin/python   # CPU-only torch venv (the locked cu128 torch won't install on macOS)
$PY -m examples.microbiome_jepa.run_ablation --seeds 0 --epochs 6 --n_traj 64 --d_model 64   # ablation
$PY -m examples.microbiome_jepa.main --fname examples/microbiome_jepa/cfgs/layerA_vicreg.yaml \
   --optim.epochs 2 --data.size 256                                                          # Layer A
$PY examples/microbiome_jepa/eval_collapse.py        # collapse-probe sanity
$PY examples/microbiome_jepa/_smoke_plan.py          # planning harness sanity
```
Note: fire override syntax is `--key value` (bare `key=value` binds to the positional `cfg` and breaks).

## File map
| file | role |
|---|---|
| `cfgs/layerA_vicreg.yaml` | Layer A (static two-view set-JEPA) config |
| `cfgs/layerB_worldmodel.yaml` | Layer B (action-conditioned world model) config; `model.regularizer.idm_coeff` is the ablation knob |
| `main.py` | Layer A trainer (two-view VICReg/BCS over OTU communities; collapse-watch logging) |
| `train_worldmodel.py` | Layer B trainer (set-encoder + GRU predictor + IDM + VC/sim regularizer) |
| `eval_collapse.py` | collapse metric: frozen-encoder probes (fast action/dynamics vs slow identity) |
| `run_ablation.py` | IDM-ablation driver (idm-on/off × seeds → mean±se table + JSON + figure) |
| `run_glv_final.sh` | headline: 3-seed ablation, both regimes (GPU) |
| `plan_glv.py` | latent-MPPI intervention planning + MPC loop + baselines |
| `run_glv_plan.sh` | train a world model then plan (GPU) |
| `probe_downstream.py`, `baselines_port.py` | Layer A downstream probe vs Susagi MLP baseline + sequencing-tech-invariance (real-data; in progress) |
| `eb_jepa/datasets/microbiome/{glv,otu_data,transforms,traj}.py` | gLV simulator + OTU/trajectory datasets + CLR/z-score |
| `results/` | committed figures + raw JSON of the measured runs |

## 3-minute demo flow
1. The problem: microbiome is noisy/sparse/temporal — reconstruction is hopeless, so predict in latent
   space (JEPA). Show the gLV simulator's non-monotonic attractors (greedy planning fails 6/6 pairs).
2. The headline figure ([results/ablation_collapse.png](results/ablation_collapse.png)): without IDM the
   world model forgets the intervention (the Sobal slow-feature collapse); IDM recovers it — a clean
   collapse-and-recovery, 3 seeds, error bars.
3. Planning: latent-MPPI to drive a community to a target attractor vs baselines — an honest NEGATIVE
   (0% success all methods; MPPI doesn't beat random/greedy; `results/planning_*`). Shows where the
   learned world model is/ isn't good enough — reported as-is.
4. One paragraph on what we learned about JEPAs (regime-dependence of the IDM term).
