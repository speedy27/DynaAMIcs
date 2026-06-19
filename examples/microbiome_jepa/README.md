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

# Planning loop (the big bet): train a K=24 world model, then latent-MPPI vs baselines
sbatch examples/microbiome_jepa/run_glv_plan_k24.sh     # K=24, default reg
sbatch examples/microbiome_jepa/run_glv_k24_lowreg.sh   # K=24, weak reg (best decoded planning)

# Real-data Layer A: corpus pretrain + FAIR probe (frozen linear + MLP + corpus z-score)
sbatch examples/microbiome_jepa/run_realdata_big.sh     # 100ep/50k/d256 (+ finetune upper bound)

# Sequencing-tech invariance (amplicon vs wgs; JEPA vs raw / random / Susagi-imposter reps)
sbatch examples/microbiome_jepa/run_tech_invariance.sh
```

Fast local CPU runs (no GPU, no data download — the gLV planning DIAGNOSIS is fully CPU, seconds–minutes):
```bash
PY=.venv-cpu/bin/python   # CPU-only torch venv (the locked cu128 torch won't install on macOS)
$PY -m examples.microbiome_jepa.oracle_K_sweep        # controllability curve (the planning gate) + figure
$PY -m examples.microbiome_jepa.plan_glv_decoded --checkpoint <wm_ckpt> --tag lowreg \
   --overrides '{"data.n_candidate":24,"model.d_model":128}'   # decoded-state MPPI + readout-fidelity
$PY -m examples.microbiome_jepa.diagnose_planning --checkpoint <wm_ckpt> --n_candidate 24  # 3 diagnostics
$PY -m examples.microbiome_jepa.make_planning_figure  # rebuild the planning figure from result JSONs
$PY -m examples.microbiome_jepa.run_ablation --seeds 0 --epochs 6 --n_traj 64 --d_model 64   # ablation smoke
$PY examples/microbiome_jepa/eval_collapse.py         # collapse-probe sanity
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
| `oracle_K_sweep.py` | controllability gate: oracle (perfect-model) MPPI vs action-panel size K → curve + figure |
| `plan_glv_decoded.py` | decoded-state MPPI (state-aligned cost via a linear/MLP readout) + readout-fidelity sweep |
| `diagnose_planning.py` | 3 planning diagnostics: oracle controllability / latent-cost alignment / rollout fidelity |
| `make_planning_figure.py` | builds the planning figure (controllability + readout-fidelity) from result JSONs |
| `run_glv_plan_k24.sh`, `run_glv_k24_lowreg.sh`, `run_glv_k24_big.sh` | K=24 planning world models (GPU) |
| `realdata.py` | real-corpus Layer A probe: frozen **linear + MLP** vs Susagi MLP on the true abundance matrix |
| `tech_invariance.py` | sequencing-tech invariance: amplicon-vs-wgs recoverability from JEPA vs raw / random / Susagi reps |
| `run_realdata_big.sh`, `run_realdata_eval.sh`, `run_tech_invariance.sh` | real-data pretrain/probe + tech-invariance |
| `probe_downstream.py`, `baselines_port.py` | earlier Layer A downstream probe scaffolding |
| `eb_jepa/datasets/microbiome/{glv,otu_data,transforms,traj}.py` | gLV simulator + OTU/trajectory datasets + CLR/z-score |
| `results/` | committed figures + raw JSON of the measured runs |

## 3-minute demo flow
1. The problem: microbiome is noisy/sparse/temporal — reconstruction is hopeless, so predict in latent
   space (JEPA). Show the gLV simulator's non-monotonic attractors (greedy planning fails 6/6 pairs).
2. The headline figure ([results/ablation_collapse.png](results/ablation_collapse.png)): without IDM the
   world model forgets the intervention (the Sobal slow-feature collapse); IDM recovers it — a clean
   collapse-and-recovery, 3 seeds, error bars.
3. Planning (the big bet): a fully DIAGNOSED, partially-closed negative. The oracle controllability
   curve ([results/oracle_K_sweep.png](results/oracle_K_sweep.png)) shows the task is solvable only at
   K=24; the learned planner then fails because raw latent distance is an uninformative cost (corr≈0)
   while the dynamics are faithful (~2%); a state-aligned DECODED cost moves the planner closer *in
   proportion to readout fidelity* (R² 0.78→0.89 ⇒ final dist 4.12→3.01, the best of any method, though
   still 0% at tol=1.0; [results/planning_diagnosis.png](results/planning_diagnosis.png)).
   Pinpoints exactly where the JEPA rep helps and where it doesn't.
4. Real data + invariance: a frozen corpus-pretrained encoder + linear probe **ties a supervised MLP on
   AUC** (infant-env, 0.896 vs 0.890); sequencing-tech invariance is an honest NEGATIVE (the rep faithfully
   keeps the amplicon-vs-wgs protocol signal — VICReg preserves nuisance it isn't taught to drop).
5. What we learned about JEPAs: regime-dependence of the IDM term; **faithful-for-one-step-prediction ≠
   metric-for-multi-step-planning**; a representation good for probing can still be the planning bottleneck.
