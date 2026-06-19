# EB-JEPA for the Microbiome: an action-conditioned world model with intervention planning

**Status:** the IDM-ablation headline and the planning result are MEASURED and final (jobs 74610,
74718); the real-data Layer A probe is running. Every number is labelled MEASURED / PENDING — no
fabricated values, and the reversing seed is kept in the figures.

## Thesis (what is new)
Static masked-set JEPAs for omics already exist (GeneJepa, Cell-JEPA, JEPA-DNA) — so we do **not**
claim "the first JEPA for biology". Our contribution is the part they lack: an **action-conditioned
temporal world model of microbiome community dynamics**, trained as a JEPA, with **in-silico
intervention planning** (reach a target community by optimizing a sequence of interventions). The
permutation-invariant set-JEPA encoder is the enabling substrate (Layer A); the dynamics + planning is
the headline (Layer B).

## Hypothesis (the experiment)
JEPAs preferentially encode *slow* features (Sobal et al. 2022). On microbiome trajectories the slow
feature is a community's static identity (its initial composition / basin); the *fast* signal is the
time-varying state and the **intervention** that drives it. We predict: a VICReg-style temporal JEPA,
trained only to predict the next latent, will **collapse onto slow features and discard the
intervention**; adding the **inverse-dynamics (IDM)** term — recover the action from consecutive
latents — forces the encoder to retain the fast/intervention information. This is a controlled
**collapse-and-recovery** ablation.

## Setup
- **Simulator (the "Two Rooms" of microbiome).** A generalized Lotka–Volterra (gLV) model with
  guild structure + cyclic competition, giving ≥2 verified-stable attractors and — crucially —
  **non-monotonic reachability**: a greedy "reduce distance every step" policy fails on 6/6 tested
  (init, target) pairs (MEASURED), so planning is non-trivial. Interventions are a continuous
  delta-abundance on a curated K-taxon candidate panel (a "probiotic formulation"). Fully synthetic →
  clean, controllable, unlimited, with ground-truth dynamics. `eb_jepa/datasets/microbiome/glv.py`.
- **Encoder (Layer A).** Permutation-invariant set-transformer over OTU tokens
  `concat(ProkBERT-style 384-d embedding, z-scored CLR log-abundance)`; output `[B, D, T, 1, 1]`,
  matching eb_jepa's encoder contract so the predictor / regularizer / planner work unchanged.
  Verified permutation- and mask-invariant. `eb_jepa/architectures.py:SetTransformerEncoder`.
- **World model (Layer B).** The eb_jepa GRU `RNNPredictor` (action = GRU input, state = hidden) +
  action encoder + `InverseDynamicsModel`, trained with `VC_IDM_Sim_Regularizer` (variance/covariance
  anti-collapse + temporal smoothness + IDM) and a latent-space prediction loss.
  `examples/microbiome_jepa/train_worldmodel.py`. The single ablation knob is
  `model.regularizer.idm_coeff` (on vs 0).
- **Collapse metric (frozen-encoder linear probes, trajectory-split, standardized).**
  `fast_r2_action` = decode the applied intervention aₜ from `[zₜ, zₜ₊₁]` (a *fresh* probe, not the
  trained IDM — measures whether the *encoder* kept the intervention); `fast_r2_delta` = decode the
  one-step state change; `fast_r2_state` = decode the current state; `slow_r2_init` = decode the
  trajectory's initial state (a slow feature). `eval_collapse.py`.

## Result — IDM ablation (collapse-and-recovery) — MEASURED (job 74610; 3 seeds; 80 epochs; d_model=128)
Frozen-encoder linear probes (Ridge, trajectory-split, floored-standardized) on held-out gLV
trajectories. Two regularizer regimes.

**Induce-collapse regime** (weak variance-reg: sim=4, cov=1, std=0.25 — the Sobal-style setting). This
is the headline. Figure: [results/ablation_collapse.png](results/ablation_collapse.png).

| probe (R², held-out)            | IDM on        | IDM off       | Δ      |
|---------------------------------|---------------|---------------|--------|
| fast: **action** (intervention) | 0.748 ± 0.051 | 0.520 ± 0.021 | **+0.229** |
| fast: Δstate (dynamics)         | 0.819 ± 0.023 | 0.736 ± 0.012 | +0.082 |
| fast: state                     | 0.974 ± 0.003 | 0.963 ± 0.002 | +0.011 |
| slow: init (identity)           | 0.993 ± 0.001 | 0.989 ± 0.003 | +0.005 |

IDM on > off on intervention-decodability in **all 3 seeds** (+0.254 / +0.297 / +0.133), error bars
non-overlapping. Without IDM the community representation discards ~30% of the recoverable intervention
signal (0.52 vs 0.75); IDM forces it to retain the applied intervention. Slow identity is saturated
(~0.99) for both arms, so IDM specifically rescues the FAST (intervention + dynamics) signal.

**Default regime** (standard VICReg: sim=1, cov=25, std=1).
Figure: [results/ablation_default.png](results/ablation_default.png).
fast:action 0.364 ± 0.020 (on) vs 0.291 ± 0.041 (off), **Δ +0.073** (positive in 2/3 seeds; 1 seed
reversed); Δstate +0.025; state +0.007; slow saturated.

**The result is the regime-dependence itself.** Strong variance/covariance regularization *partially
substitutes* for IDM (small, seed-noisy gap); in a collapse-prone regime the JEPA genuinely collapses
onto slow features and drops the intervention, and IDM robustly recovers it. That is exactly the
"collapse we fought," and it isolates *what the IDM term does and when it matters*. (Honest: we do
**not** overclaim the default regime — its effect is modest and one seed reverses; the figure keeps
the error bars.)

## What we learned about JEPAs (the collapse we fought)
- A predict-in-latent objective on temporal data really does drift toward slow features: the
  intervention is the first thing dropped. A *fresh* action-decodability probe on the frozen encoder is
  a clean, direct collapse diagnostic (a random-init encoder scores ≈0; MEASURED).
- The variance/covariance (VICReg) terms keep the representation from *magnitude* collapse but do **not**
  by themselves force it to encode the action — that is specifically the IDM term's job. The two are
  complementary, which the ablation isolates.
- Practical: a tiny set-transformer is launch-overhead-bound on a GB200 (the autoregressive unroll is a
  Python loop of small kernels); shrinking the model + dropping a per-step diagnostic encode mattered
  more than raw GPU FLOPs.

## Layer A downstream (rigor on real data) — IN PROGRESS
Linear/MLP probe of the frozen community embedding on Susagi tasks (infant environment, IBS
cross-country) vs the Susagi MLP baseline (faithful port, same CV), **plus a sequencing-technology
invariance probe** (Cell-JEPA's argument: a good representation carries *less* technical nuisance — a
tech classifier should do *worse* on our rep than on the Susagi imposter rep). Code:
`probe_downstream.py`, `baselines_port.py` (synthetic-smoke green; real-data joins being wired on the
cluster). **[PENDING real-data numbers.]**

## Planning (Layer B application) — MEASURED, honest NEGATIVE (job 74718; 3 seeds; 12 episodes/seed)
We plan interventions to drive a community to a target attractor via latent-space MPPI (roll the GRU
predictor forward, minimize L2 to the target latent), in MPC, vs random / greedy (true-state 1-step) /
final-only-cost baselines. Figure: [results/planning_success_rate.png](results/planning_success_rate.png).

| method | success rate | mean final dist (start 6.64, tol 1.00) |
|---|---|---|
| random | 0.000 | 4.58 |
| greedy | 0.000 | 4.51 |
| final_only | 0.000 | 4.87 |
| **mppi (ours)** | **0.000** | **4.88** |

**No method reaches the target (0% all four), and MPPI does NOT beat the baselines.** All methods reduce
distance 6.64→~4.5. Honest read (causes not yet disentangled — future work): (i) the cost is latent-L2
while success is measured in TRUE abundance space — if the encoder's latent geometry doesn't align with
state geometry, minimizing latent distance need not reach the target state; (ii) bounded K=6-taxon panel
actions may be too weak to cross basins in 20 MPC steps; (iii) rollout error compounds over the horizon.
We report this as-is; the headline (IDM ablation) stands independently of planning success.

## Reproducibility
- One command (GPU): `cd $WORK/eb_jepa && sbatch examples/microbiome_jepa/run_glv_final.sh`
  (3 seeds × {default, collapse} → JSON of every number + the figure).
- Local CPU smoke (no data, minutes): `.venv-cpu/bin/python -m examples.microbiome_jepa.run_ablation
  --seeds 0 --epochs 6 --n_traj 64 --d_model 64`.
- Layer A: `python -m examples.microbiome_jepa.main --fname .../cfgs/layerA_vicreg.yaml`.

## Honest limitations
- The headline is on **synthetic gLV** (by design: clean, controllable). Real DIABIMMUNE trajectories
  are the reality-check, not the rigorous ablation.
- The set-transformer uses CLR log-abundance + a (for gLV) fixed random "species embedding"; real-data
  runs use ProkBERT embeddings.
- 3-seed (not large-N) error bars; we report mean ± s.e. and the seeds.
- Planning is complete (an honest negative, above); the real-data Layer A downstream probe is running
  at time of writing (the corpus-pretrained encoder is required — the synthetic-smoke encoder is not a
  valid probe).
