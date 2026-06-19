# EB-JEPA for the Microbiome: an action-conditioned world model with intervention planning

**Status:** draft. Headline numbers are filled from the 3-seed run (job 74595, in progress); every
number here is labelled MEASURED / PRELIMINARY / PENDING. No fabricated values.

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

## Result — IDM ablation (collapse-and-recovery)
Regime-finding sweep (1 seed, GPU job 74554) located the effect; the headline is the **3-seed
confirmation** with the standardized probe (job 74595), two regularizer regimes:

<!-- FILL FROM job 74595: checkpoints/microbiome_jepa/final_{default,collapse}/ablation_results.json -->
**[PENDING — 3-seed mean ± s.e. to be inserted here on job completion.]**

Preliminary direction (1-seed sweep, pre-standardization probe — superseded by the 3-seed numbers
above): in the default VICReg regime, IDM raised intervention-decodability `fast_r2_action` from
0.128 → 0.241 (~2×) while leaving generic state-decodability flat and *reducing* slow-feature reliance
(`slow_r2_init` 0.539 → 0.500) — i.e. without IDM the world model discarded the intervention and leaned
on slow identity; IDM recovered it. Figure: `final_default/ablation_collapse_recovery.png`.

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

## Planning (Layer B headline application) — PLANNED
Reuse eb_jepa's `MPPIPlanner` + `ReprTargetDistMPCObjective` (latent-distance to a target community)
on a gLV env wrapper; success = reaching a target attractor, vs random / greedy / final-state-only
baselines (the gLV's non-monotonicity is what makes greedy fail). **[PENDING.]**

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
- Downstream real-data probe + planning are in progress at time of writing.
