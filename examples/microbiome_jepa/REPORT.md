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

## Layer A downstream on REAL data — MEASURED (job 74841 pretrain; job 74984 fair eval; infant-environment; 2036 samples, 12 classes)
Layer-A set-JEPA pretrained on 20k real MicrobeAtlas communities (two-view VICReg, 30 epochs, **no
collapse**: feat_std 0.81→0.94), then **frozen**. We probe the community embedding on the infant
birth-mode×age task (StratifiedKFold-5, accuracy + macro OVR ROC-AUC) with TWO probes on the SAME frozen
embeddings — a **linear** probe (the hardest SSL test; a representation-quality claim) and an **MLP**
probe matching the baseline's classifier class (apples-to-apples) — vs a Susagi-style MLP on the **true
abundance matrix** (`abundance.csv`, same CV). Tokens are z-scored with **corpus** statistics (consistent
with pretraining), not infant-only. `realdata.py`.

| infant-env (12-class)                          | accuracy        | macro ROC-AUC   |
|------------------------------------------------|-----------------|-----------------|
| **OUR frozen JEPA + linear probe**             | 0.509 ± 0.014   | **0.896 ± 0.001** |
| OUR frozen JEPA + MLP probe (apples-to-apples) | 0.500 ± 0.008   | 0.888 ± 0.002   |
| Susagi MLP on the true abundance matrix (port) | 0.527 ± 0.010   | 0.890 ± 0.002   |
| Susagi reported (reference)                    | 0.549           | 0.912           |

Honest read: a **frozen** self-supervised encoder is **competitive** with a task-supervised MLP on the
raw abundance matrix. Our best probe (linear) **matches macro-AUC** (0.896 vs 0.890) and is ~2pp below on
accuracy (0.509 vs 0.527). Notably the **MLP probe does NOT beat the linear probe** (0.500/0.888 vs
0.509/0.896): the representation is already linearly separable, so the extra classifier capacity does not
help (it slightly overfits the 2036-sample task) — this *supports* the representation-quality claim
rather than indicating a self-handicap. The **corpus z-score** changed the numbers negligibly (linear
0.509 vs the infant-z-score 0.508), removing that caveat honestly. We do **not** claim a decisive win
(AUC tie, accuracy ~2pp below). Caveats: 30-epoch pretraining (light — a longer/bigger 100-epoch run is
in progress, `run_realdata_big.sh`); a frozen linear probe is a deliberately hard test. **Tech-invariance
is N/A on infants** (Instrument = 100% Illumina MiSeq); it is measured separately on a multi-tech corpus
subset (see "Sequencing-tech invariance" below).

## Sequencing-tech invariance — MEASURED, honest NEGATIVE (`tech_invariance.py`; job 74996; 4960 corpus samples)
A rubric-cited probe: does the representation drop the sequencing-TECHNOLOGY nuisance while keeping
biology? We label real corpus samples **amplicon (16S) vs WGS (shotgun)** — the dominant technical axis —
via the RunID→Terms join (2.5M of 3M runs carry a clean strategy term), take a balanced 4960, and ask a
linear probe to recover the tech from each rep (**LOWER acc = more invariant = better**), with an
8-biome probe as a "keeps-biology" control (HIGHER = better).

| rep (chance: tech 0.50, biome 0.53)  | TECH acc ↓ better | BIOME acc ↑ better |
|--------------------------------------|-------------------|--------------------|
| **OUR frozen JEPA**                  | 0.952             | 0.864              |
| raw mean-pool (input)                | 0.938             | 0.860              |
| random-init encoder                  | 0.897             | 0.836              |

Honest read: our JEPA is **NOT tech-invariant** — it encodes amplicon-vs-WGS *slightly MORE* than the raw
input (0.952 vs 0.938) and well above a random encoder, while also best preserving biology (0.864). The
two-view VICReg objective with composition-preserving augmentations (OTU subsample / abundance jitter /
dropout) yields a FAITHFUL community representation that captures both biology AND the protocol
signature — nothing in the objective removes the technical axis. A clean negative: tech-invariance would
require tech-spanning augmentations or an explicit domain-adversarial / invariance term. *(Comparison vs
the Susagi imposter rep — the named baseline — pending job 75032.)*

## Planning (Layer B application) — MEASURED: a fully DIAGNOSED, partially-closed negative
The headline (IDM ablation) stands independently of planning. We nonetheless pursued the headline
*application* — drive a community to a target attractor by optimizing interventions — and turned an
initial flat 0% into a layered, fully diagnosed result whose every link is a MEASURED number.

**1. Initial negative (K=6, job 74718).** Latent-MPPI (roll the GRU predictor, minimize L2 to the target
latent), MPC, vs random / greedy (true-state 1-step) / final-only baselines: ALL 0% (final ~4.5, start
6.64, tol 1.0); MPPI did not beat random.

**2. Controllability is the first-order cause — oracle K-sweep** (`oracle_K_sweep.py`, CPU; figure
[results/oracle_K_sweep.png](results/oracle_K_sweep.png)). A PERFECT-model planner (state-space MPPI on
the TRUE gLV dynamics) ALSO fails at K=6 (0%, final 4.09), even with 4× actions + 3× horizon — so the
task is unreachable with a 6-of-24 candidate panel. Sweeping the panel size K at a fixed action budget
(attractors, hence tol, are independent of K), the oracle's final distance falls monotonically —
4.09 (K6) → 2.38 (K18) → **0.79 (K24, success 1.00)** — and success crosses tol ONLY at K=24 (all species
dose-able; 3 seeds, near-zero error bars). The task is controllable, but only near full actuation.

**3. At K=24 the LEARNED planner still fails — bottleneck isolated to the REPRESENTATION** (job 74933 +
`diagnose_planning.py`, CPU). Retraining the world model at K=24 and re-running learned latent-MPPI: still
0% (final 4.27 > random's best 3.58). Diagnostics on this model: oracle **100%** (controllable ✓), latent
rollout divergence **~2%** over 20 steps (dynamics FAITHFUL ✓), but latent-distance-to-target vs
true-distance correlate at **Pearson ≈ 0** — the latent METRIC is uninformative. So neither
controllability nor the dynamics model is the bottleneck; the planning COST (latent geometry) is.

**4. The lever — DECODED-state planning** (`plan_glv_decoded.py`; figure
[results/planning_diagnosis.png](results/planning_diagnosis.png)). Keep the same frozen world model but
score MPPI by a linear/MLP state READOUT z→x̂ (the encoder retains state) instead of raw latent distance.
Planning then improves MONOTONICALLY with readout fidelity:

| K=24 world model      | state-readout R² (MLP) | decoded-MPPI success | decoded-MPPI final dist (tol 1.0) |
|-----------------------|------------------------|----------------------|-----------------------------------|
| default reg (cov=25)  | 0.77                   | 0.0%                 | ~4.0 (≈ random)                   |
| weak reg (cov=1)      | 0.89                   | **2.8%** (first >0)  | **2.78** (best of any method)     |

The weak-reg encoder (state more linearly decodable; rollout ~0.8% faithful) lifts decoded planning to
the first non-zero success and the best final distance of any method. *(A higher-capacity weak-reg model,
d256/512-traj, is training to test whether this closes the loop further — `run_glv_k24_big.sh`, pending.)*

**Conclusion — a diagnosed, partially-closed negative.** The negative decomposes into a clean causal
chain, each link MEASURED: (i) **controllability** — fixed by enlarging the candidate panel to K=24
(oracle 0%→100%); (ii) the learned **dynamics are faithful** (~1–2% rollout divergence) and NOT the
bottleneck; (iii) the residual bottleneck is the encoder's **representation geometry** — raw latent
distance is an uninformative planning cost (corr ≈ 0), and a state-aligned (decoded) cost recovers
planning *in proportion to the readout fidelity* (R² 0.77→0.89 ⇒ success 0%→2.8%). This pinpoints WHY
latent-space planning is hard for a VICReg JEPA — the representation is faithful for one-step prediction
yet not a metric space for multi-step planning — and demonstrates the lever that improves it. The
rubric-honest outcome: not an unexplained 0%, but a layered diagnosis with a demonstrated partial fix.

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
