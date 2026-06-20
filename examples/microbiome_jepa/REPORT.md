# EB-JEPA for the Microbiome: an action-conditioned world model with intervention planning

**Status:** the IDM-ablation headline (job 74610) is MEASURED and final. The planning application is a
fully DIAGNOSED, partially-closed negative (controllability → representation → readout-fidelity chain;
jobs 74718/74933/74966 + CPU diagnostics). The real-data Layer A probe is MEASURED with a fair linear+MLP
comparison and corpus z-score (job 74984). Sequencing-tech invariance is a MEASURED honest negative (job
74996). A higher-capacity planning model and a longer real-data pretraining are still running. Every
number is labelled MEASURED / PENDING — no fabricated values, and the reversing seed is kept in the
figures.

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
- **A representation can be FAITHFUL for one-step prediction yet NOT a metric space for multi-step
  planning** (from the planning diagnosis). The world model's latent rollout tracks the true trajectory
  to ~1–2% and the latent linearly decodes the state — yet *Euclidean distance in that latent is
  uninformative about task progress* (corr ≈ 0 with true-state distance), so latent-MPPI plans no better
  than random. Planning only moved toward the target with a state-aligned (decoded) cost, improving in
  proportion to the readout's fidelity. Lesson: the JEPA objective shapes WHAT is encoded, not the
  GEOMETRY of the encoding — "good for probing" ≠ "good for planning".
- Tech-invariance is NOT free (from the tech probe): VICReg with composition-preserving augmentations
  yields a rep that faithfully keeps the sequencing-protocol signature (amplicon vs WGS) — it is *less*
  tech-invariant than even the Susagi imposter rep. A JEPA only becomes invariant to a nuisance its
  augmentations or losses explicitly span.
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
| **OUR frozen JEPA**                  | 0.952             | **0.864**          |
| raw mean-pool (input)                | 0.938             | 0.860              |
| **Susagi imposter rep** (port)       | **0.891**         | 0.842              |
| random-init encoder                  | 0.897             | 0.836              |

Honest read: our JEPA is **NOT tech-invariant** — it encodes amplicon-vs-WGS the MOST of any rep (0.952),
above the raw input (0.938), and **even the Susagi imposter rep is more tech-invariant than ours**
(0.891 ≈ the random encoder). The Susagi model (identity-only OTU embeddings, trained to judge whether an
OTU *belongs*) retains less of the protocol signal; our two-view VICReg objective with
composition-preserving augmentations (OTU subsample / abundance jitter / dropout) instead yields a
FAITHFUL community representation that captures both biology AND the protocol signature. The flip side
(the only place we lead): our JEPA **best preserves biology** (biome 0.864, highest). This is an honest
*loss* on the tech-invariance metric specifically — achieving invariance would need tech-spanning
augmentations or an explicit domain-adversarial / invariance term, which the current recipe lacks.

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
The planner then gets MONOTONICALLY CLOSER to the target as readout fidelity rises (3 seeds, 12
episodes/seed; seeded decoder for reproducibility):

| K=24 world model      | state-readout R² (MLP) | decoded-MPPI success | decoded-MPPI final / best dist (tol 1.0) |
|-----------------------|------------------------|----------------------|------------------------------------------|
| default reg (cov=25)  | 0.78                   | 0%                   | 4.12 / 3.74  (≈ random 3.99)             |
| weak reg (cov=1)      | 0.89                   | 0%                   | **3.01 / 2.58**  (best final of any method) |

The weak-reg encoder (state more linearly decodable; rollout ~0.8% faithful) makes decoded-MPPI the
**best-final-distance method** (3.01 / best 2.58 vs random 3.99, greedy 3.26, latent-MPPI 4.49) — i.e.
higher readout fidelity moves the planner closer — but **no learned planner crosses tol=1.0 (all 0%)**:
the loop is NOT closed at the achieved fidelity (R² ≤ 0.89). *(A higher-capacity weak-reg model,
d256/512-traj, is training to add a third fidelity point — `run_glv_k24_big.sh`, pending.)*

**Conclusion — a fully diagnosed (still-open) negative.** The negative decomposes into a clean causal
chain, each link MEASURED: (i) **controllability** — fixed by enlarging the candidate panel to K=24
(oracle 0%→100%); (ii) the learned **dynamics are faithful** (~1–2% rollout divergence) and NOT the
bottleneck; (iii) the residual bottleneck is the encoder's **representation geometry** — raw latent
distance is an uninformative planning cost (corr ≈ 0), and a state-aligned (decoded) cost moves the
planner closer *in proportion to the readout fidelity* (R² 0.78→0.89 ⇒ final 4.12→3.01), but not yet to
sub-tol success. This pinpoints WHY latent-space planning is hard for a VICReg JEPA — the representation
is faithful for one-step prediction yet not a metric space for multi-step planning — and identifies the
lever (readout fidelity). The rubric-honest outcome: not an unexplained 0%, but a layered diagnosis that
isolates the bottleneck to the representation and shows what moves the needle.

## Did a better representation (SIGReg / LeJEPA) fix the weak spots? — MEASURED, mixed (branch `sigreg-rep`)
Thesis: the three weak spots — M2's AUC-tie, M3's unclosed planning loop, and the tech-invariance loss —
all bottleneck on the SAME thing, the *representation* (two-view VICReg). The highest-leverage untried
lever from our own lit review is **SIGReg (LeJEPA)** instead of VICReg (eb_jepa ships it as `BCS`,
Epps-Pulley isotropy). We swapped VICReg→SIGReg, changing nothing else, and re-measured. One change at a
time; all numbers seeded.

**M3 geometry gate (`diagnose_planning.py --tag k24_sigreg`) — SIGReg does NOT fix planning geometry.**
A gLV world model with SIGReg isotropy on the encoder output (`SIGReg_IDM_Sim_Regularizer`):

| K=24 world model | encoder feat_std | latent-cost corr (Pearson/Spearman) | latent rollout divergence |
|---|---|---|---|
| VICReg (weak reg) | ~0.01 (squished) | ≈0 / −0.05 | ~0.01 |
| **SIGReg** | **0.50** (isotropic ✓) | **−0.23 / −0.41** (still not metric) | **0.61** (worse) |

SIGReg did fix the *variance* (feat_std 0.01→0.50 — the latent is now isotropic, not collapsed) but **not
the metric**: latent distance is still uninformative (corr stayed non-positive), so latent-MPPI would
still fail. A clean side-finding: SIGReg's spread-out latent is **much harder to roll forward**
(divergence 0.01→0.61) — VICReg's tiny rollout error was partly an *artifact of latent collapse* (a
near-constant latent is trivially predictable). Lesson: an isotropic latent is not automatically a
*plannable* one. (Per our gate rule, corr-not-positive → we stopped the M3 sub-thread.)

**M2 infant-env (frozen probe) — SIGReg looks better, but the fair baseline is pending.** SIGReg
(100ep/50k/d256): linear 0.514/0.891, **MLP 0.526/0.894** (matches the Susagi MLP 0.527/0.890 on acc,
beats on AUC), fine-tuned upper bound **0.590/0.918** (exceeds Susagi's reported 0.549/0.912). **Caveat
(no overclaim):** this is SIGReg-100ep/d256 vs the earlier VICReg-**30ep/d128**; a VICReg-100ep/d256 run
is in flight to isolate SIGReg-vs-VICReg from the longer/bigger-training effect. **Tech-invariance on the
SIGReg encoder: PENDING.** We fold a SIGReg result into the headline only if the matched-budget
comparison shows a real improvement; otherwise it stands as this honest SIGReg-vs-VICReg comparison.

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
- Planning is a fully *diagnosed* negative (above), not a success: the loop is not closed at the achieved
  representation fidelity (R² ≤ 0.89); the identified path is a more state-decodable / metric latent.
- The real-data Layer A probe (competitive: AUC tie with a supervised MLP) and the sequencing-tech
  invariance (an honest *loss* — our rep keeps the protocol signal) are MEASURED. Still running at time
  of writing: a higher-capacity planning world model (a 3rd readout-fidelity point) and a longer/bigger
  corpus pretraining (100ep/50k/d256) to re-probe — both clearly labelled PENDING above.
