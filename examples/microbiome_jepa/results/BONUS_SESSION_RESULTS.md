# Bonus-experiment session — consolidated results (2026-06-20)

All numbers below are **MEASURED** from real runs this session (GB200 training + local-CPU eval), seeded
for reproducibility. `bnz` (the shippable submission) was **NOT modified**: it stays at `19387f5`. Each
experiment is on its own branch; per-experiment data + tables are in this `results/` dir on that branch.

Substrate everywhere: K=full-actuation, d128 (unless swept), weak-reg (sim4/cov1/std0.25), idm on,
80 epochs. Planning eval = `plan_glv_learned.py`, 3 seeds × 12 episodes, tol = 0.15 × mean
inter-attractor distance. `mppi_latent` = raw latent-distance MPPI (no decoder/learned cost) — the clean
closure test. Oracle = true-dynamics MPPI (controllability reference).

---

## PRIORITY 0 — metric_coeff tradeoff (branch `m3-metric-loss-hybrid`, commit 283ca9e)
Files: `metric_sweep_consolidated.{json,md}`, `m3_recognition_mc10.json`, `m3_recognition_mc30.json`.

| metric_coeff | SUCCESS (mppi_latent) | final | free-run 6-step rollout | latent↔true Spearman | recog guild | recog basin |
|---|---|---|---|---|---|---|
| pure JEPA (mc=0) | 0.0% | 4.531 | 0.084 | +0.085 | 0.899 | 0.690 |
| HYBRID mc=0.3 | **100.0% ± 0.0** | 0.804 | 0.283 | +0.990 | 0.971 | 0.812 |
| HYBRID mc=1.0 | 97.2% ± 2.8 | 0.840 | 0.365 | +0.992 | 0.967 | 0.771 |
| HYBRID mc=3.0 | 91.7% ± 4.8 | 0.866 | 0.415 | +0.991 | 0.970 | 0.779 |
| ORACLE (ref) | 100% | 0.790 | — | — | — | — |

- **mc=0.3 100% is the FULL eval** (36 episodes = 3 seeds × 12, tol=0.9958, 20 MPC steps) — not a smoke run.
- **Higher metric_coeff erodes planning SUCCESS** (100→97.2→91.7%), not only rollout fidelity. The cost
  geometry SATURATES (Spearman ~0.99 for all mc>0), so success tracks the rising free-running rollout
  error (0.283→0.415) — degraded **predictability**, not cost geometry. mc=0.3 is the sweet spot on all
  three axes. (recog mc=1.0/3.0 measured here from the local checkpoints; matches the report's cited 0.771.)

## EXP 1 — generalization across gLV instances (branch `m3-generalization`) — **POSITIVE**
Files: `exp1_generalization.{json,md}`, `exp1_instance_screen.json`. HYBRID mc=0.3 trained per instance;
matched oracle-vs-learned eval on the same episodes/tol.

> First question answered: the headline closure's 3 "seeds" vary ONLY the planning eval (episode + MPPI
> sampling) on ONE fixed gLV system — the gLV interaction matrix A + attractors are deterministic in the
> structural config (the seed only seeds trajectory noise). So new instances require varying STRUCTURAL
> knobs. Honest deviation from "random A": A is engineered deterministically (stability-guaranteed), so I
> varied n_guilds / competition strengths / species count — each yields a genuinely different A +
> attractors, all stability- + controllability-verified on CPU before training.

| instance | guilds/S/K | tol | oracle succ/final | mppi_latent succ/final | crosses tol |
|---|---|---|---|---|---|
| g3_s18 | 3/18/18 | 0.862 | 100% / 0.61 | **100% / 0.74** | yes (= oracle) |
| g5_s30 | 5/30/30 | 0.861 | 100% / 0.67 | **100% / 0.65** | yes (= oracle) |
| g4_s24 | 4/24/24 | 0.861 | 100% / 0.65 | 89% ± 7 / 0.78 | yes |
| g3_s24_strongcomp | 3/24/24 | 0.988 | 100% / 0.79 | 89% ± 7 / 0.95 | yes |

**The closure is NOT cherry-picked.** All 4 new instances cross tol (pure JEPA = 0%); 2/4 match the oracle
at 100%, the other 2 reach 89% (32/36 episodes) with final distance close to oracle. Holds across guild
counts {3,4,5}, sizes {18,24,30}, and stronger competition.

## EXP 2 — IDM-reweight self-supervised closure (branch `m3-idm-selfsup`) — **NEGATIVE (sharpens claim)**
Files: `exp2_idm.{json,md}`. Pure JEPA (metric_coeff=0); IDM uses only (z_t,z_{t+1})→action (no true-state
distance — fully self-supervised).

| variant | mppi_latent | latent↔true Spearman | rollout6 | recog basin | learned-cost head Spearman |
|---|---|---|---|---|---|
| pure-JEPA idm=1.0 (ref) | 0% / 4.53 | +0.085 | 0.084 | 0.690 | — |
| HYBRID mc=0.3 (upper bar) | 100% / 0.80 | +0.990 | 0.283 | 0.812 | — |
| idm=2 | 0% / 4.33 | −0.017 | 0.084 | 0.731 | 0.53 |
| idm=5 | 0% / 4.26 | −0.196 | 0.096 | 0.725 | 0.73 |
| idm=10 | 0% / 4.38 | −0.313 | 0.086 | 0.696 | 0.74 |

Strengthening IDM **does not** close planning — raw-latent MPPI stays **0% at every weight**, and the raw
latent↔true Spearman gets *worse* (more negative). The learned-cost head Spearman *does* climb (0.53→0.74):
IDM induces a **control metric**, but it does NOT transfer to the raw Euclidean latent the tol is defined
on, and 0% even with a learned cost on it. **The privileged true-state metric is necessary; self-supervised
IDM cannot substitute.** (Remaining gap: raw latent Spearman ≤0 vs the hybrid's +0.99.)

## EXP 3 — bottleneck shrink (branch `m3-bottleneck`) — **NEGATIVE**
Files: `exp3_dim.{json,md}`. Pure JEPA (metric_coeff=0); latent dim swept toward the true-state dim S=24.

| variant | d | mppi_latent | latent↔true Spearman | rollout6 | recog guild/basin | decode R² |
|---|---|---|---|---|---|---|
| pure-JEPA idm=1.0 (ref) | 128 | 0% / 4.53 | +0.085 | 0.084 | 0.899/0.690 | — |
| HYBRID mc=0.3 (upper bar) | 128 | 100% / 0.80 | +0.990 | 0.283 | 0.971/0.812 | — |
| d=16 | 16 | 0% / 4.82 | +0.254 | 0.041 | 0.672/0.575 | 0.66 |
| d=24 | 24 | 0% / 5.09 | +0.197 | 0.050 | 0.676/0.533 | 0.72 |
| d=32 | 32 | 0% / 4.23 | +0.237 | 0.057 | 0.747/0.581 | 0.78 |

Shrinking the latent toward the true-state dim **does not** close planning (0% at every dim). It gives only
a *marginal* metric nudge (Spearman 0.085→~0.2, far from the 0.99 needed), **costs recognition** (guild
0.90→0.67–0.75, basin 0.69→0.53–0.58) and decodability, while rollout improves (not the binding gap).

---

## One-paragraph synthesis
The headline M3 closure (a true-state isometry auxiliary makes raw latent-MPPI plan at oracle level)
**generalizes across diverse gLV systems** (Exp 1, positive: 4/4 cross tol, 2/4 = oracle), and the
metric_coeff tradeoff is now fully mapped (Priority 0: success 100→92% as the metric weight rises, tracking
rollout predictability, not cost geometry). Two attempts to get the *same* closure **without** the
privileged metric both fail honestly: strengthening the self-supervised IDM (Exp 2) induces a *control*
metric (learned-cost Spearman 0.74) but not the Euclidean state metric — 0% planning, and it degrades the
raw latent metric; shrinking the latent dimension toward the true-state dim (Exp 3) gives only a marginal
metric nudge and 0% planning while costing recognition. Net: the closure is robust and not toy-specific,
and the privileged true-state metric is **necessary** — neither IDM reweighting nor dimensional
bottlenecking is a self-supervised substitute. `bnz` remains untouched and shippable.
