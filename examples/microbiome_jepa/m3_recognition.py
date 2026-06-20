"""
HYBRID M3 — recognition-vs-planning TRADEOFF probe on the gLV encoder. CPU-only.

The metric-preserving auxiliary (train_worldmodel model.regularizer.metric_coeff>0; HYBRID, uses
TRUE-state supervision) forces the latent geometry to mirror raw abundance-Euclidean geometry. Thesis:
metric-preservation trades away ABSTRACTION, so a RECOGNITION probe on the frozen latent should DEGRADE
vs the pure-JEPA encoder. This is the honest, apples-to-apples way to measure the "M2 impact" — ON THE
gLV ENCODER (the real-data M2 encoder is separate and has no ground-truth state metric, so a real-data
metric run would need a biological community distance, e.g. Bray-Curtis; that is future work, not here).

Two abstract community labels available in the sim (both 3-class for n_guilds=3):
  - dominant guild : argmax of guild-summed abundance of the CURRENT state (categorical community type;
    has within-class Euclidean spread, so a metric latent that spreads same-type states apart can hurt
    linear separability),
  - basin          : which attractor the state relaxes to under NO intervention (topological/dynamical
    label, NOT a simple function of the current metric — the cleanest abstraction test).

Linear probe = logistic regression on frozen latents, fixed seeded split; reported for BOTH encoders
(pure-JEPA weak-reg ref vs metric HYBRID). Also a few-shot point (small train set) where representation
quality matters most. Majority-class baseline + class balance reported for honesty.

INTEGRITY: measured; both encoders frozen; identical data/split/seed; the metric model is HYBRID and
labeled as such. No fabrication.

Run (CPU):
  .venv-cpu/bin/python -m examples.microbiome_jepa.m3_recognition \
     --checkpoint checkpoints/plan_model_k24_metric/latest.pth.tar \
     --ref_checkpoint checkpoints/plan_model_k24_lowreg/latest.pth.tar --device cpu \
     --overrides '{"data.n_candidate":24,"model.d_model":128,"model.regularizer.sim_coeff_t":4,"model.regularizer.cov_coeff":1,"model.regularizer.std_coeff":0.25}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eb_jepa.logging import get_logger
from examples.microbiome_jepa.plan_glv import build_glv_and_encoder, build_world_model
from examples.microbiome_jepa.plan_glv_decoded import _encode_states

logger = get_logger(__name__)


def _relax_batch(sim, X, n_steps=400):
    """Vectorized zero-action gLV relaxation (same dynamics as GLVSimulator._deriv/_rk4_step) for a
    BATCH of states X[B,S]; returns the relaxed states. Used to assign basin-of-attraction labels."""
    A = sim.interaction_matrix          # [S,S]
    r = sim.growth                      # [S]
    m = float(sim.config.immigration)
    dt = float(sim.config.dt)
    x = np.clip(np.asarray(X, dtype=np.float64).copy(), 0.0, 1e6)   # [B,S]

    def deriv(z):
        return z * (r + z @ A.T) + m    # (A @ z_i) for each row i == z @ A.T

    for _ in range(n_steps):
        k1 = deriv(x)
        k2 = deriv(np.maximum(x + 0.5 * dt * k1, 0.0))
        k3 = deriv(np.maximum(x + 0.5 * dt * k2, 0.0))
        k4 = deriv(np.maximum(x + dt * k3, 0.0))
        x = np.clip(x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1e6)
    return x


def _labels(sim, X, do_basin=True, n_basin=2400, seed=0):
    """dominant-guild label (all states) + basin label (subsample, via zero-action relaxation)."""
    guild = sim.guild                                   # [S]
    ng = int(guild.max()) + 1
    # dominant guild = argmax over guilds of summed abundance
    guild_sums = np.stack([X[:, guild == g].sum(1) for g in range(ng)], axis=1)  # [M, ng]
    y_dom = guild_sums.argmax(1)
    out = {"dominant_guild": y_dom}
    if do_basin:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))[:min(n_basin, len(X))]
        relaxed = _relax_batch(sim, X[idx])
        attr = sim.attractors                            # [ng, S]
        d = np.linalg.norm(relaxed[:, None, :] - attr[None, :, :], axis=-1)   # [b, ng]
        out["basin"] = (idx, d.argmin(1))
    return out


def _probe(Z, y, seed=0, frac=0.8, few_n=30):
    """Linear probe (logistic regression on standardized frozen latents). Returns full-train test acc,
    a few-shot test acc (few_n per class), and the majority-class baseline. Seeded split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z))
    cut = int(frac * len(Z))
    tr, te = perm[:cut], perm[cut:]
    sc = StandardScaler().fit(Z[tr])
    Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
    ytr, yte = y[tr], y[te]

    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Ztr, ytr)
    acc_full = float((clf.predict(Zte) == yte).mean())

    # few-shot: few_n examples per class from the train split
    few_idx = []
    for c in np.unique(ytr):
        ci = np.where(ytr == c)[0]
        few_idx.extend(rng.permutation(ci)[:few_n].tolist())
    few_idx = np.array(few_idx)
    clf_few = LogisticRegression(max_iter=3000, C=1.0).fit(Ztr[few_idx], ytr[few_idx])
    acc_few = float((clf_few.predict(Zte) == yte).mean())

    vals, cnts = np.unique(yte, return_counts=True)
    majority = float(cnts.max() / cnts.sum())
    balance = {int(v): int(c) for v, c in zip(vals, cnts)}
    return {"acc_full": acc_full, "acc_fewshot": acc_few, "majority": majority,
            "n_train": int(len(tr)), "n_test": int(len(te)), "test_class_balance": balance}


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: Optional[str] = None,        # HYBRID metric-loss model
    ref_checkpoint: Optional[str] = None,    # pure-JEPA weak-reg substrate
    n_traj: int = 256,
    seed: int = 0,
    device: str = "cpu",
    out: str = "examples/microbiome_jepa/results",
    overrides: Optional[dict] = None,
    ref_overrides: Optional[dict] = None,
):
    dev = torch.device(device)
    if ref_overrides is None:
        ref_overrides = overrides

    # build the metric model + sim (sim/labels are model-independent; reuse one sim)
    jepa_m, cfg, K = build_world_model(fname, checkpoint, dev, overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)

    data = sim.generate_trajectories(n=n_traj, T=int(cfg.data.T), action_policy="random", seed=seed + 5)
    X = data["states"].reshape(-1, sim.n_species).astype(np.float32)
    labs = _labels(sim, X, seed=seed)

    Z_metric = _encode_states(jepa_m, state_enc, X)
    encoders = {"metric_hybrid": Z_metric}
    if ref_checkpoint is not None:
        jepa_r, _, _ = build_world_model(fname, ref_checkpoint, dev, overrides=ref_overrides)
        encoders["pure_jepa_ref"] = _encode_states(jepa_r, state_enc, X)

    print("================ HYBRID METRIC-LOSS: recognition tradeoff (gLV encoder) ================")
    print("NOTE: the metric model is HYBRID (true-state supervision in training), NOT pure JEPA.")
    print(f"n_states={len(X)}  guild classes={int(sim.guild.max())+1}")

    results = {"checkpoint": checkpoint, "ref_checkpoint": ref_checkpoint, "tasks": {}}
    tasks = ["dominant_guild"] + (["basin"] if "basin" in labs else [])
    for task in tasks:
        if task == "basin":
            idx, y = labs["basin"]
        else:
            idx, y = np.arange(len(X)), labs[task]
        print(f"\n---- task: {task}  (n={len(y)}) ----")
        results["tasks"][task] = {}
        for name, Z in encoders.items():
            pr = _probe(Z[idx], y, seed=seed)
            results["tasks"][task][name] = pr
            print(f"  {name:16s}: linear acc {pr['acc_full']:.3f}  few-shot {pr['acc_fewshot']:.3f}  "
                  f"(majority {pr['majority']:.3f}, n_test {pr['n_test']})")
        if "pure_jepa_ref" in encoders:
            d_full = (results["tasks"][task]["metric_hybrid"]["acc_full"]
                      - results["tasks"][task]["pure_jepa_ref"]["acc_full"])
            results["tasks"][task]["delta_metric_minus_pure_full"] = d_full
            print(f"  Δ(metric − pure) linear acc = {d_full:+.3f}  "
                  f"({'recognition COST (tradeoff)' if d_full < -0.005 else 'no degradation' if d_full < 0.005 else 'recognition GAIN'})")

    Path(out).mkdir(parents=True, exist_ok=True)
    fn = Path(out) / "m3_recognition.json"
    with open(fn, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {fn}")
    return results


if __name__ == "__main__":
    fire.Fire(run)
