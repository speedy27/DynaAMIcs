"""
HYBRID M3 — GATE for the metric-preserving (isometry) gLV encoder. CPU-only, ~minutes.

This is the CHEAP GATE the experiment runs FIRST: did baking a metric-preserving auxiliary into the
gLV encoder (train_worldmodel with model.regularizer.metric_coeff>0; HYBRID — uses TRUE-state
supervision, NOT pure JEPA) actually put the true metric into the latent? The M3 diagnosis pinned the
planning wall to the latent's METRIC precision (weak-reg: latent-vs-true distance corr ~0, decode
R^2 ~0.89), not the dynamics. So we measure, on the trained encoder:

  (A) latent-distance vs TRUE-state-distance correlation (Pearson+Spearman), two ways:
        - to-TARGET (the planning cost: ||z - z_tgt|| vs ||x - tgt|| over a random walk; the diagnose_
          planning DIAG2 number, ~ -0.06/-0.10 before),
        - general PAIRWISE (any two states; matches what the isometry loss is trained on).
  (B) decode R^2 (linear Ridge + MLP readout z->x; was ~0.84/0.89 before) and feat_std.
  (C) KNOWN RISK: a more metric/spread latent can be harder to roll forward (SIGReg's isotropic latent
      worsened free-running rollout to 0.61). So we ALSO measure the model's own (jointly-trained)
      predictor free-running k-step rollout error + 1-step error.

Pass --ref_checkpoint (the pure-JEPA weak-reg substrate) to recompute the SAME gate on it in this
process => a fully apples-to-apples before/after with NO hardcoded references.

GATE verdict: if (A) and (B) rise clearly on the metric model => the metric is now in the latent =>
proceed to re-plan (plan_glv_learned.py) + recognition (m3_recognition.py). If they do not move =>
the encoder cannot be made metric without breaking other things; record and stop.

INTEGRITY: every number printed is measured from this run; the metric model is HYBRID (true-state
supervision) and is labeled as such; seeded throughout. No fabrication.

Run (CPU):
  .venv-cpu/bin/python -m examples.microbiome_jepa.m3_metric_gate \
     --checkpoint checkpoints/plan_model_k24_metric/latest.pth.tar \
     --ref_checkpoint checkpoints/plan_model_k24_lowreg/latest.pth.tar --device cpu --k 6 \
     --overrides '{"data.n_candidate":24,"model.d_model":128,"model.regularizer.sim_coeff_t":4,"model.regularizer.cov_coeff":1,"model.regularizer.std_coeff":0.25}'
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from eb_jepa.logging import get_logger
from examples.microbiome_jepa.plan_glv import build_glv_and_encoder, build_world_model
from examples.microbiome_jepa.plan_glv_decoded import _encode_states, fit_decoders
from examples.microbiome_jepa.m3_multistep import encode_sequences, free_running_error, one_step_error

logger = get_logger(__name__)


@torch.no_grad()
def _corr_to_target(jepa, state_enc, sim, mpc_steps=20, seed=0):
    """DIAG2-style: latent-dist-to-target vs true-dist-to-target over a random walk (the planning cost)."""
    attractors = sim.attractors
    n_attr = len(attractors)
    action_max = float(sim.config.action_max)
    K = int(sim.action_dim)
    rng = np.random.default_rng(seed)
    lat_d, true_d, zs = [], [], []
    for (src, tgt) in permutations(range(n_attr), 2):
        target = attractors[tgt]
        z_tgt = state_enc.encode(jepa, target).flatten(1)
        x = sim.reset(attractor=src).astype(np.float32)
        for _ in range(mpc_steps):
            z = state_enc.encode(jepa, x).flatten(1)
            zs.append(z[0])
            lat_d.append(float(torch.linalg.norm(z - z_tgt)))
            true_d.append(float(np.linalg.norm(x - target)))
            a = rng.uniform(0.0, action_max, size=K).astype(np.float32)
            x = sim.step(a).astype(np.float32)
    lat_d, true_d = np.array(lat_d), np.array(true_d)
    feat_std = float(torch.stack(zs).std(dim=0).mean())
    return (float(pearsonr(lat_d, true_d)[0]), float(spearmanr(lat_d, true_d)[0]), feat_std, len(lat_d))


@torch.no_grad()
def _corr_pairwise(jepa, state_enc, sim, n_traj=128, T=24, n_pairs=4000, seed=1):
    """General PAIRWISE: ||z_a-z_b|| vs ||x_a-x_b|| over random state pairs (what the isometry loss sees)."""
    data = sim.generate_trajectories(n=n_traj, T=T, action_policy="random", seed=seed)
    X = data["states"].reshape(-1, sim.n_species).astype(np.float32)
    Z = _encode_states(jepa, state_enc, X)
    rng = np.random.default_rng(seed + 7)
    M = len(X)
    i = rng.integers(0, M, n_pairs); j = rng.integers(0, M, n_pairs)
    j = np.where(i == j, (j + 1) % M, j)
    dz = np.linalg.norm(Z[i] - Z[j], axis=-1)
    dx = np.linalg.norm(X[i] - X[j], axis=-1)
    return float(pearsonr(dz, dx)[0]), float(spearmanr(dz, dx)[0])


def gate_one(tag, fname, checkpoint, dev, overrides, k):
    """Compute the full gate (corr to-target + pairwise, decode R^2, feat_std, rollout err) for one model."""
    jepa, cfg, K = build_world_model(fname, checkpoint, dev, overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)

    pear_t, spear_t, feat_std, n_t = _corr_to_target(jepa, state_enc, sim)
    pear_p, spear_p = _corr_pairwise(jepa, state_enc, sim, T=int(cfg.data.T))
    decoders = fit_decoders(jepa, state_enc, sim, T=int(cfg.data.T), device=dev)
    r2_lin, r2_mlp = decoders["linear"][1], decoders["mlp"][1]

    # the model's OWN (jointly-trained) predictor: free-running k-step + 1-step rollout error (held-out)
    Zho, Aho = encode_sequences(sim, state_enc, jepa, 48, int(cfg.data.T), seed=999)
    err_1s = one_step_error(jepa.predictor, Zho, Aho)
    err_fr = free_running_error(jepa.predictor, Zho, Aho, k)

    res = {"tag": tag, "checkpoint": checkpoint,
           "corr_to_target": {"pearson": pear_t, "spearman": spear_t, "n": n_t},
           "corr_pairwise": {"pearson": pear_p, "spearman": spear_p},
           "decode_r2": {"linear": float(r2_lin), "mlp": float(r2_mlp)},
           "feat_std": feat_std,
           "rollout": {"one_step_err": err_1s, f"freerun_{k}step_err": err_fr}}
    print(f"\n---- GATE [{tag}] ----  (checkpoint={checkpoint})")
    print(f"  corr latent-vs-true  to-TARGET : pearson {pear_t:+.3f}  spearman {spear_t:+.3f}   (n={n_t})")
    print(f"  corr latent-vs-true  PAIRWISE  : pearson {pear_p:+.3f}  spearman {spear_p:+.3f}")
    print(f"  decode R^2  linear {r2_lin:.3f}  mlp {r2_mlp:.3f}   feat_std {feat_std:.4f}")
    print(f"  predictor rollout (held-out): 1-step {err_1s:.4f}   free-run {k}-step {err_fr:.4f}")
    return res


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: Optional[str] = None,        # the HYBRID metric-loss model
    ref_checkpoint: Optional[str] = None,    # the pure-JEPA weak-reg substrate (before)
    k: int = 6,
    device: str = "cpu",
    out: str = "examples/microbiome_jepa/results",
    overrides: Optional[dict] = None,
    ref_overrides: Optional[dict] = None,
):
    dev = torch.device(device)
    if ref_overrides is None:
        ref_overrides = overrides
    print("================ HYBRID METRIC-LOSS GATE (latent metric + rollout) ================")
    print("NOTE: the metric model is HYBRID (true-state supervision in training), NOT pure JEPA.")

    out_d = {}
    if ref_checkpoint is not None:
        out_d["pure_jepa_ref"] = gate_one("pure-JEPA (weak-reg, before)", fname, ref_checkpoint, dev,
                                          ref_overrides, k)
    out_d["metric_hybrid"] = gate_one("metric HYBRID (after)", fname, checkpoint, dev, overrides, k)

    # verdict: did the metric land in the latent? (corr + decode both clearly up vs the pure-JEPA ref)
    m = out_d["metric_hybrid"]
    verdict = "metric model trained; "
    if "pure_jepa_ref" in out_d:
        r = out_d["pure_jepa_ref"]
        up_corr = m["corr_to_target"]["spearman"] - r["corr_to_target"]["spearman"]
        up_r2 = m["decode_r2"]["mlp"] - r["decode_r2"]["mlp"]
        roll_worse = m["rollout"][f"freerun_{k}step_err"] - r["rollout"][f"freerun_{k}step_err"]
        gate_pass = (m["corr_to_target"]["spearman"] > 0.3) and (up_corr > 0.2)
        verdict += (f"to-target Spearman {r['corr_to_target']['spearman']:+.3f} -> "
                    f"{m['corr_to_target']['spearman']:+.3f} (Δ{up_corr:+.3f}); decode R^2(mlp) "
                    f"{r['decode_r2']['mlp']:.3f} -> {m['decode_r2']['mlp']:.3f} (Δ{up_r2:+.3f}); "
                    f"free-run {k}-step rollout {r['rollout'][f'freerun_{k}step_err']:.3f} -> "
                    f"{m['rollout'][f'freerun_{k}step_err']:.3f} (Δ{roll_worse:+.3f}). "
                    f"GATE {'PASS -> re-plan' if gate_pass else 'WEAK -> metric did not clearly land'}.")
        out_d["gate_pass"] = bool(gate_pass)
    print(f"\nVERDICT: {verdict}")
    out_d["verdict"] = verdict

    Path(out).mkdir(parents=True, exist_ok=True)
    fn = Path(out) / "m3_metric_gate.json"
    with open(fn, "w") as f:
        json.dump(out_d, f, indent=2)
    print(f"saved -> {fn}")
    return out_d


if __name__ == "__main__":
    fire.Fire(run)
