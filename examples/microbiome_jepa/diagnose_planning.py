"""
Diagnose the M3 planning NEGATIVE: which of three causes is responsible? (CPU-only, ~minutes.)

Same setup as plan_glv (gLV params, attractors, bounded K-panel actions, horizon, tol≈1.0). Three
cheap diagnostics, each isolating one hypothesis:

1. ORACLE / task-solvability: state-space MPPI on the TRUE gLV dynamics (sim._step_from), cost = true
   distance to the target attractor. If a perfect model also ~0% / stalls near the same final distance,
   the bottleneck is the TASK SPEC (action bound / horizon / tolerance), not the learned model.
2. LATENT-COST ALIGNMENT: Pearson + Spearman between latent-distance-to-target (what MPPI minimizes)
   and true-state-distance-to-target (what success measures), over states visited along rollouts. Low
   correlation => the latent cost is a poor proxy => MPPI ≈ random.
3. WORLD-MODEL ROLLOUT ACCURACY: roll the learned latent forward H steps under a fixed action sequence
   (z_model[t]) vs the encoder's latent of the TRUE trajectory under the same actions (z_true[t]).
   Fast normalized divergence => planning on the model is doomed regardless of cost/actions.

Run (local CPU): .venv-cpu/bin/python -m examples.microbiome_jepa.diagnose_planning \
    --checkpoint checkpoints/plan_model/latest.pth.tar
INTEGRITY: every number printed is measured from this run; nothing fabricated.
"""

import json
from itertools import permutations
from pathlib import Path

import fire
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from examples.microbiome_jepa.plan_glv import (
    MPPIConfig,
    build_glv_and_encoder,
    build_world_model,
    rollout_latent,
)


def _attr_scale(attractors):
    dists = [np.linalg.norm(attractors[i] - attractors[j])
             for i, j in permutations(range(len(attractors)), 2)]
    return float(np.mean(dists))


def _oracle_mppi_action(sim, x0, target_state, action_max, K, H, n_samples, n_elites, n_iters,
                        init_std, temperature, rng):
    """State-space MPPI on TRUE gLV dynamics. Returns the planned first action [K]."""
    mean = np.zeros((H, K), dtype=np.float32)
    std = np.full((H, K), init_std, dtype=np.float32)
    for _ in range(n_iters):
        noise = rng.standard_normal((n_samples, H, K)).astype(np.float32)
        actions = np.clip(mean[None] + std[None] * noise, -action_max, action_max)
        costs = np.zeros(n_samples, dtype=np.float64)
        for n in range(n_samples):
            x = x0.copy()
            for t in range(H):
                x = sim._step_from(x, actions[n, t])      # TRUE functional step (perfect model)
                costs[n] += np.linalg.norm(x - target_state)   # cumulative true-state cost
        elite_idx = np.argsort(costs)[:n_elites]
        ec = costs[elite_idx]
        w = np.exp(temperature * (costs.min() - ec))
        w = w / (w.sum() + 1e-9)
        ea = actions[elite_idx]                            # [n_el, H, K]
        mean = (w[:, None, None] * ea).sum(0).astype(np.float32)
        var = (w[:, None, None] * (ea - mean[None]) ** 2).sum(0)
        std = np.sqrt(var).clip(min=0.02).astype(np.float32)
    return np.clip(mean[0], -action_max, action_max)


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: str = "checkpoints/plan_model/latest.pth.tar",
    d_model: int = 128,
    n_candidate: int = None,    # None -> use cfg (K=6, the committed K=6 diagnosis); set 24 for the big bet
    collapse_reg: bool = True,  # True -> collapse-regime coeffs (committed default); False -> cfg defaults
    mpc_steps: int = 20,
    horizon: int = 6,
    n_samples: int = 96,
    n_elites: int = 16,
    n_iters: int = 3,
    tol_frac: float = 0.15,
    out: str = "checkpoints/microbiome_jepa/planning",
    tag: str = "",
):
    dev = torch.device("cpu")
    ov = {"model.d_model": d_model}
    if collapse_reg:
        ov.update({"model.regularizer.sim_coeff_t": 4, "model.regularizer.cov_coeff": 1,
                   "model.regularizer.std_coeff": 0.25})
    if n_candidate is not None:
        ov["data.n_candidate"] = int(n_candidate)
    jepa, cfg, K = build_world_model(fname, checkpoint, dev, overrides=ov)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    attractors = sim.attractors
    n_attr = len(attractors)
    attr_scale = _attr_scale(attractors)
    tol = tol_frac * attr_scale
    action_max = float(sim.config.action_max)
    pairs = list(permutations(range(n_attr), 2))
    rng = np.random.default_rng(0)
    print(f"[diag] n_attr={n_attr} attr_scale={attr_scale:.3f} tol={tol:.3f} action_max={action_max} "
          f"K={K} horizon={horizon} mpc_steps={mpc_steps} pairs={len(pairs)}")

    # ---- DIAG 1: ORACLE (true-dynamics MPPI) ----
    n_succ, finals, starts = 0, [], []
    for (src, tgt) in pairs:
        target = attractors[tgt]
        x = sim.reset(attractor=src).astype(np.float32)
        starts.append(float(np.linalg.norm(x - target)))
        for _ in range(mpc_steps):
            a = _oracle_mppi_action(sim, x, target, action_max, K, horizon, n_samples, n_elites,
                                    n_iters, 0.25, 1.0, rng)
            x = sim.step(a).astype(np.float32)
            if np.linalg.norm(x - target) < tol:
                break
        fd = float(np.linalg.norm(x - target))
        finals.append(fd)
        n_succ += int(fd < tol)
    oracle = {"success_rate": n_succ / len(pairs), "mean_final_dist": float(np.mean(finals)),
              "mean_start_dist": float(np.mean(starts)), "tol": tol, "n_pairs": len(pairs)}
    print(f"[DIAG1 oracle/true-dynamics MPPI] success={oracle['success_rate']:.3f} "
          f"final_dist={oracle['mean_final_dist']:.3f} (start {oracle['mean_start_dist']:.3f}, tol {tol:.3f})")

    # ---- DIAG 2: LATENT-COST ALIGNMENT ----
    lat_d, true_d = [], []
    for (src, tgt) in pairs:
        target = attractors[tgt]
        z_tgt = state_enc.encode(jepa, target).flatten(1)  # [1,D]
        x = sim.reset(attractor=src).astype(np.float32)
        for _ in range(mpc_steps):
            z = state_enc.encode(jepa, x).flatten(1)        # [1,D]
            lat_d.append(float(torch.linalg.norm(z - z_tgt)))
            true_d.append(float(np.linalg.norm(x - target)))
            a = rng.uniform(0.0, action_max, size=K).astype(np.float32)  # random walk to span states
            x = sim.step(a).astype(np.float32)
    lat_d, true_d = np.array(lat_d), np.array(true_d)
    pear = float(pearsonr(lat_d, true_d)[0]); spear = float(spearmanr(lat_d, true_d)[0])
    align = {"pearson": pear, "spearman": spear, "n_points": int(len(lat_d))}
    print(f"[DIAG2 latent-vs-true distance] pearson={pear:.3f} spearman={spear:.3f} (n={len(lat_d)})")

    # ---- DIAG 3: WORLD-MODEL ROLLOUT ACCURACY (latent space) ----
    H = mpc_steps
    divs = []
    for (src, tgt) in pairs:
        x0 = sim.reset(attractor=src).astype(np.float32)
        acts = rng.uniform(0.0, action_max, size=(H, K)).astype(np.float32)
        z0 = state_enc.encode(jepa, x0)                                   # [1,D,1,1,1]
        a_thk = torch.from_numpy(acts.T[None]).float()                   # [1,K,H]
        z_model = rollout_latent(jepa.predictor, z0, a_thk)[0]           # [H,D]
        x = x0.copy(); z_true = []
        for t in range(H):
            x = sim.step(acts[t]).astype(np.float32)
            z_true.append(state_enc.encode(jepa, x).flatten(1)[0])       # [D]
        z_true = torch.stack(z_true)                                     # [H,D]
        d = (torch.linalg.norm(z_model - z_true, dim=-1) /
             torch.linalg.norm(z_true, dim=-1).clamp_min(1e-6))          # [H] normalized
        divs.append(d.numpy())
    divs = np.stack(divs)  # [pairs, H]
    rollout = {"norm_div_t1": float(divs[:, 0].mean()), "norm_div_tH": float(divs[:, -1].mean()),
               "norm_div_mean": float(divs.mean()), "horizon": H}
    print(f"[DIAG3 model rollout vs true (latent)] norm_div t=1 {rollout['norm_div_t1']:.3f} "
          f"t={H} {rollout['norm_div_tH']:.3f} mean {rollout['norm_div_mean']:.3f}")

    res = {"oracle": oracle, "latent_alignment": align, "rollout_accuracy": rollout,
           "learned_mppi_success_rate": 0.0,
           "learned_mppi_note": "from job 74718: mppi 0% success, final_dist 4.88 (>= random 4.58)"}
    Path(out).mkdir(parents=True, exist_ok=True)
    fn = Path(out) / (f"planning_diagnosis{('_' + tag) if tag else ''}.json")
    with open(fn, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[diag] saved -> {fn}")
    return res


if __name__ == "__main__":
    fire.Fire(run)
