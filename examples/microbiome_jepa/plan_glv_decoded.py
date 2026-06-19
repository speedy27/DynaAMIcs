"""
A3 FIX — decoded-state planning (completes the M3 story).

The chain of MEASURED evidence (all on the SAME K=24 default-reg world model unless noted):
  * Oracle K-sweep (oracle_K_sweep.py): the gLV task is CONTROLLABLE at K=24 — a PERFECT-model MPPI
    reaches all targets (success 1.00, final 0.78) while 0% at K<=21.
  * Learned latent-MPPI at K=24 (job 74933): 0% (final 4.27), WORSE than random's best (3.58).
  * Diagnostics on the K=24 model (diagnose_planning --tag k24):
      - oracle 100% (controllable), rollout divergence ~2% over 20 steps (world model FAITHFUL),
      - latent-cost alignment Pearson -0.00 / Spearman 0.02 => the LATENT distance is UNINFORMATIVE
        about true proximity to the target. THE COST is the bottleneck, not controllability or rollout.

THE FIX (this file): keep the SAME frozen world model (set-transformer encoder f_theta + GRU predictor
g_phi). Fit a state readout D: z -> x_hat (a probe on encoded gLV states) and have MPPI roll the latent
forward with the LEARNED predictor but score candidate actions by ||decode(z_t) - x_target|| in (decoded)
state space. We compare two readouts to characterize HOW MUCH readout fidelity planning needs:
  - linear (Ridge)   — a strict linear probe;
  - mlp (1 hidden)   — a small nonlinear probe.
Everything else (env, success = TRUE final state within tol, baselines) is identical to plan_glv.

INTEGRITY: every success number comes from real MPC rollouts on the true GLVSimulator. The decoders are
probes fit on encoded simulator states (a standard frozen-encoder readout); we report each decoder's
held-out R^2 so its fidelity is explicit. No number is fabricated.

Run (CPU, with the K=24 checkpoint):
  .venv-cpu/bin/python -m examples.microbiome_jepa.plan_glv_decoded \
      --checkpoint checkpoints/plan_model_k24/latest.pth.tar --device cpu --seeds 0,1,2 \
      --n_episodes 12 --overrides '{"data.n_candidate": 24, "model.d_model": 128}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from examples.microbiome_jepa.plan_glv import (
    MPPIConfig,
    _agg,
    _greedy_action,
    build_glv_and_encoder,
    build_world_model,
    mppi_plan,
    rollout_latent,
)
from eb_jepa.logging import get_logger

logger = get_logger(__name__)


# ==================================================================================================
# State readouts  D: z -> x   (probes fit on encoded gLV states); returned as torch decode_fn(zs)->x_hat
# ==================================================================================================
@torch.no_grad()
def _encode_states(jepa, state_enc, X: np.ndarray, bs: int = 256) -> np.ndarray:
    ds = state_enc.ds
    Z = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().unsqueeze(1)        # [b,1,S]
        raw_tok, mask = ds._build_tokens(xb, ds.cfg)
        z = ds.zscore.transform(raw_tok) * mask.unsqueeze(-1).to(torch.float32)
        obs = {"otu": z.to(torch.float32).to(state_enc.device),
               "mask": mask.to(torch.bool).to(state_enc.device)}
        Z.append(jepa.encode(obs).flatten(1).float().cpu().numpy())
    return np.concatenate(Z, 0)


def _split(n, seed=123, frac=0.8):
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(frac * n)
    return perm[:cut], perm[cut:]


def fit_decoders(jepa, state_enc, sim, n_traj=256, T=24, seed=123, device=None, alpha=1.0,
                 mlp_hidden=256, mlp_steps=600):
    """Encode gLV states once, fit BOTH a linear (Ridge) and an MLP readout z->x. Returns
    {name: (decode_fn, r2)} where decode_fn maps a torch tensor [...,D] -> [...,S]."""
    dev = device or state_enc.device
    data = sim.generate_trajectories(n=n_traj, T=T, action_policy="random", seed=seed)
    X = data["states"].reshape(-1, sim.n_species).astype(np.float32)
    Z = _encode_states(jepa, state_enc, X)
    tr, te = _split(len(Z), seed)

    mu = Z[tr].mean(0, keepdims=True)
    sd = Z[tr].std(0, keepdims=True)
    sd = np.where(sd < 1e-4, 1.0, sd)
    mu_t = torch.from_numpy(mu).float().to(dev)
    sd_t = torch.from_numpy(sd).float().to(dev)

    out = {}

    # ---- linear (Ridge), folded standardization into an effective linear map ----
    ridge = Ridge(alpha=alpha).fit((Z[tr] - mu) / sd, X[tr])
    r2_lin = float(r2_score(X[te], ridge.predict((Z[te] - mu) / sd), multioutput="uniform_average"))
    ridge_all = Ridge(alpha=alpha).fit((Z - mu) / sd, X)
    W = torch.from_numpy((ridge_all.coef_ / sd).astype(np.float32)).to(dev)         # [S,D]
    b = torch.from_numpy((ridge_all.intercept_ - (mu / sd) @ ridge_all.coef_.T)
                         .reshape(-1).astype(np.float32)).to(dev)                    # [S]
    out["linear"] = ((lambda zs: zs @ W.T + b), r2_lin)

    # ---- mlp (1 hidden layer), trained with Adam on standardized z ----
    Zt = torch.from_numpy((Z - mu) / sd).float().to(dev)
    Xt = torch.from_numpy(X).float().to(dev)
    Ztr, Xtr = Zt[tr], Xt[tr]
    mlp = nn.Sequential(nn.Linear(Z.shape[1], mlp_hidden), nn.GELU(),
                        nn.Linear(mlp_hidden, sim.n_species)).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(mlp_steps):
        opt.zero_grad()
        loss = lossf(mlp(Ztr), Xtr)
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        pred_te = mlp(Zt[te]).cpu().numpy()
    r2_mlp = float(r2_score(X[te], pred_te, multioutput="uniform_average"))
    # decode_fn standardizes raw latents before the (already-standardized-input) MLP
    out["mlp"] = ((lambda zs: mlp((zs - mu_t) / sd_t)), r2_mlp)

    logger.info(f"[decoders] linear R^2={r2_lin:.3f}  mlp R^2={r2_mlp:.3f}  (n={len(Z)})")
    return out


# ==================================================================================================
# Decoded-state MPPI (cost in decoded TRUE-state space) — mirrors plan_glv.mppi_plan
# ==================================================================================================
@torch.no_grad()
def mppi_plan_decoded(predictor, z0, x_tgt, decode_fn: Callable, action_dim, action_max,
                      cfg: MPPIConfig, mean_init=None, generator=None):
    device = z0.device
    H, K = cfg.horizon, action_dim
    x_tgt = x_tgt.reshape(1, 1, -1)
    mean = torch.zeros(H, K, device=device) if mean_init is None else mean_init.clone().to(device)
    std = torch.full((H, K), cfg.init_std, device=device)
    for _ in range(cfg.n_iters):
        noise = torch.randn(cfg.n_samples, H, K, device=device, generator=generator)
        actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-action_max, action_max)
        zs = rollout_latent(predictor, z0, actions.permute(0, 2, 1).contiguous())   # [N,H,D]
        x_hat = decode_fn(zs)                                                        # [N,H,S]
        step_dist = torch.linalg.norm(x_hat - x_tgt, dim=-1)
        cost = step_dist.sum(dim=1) if cfg.cumulative else step_dist[:, -1]
        n_el = min(cfg.n_elites, cfg.n_samples)
        elite_cost, elite_idx = torch.topk(-cost, n_el)
        elite_cost = -elite_cost
        elite_actions = actions[elite_idx]
        w = torch.exp(cfg.temperature * (cost.min() - elite_cost))
        w = w / (w.sum() + 1e-9)
        mean = (w.view(n_el, 1, 1) * elite_actions).sum(dim=0)
        var = (w.view(n_el, 1, 1) * (elite_actions - mean.unsqueeze(0)) ** 2).sum(dim=0)
        std = var.sqrt().clamp_min(cfg.min_std)
    return mean[0].clamp(-action_max, action_max), mean


@torch.no_grad()
def run_episode(method, sim, jepa, state_enc, decoders, src, tgt, tol, mpc_steps, mppi_cfg,
                rng, torch_gen=None):
    action_max = float(sim.config.action_max)
    K = int(sim.action_dim)
    target_state = sim.attractors[tgt]
    x = sim.reset(attractor=src).astype(np.float32)
    start_dist = float(np.linalg.norm(x - target_state))
    best_dist = start_dist
    z_tgt = state_enc.encode(jepa, target_state).flatten(1)
    x_tgt_t = torch.from_numpy(target_state.astype(np.float32)).to(state_enc.device)
    warm = None
    for _ in range(mpc_steps):
        if method == "random":
            a = rng.uniform(0.0, action_max, size=K).astype(np.float32)
        elif method == "greedy":
            a = _greedy_action(sim, x, target_state, action_max)
        elif method == "mppi_latent":
            z0 = state_enc.encode(jepa, x)
            a_t, mean_plan = mppi_plan(jepa.predictor, z0, z_tgt, K, action_max, mppi_cfg,
                                       mean_init=warm, generator=torch_gen)
            warm = torch.zeros_like(mean_plan); warm[:-1] = mean_plan[1:]
            a = a_t.detach().cpu().numpy().astype(np.float32)
        elif method.startswith("mppi_decoded_"):
            decode_fn = decoders[method.split("mppi_decoded_")[1]][0]
            z0 = state_enc.encode(jepa, x)
            a_t, mean_plan = mppi_plan_decoded(jepa.predictor, z0, x_tgt_t, decode_fn, K, action_max,
                                               mppi_cfg, mean_init=warm, generator=torch_gen)
            warm = torch.zeros_like(mean_plan); warm[:-1] = mean_plan[1:]
            a = a_t.detach().cpu().numpy().astype(np.float32)
        else:
            raise ValueError(f"unknown method {method!r}")
        if not np.all(np.isfinite(a)):
            raise FloatingPointError(f"method {method!r} produced non-finite action")
        x = sim.step(a).astype(np.float32)
        d = float(np.linalg.norm(x - target_state))
        best_dist = min(best_dist, d)
        if d < tol:
            break
    final_dist = float(np.linalg.norm(x - target_state))
    return {"method": method, "success": bool(final_dist < tol), "final_dist": final_dist,
            "start_dist": start_dist, "best_dist": best_dist, "src": int(src), "tgt": int(tgt)}


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: Optional[str] = None,
    seeds: str = "0,1,2",
    n_episodes: int = 12,
    mpc_steps: int = 20,
    horizon: int = 6,
    n_samples: int = 128,
    n_elites: int = 16,
    n_iters: int = 3,
    tol_frac: float = 0.15,
    dec_n_traj: int = 256,
    device: str = "cpu",
    out: str = "examples/microbiome_jepa/results",
    overrides: Optional[dict] = None,
):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    if isinstance(seeds, (list, tuple)):
        seed_list = [int(s) for s in seeds]
    elif isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip() != ""]

    jepa, cfg, K = build_world_model(fname, checkpoint, dev, overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    attr = sim.attractors
    n_attr = int(attr.shape[0])
    inter = [float(np.linalg.norm(attr[i] - attr[j])) for i in range(n_attr) for j in range(n_attr) if i != j]
    attr_scale = float(np.mean(inter))
    tol = tol_frac * attr_scale

    decoders = fit_decoders(jepa, state_enc, sim, n_traj=dec_n_traj, T=int(cfg.data.T), device=dev)
    mppi_cfg = MPPIConfig(horizon=horizon, n_samples=n_samples, n_elites=n_elites, n_iters=n_iters,
                          init_std=0.25, cumulative=True)
    methods = ["random", "greedy", "mppi_latent", "mppi_decoded_linear", "mppi_decoded_mlp"]
    print(f"[plan-decoded] trained={checkpoint is not None} K={K} D={jepa.predictor.rnn.hidden_size} "
          f"tol={tol:.3f} | decoder R^2: linear={decoders['linear'][1]:.3f} mlp={decoders['mlp'][1]:.3f}")

    records: List[dict] = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        tgen = torch.Generator(device=dev).manual_seed(seed)
        pairs = []
        for _ in range(n_episodes):
            s = int(rng.integers(n_attr)); t = int(rng.integers(n_attr - 1)); t += int(t >= s)
            pairs.append((s, t))
        for (src, tgt) in pairs:
            for m in methods:
                records.append({"seed": seed, **run_episode(m, sim, jepa, state_enc, decoders, src,
                                tgt, tol, mpc_steps, mppi_cfg, rng, torch_gen=tgen)})

    summary = {}
    for m in methods:
        per_seed = []
        for seed in seed_list:
            ep = [r for r in records if r["method"] == m and r["seed"] == seed]
            per_seed.append(float(np.mean([r["success"] for r in ep])) if ep else float("nan"))
        mean, se = _agg(per_seed)
        allm = [r for r in records if r["method"] == m]
        summary[m] = {"success_rate_mean": mean, "success_rate_se": se,
                      "per_seed_success_rate": per_seed,
                      "mean_final_dist": float(np.mean([r["final_dist"] for r in allm])),
                      "mean_best_dist": float(np.mean([r["best_dist"] for r in allm])),
                      "mean_start_dist": float(np.mean([r["start_dist"] for r in allm]))}

    print("\n========= gLV PLANNING: latent cost vs DECODED-STATE cost (K=%d) =========" % K)
    print(f"decoder R^2: linear={decoders['linear'][1]:.3f}  mlp={decoders['mlp'][1]:.3f}  | "
          f"tol={tol:.3f} start={summary['random']['mean_start_dist']:.2f}")
    print("method".ljust(20) + "success_rate".ljust(22) + "final_dist".ljust(14) + "best_dist")
    for m in methods:
        s = summary[m]
        print(m.ljust(20) + f"{s['success_rate_mean']:.3f} ± {s['success_rate_se']:.3f}".ljust(22)
              + f"{s['mean_final_dist']:.3f}".ljust(14) + f"{s['mean_best_dist']:.3f}")

    res = {"checkpoint": checkpoint, "action_dim": K, "tol": tol, "attr_scale": attr_scale,
           "decoder_r2": {k: v[1] for k, v in decoders.items()}, "seeds": seed_list,
           "n_episodes": n_episodes, "mpc_steps": mpc_steps, "mppi_cfg": mppi_cfg.__dict__,
           "summary": summary, "records": records}
    Path(out).mkdir(parents=True, exist_ok=True)
    fn = Path(out) / "planning_decoded.json"
    with open(fn, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved -> {fn}")
    return res


if __name__ == "__main__":
    import fire
    fire.Fire(run)
