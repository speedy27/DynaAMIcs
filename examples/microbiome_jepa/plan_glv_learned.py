"""
M3 LAST LEVER — a LEARNED MONOTONIC COST on the frozen weak-reg latent.

Both representation bets left the planning latent NON-METRIC (raw latent distance corr ≈ 0), and
decoded-MPPI capped at the encoder's state-RECONSTRUCTION fidelity (~0.89 R²): it had to rebuild the
exact state to score it. A learned cost RELAXES that — it only needs to RANK candidate next-states by
true distance correctly (monotonicity), not reconstruct them. An imperfect latent may support a
monotonic distance signal where it cannot support exact reconstruction.

In-JEPA-spirit: this adds a small COST HEAD on the FROZEN latent; it does NOT turn the model into a
reconstruction / state-space model. We:
  * Substrate = the WEAK-REG world model (highest state-retention ~0.89 AND accurate latent rollout —
    the only latent where the rollout is good, so the cost is the sole remaining gap). NOT full-VICReg
    (collapsed, rollout artifact) nor SIGReg (rollout 0.61, non-metric).
  * Train a head h(z_a, z_b) on frozen-encoder latent PAIRS to RANK the true gLV state distance
    ||x_a - x_b|| (pairwise ranking / monotonic loss; target-agnostic, works for any target).
  * Plan with MPPI using h as the cost: encode target -> z_target, roll the latent forward with the
    (accurate) weak-reg predictor, score each candidate by sum_t h(z_rolled_t, z_target).
  * Compare vs oracle / random / raw latent-MPPI / decoded-MPPI on the same K=24 setup. Metrics:
    success@tol=1.0, final dist, and the head's held-out Spearman with true distance (the diagnostic).

LAST M3 lever: if learned-cost-MPPI crosses tol, M3 flips positive (fold to bnz); if not, it is a clean
capstone — the wall is the encoder's state-retention even for a ranking objective.

INTEGRITY: success from real MPC rollouts on the true GLVSimulator; the head is a probe on frozen
latents; EVERYTHING seeded (the unseeded-decoder fluke lesson). No fabrication.

Run (CPU, weak-reg substrate):
  .venv-cpu/bin/python -m examples.microbiome_jepa.plan_glv_learned \
      --checkpoint checkpoints/plan_model_k24_lowreg/latest.pth.tar --device cpu --seeds 0,1,2 \
      --overrides '{"data.n_candidate":24,"model.d_model":128,"model.regularizer.sim_coeff_t":4,"model.regularizer.cov_coeff":1,"model.regularizer.std_coeff":0.25}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from examples.microbiome_jepa.plan_glv import (
    MPPIConfig,
    _agg,
    _greedy_action,
    build_glv_and_encoder,
    build_world_model,
    mppi_plan,
    rollout_latent,
)
from examples.microbiome_jepa.plan_glv_decoded import _encode_states, fit_decoders, mppi_plan_decoded
from eb_jepa.logging import get_logger

logger = get_logger(__name__)


# ==================================================================================================
# Learned monotonic cost head  h(z_a, z_b) ~ rank of ||x_a - x_b||   (symmetric, frozen-latent probe)
# ==================================================================================================
class RankHead(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        # symmetric pair features: |z_a - z_b| and z_a * z_b  -> distance-like scalar (>=0)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Softplus(),
        )

    def forward(self, za, zb):
        feat = torch.cat([(za - zb).abs(), za * zb], dim=-1)
        return self.net(feat).squeeze(-1)


def fit_rank_head(jepa, state_enc, sim, n_traj=256, T=24, seed=0, device=None, hidden=256,
                  steps=3000, batch=256, attractor_frac=0.5):
    """Fit RankHead on frozen-encoder latent pairs to RANK true gLV state distance (pairwise BCE on the
    sign of h-differences vs true-distance-differences — pure ordering, not exact value). Half the pairs
    are anchored on an attractor (the planning distribution: distance-to-target). Returns (head, info)."""
    dev = device or state_enc.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = sim.generate_trajectories(n=n_traj, T=T, action_policy="random", seed=seed + 1)
    X = data["states"].reshape(-1, sim.n_species).astype(np.float32)
    Z = _encode_states(jepa, state_enc, X)                                  # [M,D]
    attractors = sim.attractors.astype(np.float32)                          # [A,S]
    Za = _encode_states(jepa, state_enc, attractors)                        # [A,D]
    M, A = len(Z), len(attractors)
    Zt = torch.from_numpy(Z).float().to(dev)
    Xt = torch.from_numpy(X).float().to(dev)
    Zat = torch.from_numpy(Za).float().to(dev)
    Xat = torch.from_numpy(attractors).float().to(dev)

    # held-out pair set for the Spearman diagnostic (attractor-anchored, like planning)
    n_held = 2000
    hi = rng.integers(0, M, n_held)
    ha = rng.integers(0, A, n_held)
    z_held_a, z_held_b = Zt[hi], Zat[ha]
    d_held = torch.linalg.norm(Xt[hi] - Xat[ha], dim=-1)

    head = RankHead(Zt.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def sample_batch():
        ia = torch.randint(0, M, (batch,), generator=g)
        # half the partners are attractors (target-anchored), half random states
        use_attr = torch.rand(batch, generator=g) < attractor_frac
        ib = torch.randint(0, M, (batch,), generator=g)
        ja = torch.randint(0, A, (batch,), generator=g)
        za = Zt[ia]
        zb = torch.where(use_attr.unsqueeze(-1).to(dev), Zat[ja], Zt[ib])
        xa = Xt[ia]
        xb = torch.where(use_attr.unsqueeze(-1).to(dev), Xat[ja], Xt[ib])
        d = torch.linalg.norm(xa - xb, dim=-1)
        return za, zb, d

    bce = nn.BCEWithLogitsLoss()
    head.train()
    for step in range(steps):
        za, zb, d = sample_batch()
        h = head(za, zb)                                        # [B]
        # pairwise ranking within the batch: h_i - h_j should be >0 iff d_i > d_j
        dh = h.unsqueeze(1) - h.unsqueeze(0)                    # [B,B]
        dd = d.unsqueeze(1) - d.unsqueeze(0)
        mask = dd.abs() > 1e-6
        target = (dd > 0).float()
        loss = bce(dh[mask], target[mask])
        opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        h_held = head(z_held_a, z_held_b).cpu().numpy()
    sp = float(spearmanr(h_held, d_held.cpu().numpy())[0])
    info = {"head_spearman_heldout": sp, "n_train_states": int(M), "steps": steps, "hidden": hidden}
    logger.info(f"[rank-head] held-out Spearman(h, true_dist) = {sp:.3f} (n={n_held})")
    return head, info


@torch.no_grad()
def mppi_plan_learned(predictor, z0, z_tgt, head, action_dim, action_max, cfg: MPPIConfig,
                      mean_init=None, generator=None):
    """MPPI in latent rollout, COST = learned head h(z_rolled_t, z_tgt) summed over the horizon."""
    device = z0.device
    H, K = cfg.horizon, action_dim
    mean = torch.zeros(H, K, device=device) if mean_init is None else mean_init.clone().to(device)
    std = torch.full((H, K), cfg.init_std, device=device)
    for _ in range(cfg.n_iters):
        noise = torch.randn(cfg.n_samples, H, K, device=device, generator=generator)
        actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-action_max, action_max)
        zs = rollout_latent(predictor, z0, actions.permute(0, 2, 1).contiguous())   # [N,H,D]
        N, Hh, D = zs.shape
        ztgt = z_tgt.reshape(1, -1).expand(N * Hh, D)
        h = head(zs.reshape(N * Hh, D), ztgt).reshape(N, Hh)                        # [N,H]
        cost = h.sum(dim=1) if cfg.cumulative else h[:, -1]
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
def run_episode(method, sim, jepa, state_enc, head, decode_fn, src, tgt, tol, mpc_steps, mppi_cfg,
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
        else:
            z0 = state_enc.encode(jepa, x)
            if method == "mppi_latent":
                a_t, mean_plan = mppi_plan(jepa.predictor, z0, z_tgt, K, action_max, mppi_cfg,
                                           mean_init=warm, generator=torch_gen)
            elif method == "mppi_decoded":
                a_t, mean_plan = mppi_plan_decoded(jepa.predictor, z0, x_tgt_t, decode_fn, K, action_max,
                                                   mppi_cfg, mean_init=warm, generator=torch_gen)
            elif method == "mppi_learned":
                a_t, mean_plan = mppi_plan_learned(jepa.predictor, z0, z_tgt, head, K, action_max,
                                                   mppi_cfg, mean_init=warm, generator=torch_gen)
            else:
                raise ValueError(f"unknown method {method!r}")
            warm = torch.zeros_like(mean_plan); warm[:-1] = mean_plan[1:]
            a = a_t.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError(f"method {method!r} non-finite action")
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
    head_steps: int = 3000,
    device: str = "cpu",
    out: str = "examples/microbiome_jepa/results",
    tag: str = "",
    overrides: Optional[dict] = None,
):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    seed_list = ([int(s) for s in seeds] if isinstance(seeds, (list, tuple))
                 else [int(seeds)] if isinstance(seeds, int)
                 else [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip()])

    jepa, cfg, K = build_world_model(fname, checkpoint, dev, overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    attr = sim.attractors
    n_attr = int(attr.shape[0])
    inter = [float(np.linalg.norm(attr[i] - attr[j])) for i in range(n_attr) for j in range(n_attr) if i != j]
    tol = tol_frac * float(np.mean(inter))

    # learned cost head (seeded) + decoder (for the decoded-MPPI baseline row)
    head, head_info = fit_rank_head(jepa, state_enc, sim, T=int(cfg.data.T), device=dev, steps=head_steps)
    decoders = fit_decoders(jepa, state_enc, sim, T=int(cfg.data.T), device=dev)
    decode_mlp = decoders["mlp"][0]

    mppi_cfg = MPPIConfig(horizon=horizon, n_samples=n_samples, n_elites=n_elites, n_iters=n_iters,
                          init_std=0.25, cumulative=True)
    methods = ["random", "greedy", "mppi_latent", "mppi_decoded", "mppi_learned"]
    print(f"[plan-learned] K={K} D={jepa.predictor.rnn.hidden_size} tol={tol:.3f} | "
          f"head Spearman(h,true)={head_info['head_spearman_heldout']:.3f} decoder_mlp_R2={decoders['mlp'][1]:.3f}")

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
                records.append({"seed": seed, **run_episode(m, sim, jepa, state_enc, head, decode_mlp,
                                src, tgt, tol, mpc_steps, mppi_cfg, rng, torch_gen=tgen)})

    summary = {}
    for m in methods:
        per_seed = []
        for seed in seed_list:
            ep = [r for r in records if r["method"] == m and r["seed"] == seed]
            per_seed.append(float(np.mean([r["success"] for r in ep])) if ep else float("nan"))
        mean, se = _agg(per_seed)
        allm = [r for r in records if r["method"] == m]
        summary[m] = {"success_rate_mean": mean, "success_rate_se": se,
                      "mean_final_dist": float(np.mean([r["final_dist"] for r in allm])),
                      "mean_best_dist": float(np.mean([r["best_dist"] for r in allm])),
                      "mean_start_dist": float(np.mean([r["start_dist"] for r in allm]))}

    print("\n========= gLV PLANNING: learned monotonic cost vs baselines (K=%d) =========" % K)
    print(f"head Spearman(h, true_dist) = {head_info['head_spearman_heldout']:.3f}  | tol={tol:.3f} "
          f"start={summary['random']['mean_start_dist']:.2f}  (oracle ref: 1.00 / 0.79)")
    print("method".ljust(16) + "success".ljust(20) + "final_dist".ljust(14) + "best_dist")
    for m in methods:
        s = summary[m]
        print(m.ljust(16) + f"{s['success_rate_mean']:.3f} ± {s['success_rate_se']:.3f}".ljust(20)
              + f"{s['mean_final_dist']:.3f}".ljust(14) + f"{s['mean_best_dist']:.3f}")

    res = {"checkpoint": checkpoint, "action_dim": K, "tol": tol, "head": head_info,
           "decoder_r2": {k: v[1] for k, v in decoders.items()}, "seeds": seed_list,
           "n_episodes": n_episodes, "mpc_steps": mpc_steps, "mppi_cfg": mppi_cfg.__dict__,
           "summary": summary, "records": records}
    Path(out).mkdir(parents=True, exist_ok=True)
    fn = Path(out) / (f"planning_learned{('_' + tag) if tag else ''}.json")
    with open(fn, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved -> {fn}")
    return res


if __name__ == "__main__":
    import fire
    fire.Fire(run)
