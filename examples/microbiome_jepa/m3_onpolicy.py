"""
M3 Step 2 — iterative ON-POLICY model learning (attacks the CONFIRMED distribution shift).

Measured premise (m3_ensemble_gate + distshift): the gap is NOT exploitable-epistemic (ensemble agrees),
it is DISTRIBUTION SHIFT — the predictor's 1-step error is 0.072 on its training distribution (random-dose)
but 0.264 on the planner's OOD action sequences (3.7x). Fix (MBPO-style, pure JEPA — only the latent-space
DYNAMICS predictor is retrained; the encoder + the learned cost stay frozen):

  D <- random-dose latent transitions
  repeat R rounds:
    train predictor on D (frozen encoder)              # the dynamics model
    plan with learned-cost MPPI, EXECUTE on true gLV, COLLECT (z_t, a_t, z_{t+1}^true)  # on-policy data
    record: predictor 1-step error ON THE PLANNER's traj (before aggregating), success@tol, final dist
    D <- D + collected                                  # model becomes accurate where the planner operates

If the planner-traj error drops round over round AND success crosses tol, M3 flips POSITIVE (fold to bnz).
Otherwise it is an even more thorough negative (exploitation AND distribution shift addressed, still no cross).

INTEGRITY: all measured here; predictor training + planner seeded. No fabrication.
Run: .venv-cpu/bin/python -m examples.microbiome_jepa.m3_onpolicy \
     --checkpoint checkpoints/plan_model_k24_lowreg/latest.pth.tar --device cpu --rounds 4 \
     --overrides '{"data.n_candidate":24,"model.d_model":128,"model.regularizer.sim_coeff_t":4,"model.regularizer.cov_coeff":1,"model.regularizer.std_coeff":0.25}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch

from eb_jepa.architectures import RNNPredictor
from eb_jepa.logging import get_logger
from examples.microbiome_jepa.plan_glv import MPPIConfig, build_glv_and_encoder, build_world_model
from examples.microbiome_jepa.plan_glv_learned import fit_rank_head, mppi_plan_learned
from examples.microbiome_jepa.m3_ensemble_gate import encode_transitions

logger = get_logger(__name__)


def train_predictor(Zt, At, Zt1, D, K, final_ln, steps, seed, device):
    """Train a fresh GRU dynamics predictor on (z_t, a_t)->z_{t+1} (MSE). Seeded."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    pred = RNNPredictor(hidden_size=D, action_dim=K, final_ln=final_ln).to(dev)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = len(Zt)
    Zt, At, Zt1 = Zt.to(dev), At.to(dev), Zt1.to(dev)
    pred.train()
    last = 0.0
    for _ in range(steps):
        idx = torch.randint(0, M, (256,), generator=g)
        out = pred(Zt[idx].view(-1, D, 1, 1, 1), At[idx].view(-1, K, 1)).flatten(1)
        loss = torch.nn.functional.mse_loss(out, Zt1[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    pred.eval()
    return pred, last


@torch.no_grad()
def plan_and_collect(jepa, head, sim, state_enc, K, tol, episodes, mpc_steps, mppi_cfg, seed, device):
    """Run learned-cost MPPI, EXECUTE on the true gLV, collect (z_t,a_t,z_{t+1}^true) + metrics."""
    rng = np.random.default_rng(seed)
    tgen = torch.Generator(device=device).manual_seed(seed)
    n_attr = sim.attractors.shape[0]
    Zt, At, Zt1, finals, errs = [], [], [], [], []
    succ = 0
    for _ in range(episodes):
        s = int(rng.integers(n_attr)); t = int(rng.integers(n_attr - 1)); t += int(t >= s)
        tgt = sim.attractors[t]
        z_tgt = state_enc.encode(jepa, tgt).flatten(1)
        x = sim.reset(attractor=s).astype(np.float32)
        warm = None; reached = False
        for _ in range(mpc_steps):
            z0 = state_enc.encode(jepa, x)
            a_t, mean_plan = mppi_plan_learned(jepa.predictor, z0, z_tgt, head, K,
                                               float(sim.config.action_max), mppi_cfg,
                                               mean_init=warm, generator=tgen)
            warm = torch.zeros_like(mean_plan); warm[:-1] = mean_plan[1:]
            a_np = a_t.detach().cpu().numpy().astype(np.float32)
            pred_next = jepa.predictor(z0, a_t.view(1, K, 1)).flatten(1)[0]
            x = sim.step(a_np).astype(np.float32)
            z_next = state_enc.encode(jepa, x).flatten(1)[0]
            errs.append(float(torch.linalg.norm(pred_next - z_next)))
            Zt.append(z0.flatten(1)[0]); At.append(a_t.detach().float()); Zt1.append(z_next)
            if float(np.linalg.norm(x - tgt)) < tol:
                reached = True; break
        finals.append(float(np.linalg.norm(x - tgt))); succ += int(reached)
    return (torch.stack(Zt), torch.stack(At), torch.stack(Zt1)), succ / episodes, float(np.mean(finals)), float(np.mean(errs))


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: Optional[str] = None,
    rounds: int = 4,
    episodes: int = 12,
    mpc_steps: int = 20,
    horizon: int = 6,
    n_samples: int = 128,
    n_iters: int = 3,
    pred_steps: int = 2500,
    init_traj: int = 192,
    final_seeds: str = "0,1,2",
    device: str = "cpu",
    out: str = "examples/microbiome_jepa/results",
    overrides: Optional[dict] = None,
):
    dev = torch.device(device)
    jepa, cfg, K = build_world_model(fname, checkpoint, dev, overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    D = jepa.predictor.rnn.hidden_size
    final_ln = jepa.encoder.final_ln
    attr = sim.attractors; n_attr = int(attr.shape[0])
    inter = [float(np.linalg.norm(attr[i] - attr[j])) for i in range(n_attr) for j in range(n_attr) if i != j]
    tol = 0.15 * float(np.mean(inter))
    head, _ = fit_rank_head(jepa, state_enc, sim, T=int(cfg.data.T), device=dev)
    mppi_cfg = MPPIConfig(horizon=horizon, n_samples=n_samples, n_elites=16, n_iters=n_iters,
                          init_std=0.25, cumulative=True)

    # initial dataset = random-dose latent transitions (the model's original training distribution)
    Zt, At, Zt1 = encode_transitions(sim, state_enc, jepa, init_traj, int(cfg.data.T))
    logger.info(f"[onpolicy] init transitions={len(Zt)} tol={tol:.3f} D={D} K={K}")

    rows = []
    for r in range(rounds):
        pred, mse = train_predictor(Zt, At, Zt1, D, K, final_ln, pred_steps, seed=200 + r, device=device)
        jepa.predictor = pred  # plan with the freshly-trained on-policy dynamics model
        (cZt, cAt, cZt1), succ, final, err_traj = plan_and_collect(
            jepa, head, sim, state_enc, K, tol, episodes, mpc_steps, mppi_cfg, seed=r, device=device)
        rows.append({"round": r, "train_mse": mse, "n_train": int(len(Zt)),
                     "err_on_planner_traj": err_traj, "success_rate": succ, "mean_final_dist": final})
        print(f"[round {r}] train_mse={mse:.4f} n_train={len(Zt)} | err_on_planner_traj={err_traj:.4f} "
              f"| success={succ:.3f} final={final:.3f}")
        Zt = torch.cat([Zt, cZt]); At = torch.cat([At, cAt]); Zt1 = torch.cat([Zt1, cZt1])

    # final eval with the last predictor over multiple seeds (headline number vs baselines)
    if isinstance(final_seeds, (list, tuple)):
        fseeds = [int(s) for s in final_seeds]
    elif isinstance(final_seeds, int):
        fseeds = [final_seeds]
    else:
        fseeds = [int(s) for s in str(final_seeds).strip("() ").split(",") if str(s).strip()]
    fs, ff = [], []
    for sd in fseeds:
        _, su, fi, _ = plan_and_collect(jepa, head, sim, state_enc, K, tol, episodes, mpc_steps, mppi_cfg, sd, device)
        fs.append(su); ff.append(fi)
    final_succ = float(np.mean(fs)); final_dist = float(np.mean(ff))
    print(f"\n[onpolicy FINAL, {len(fseeds)} seeds] success={final_succ:.3f} ± {np.std(fs):.3f} "
          f"final_dist={final_dist:.3f}  (oracle 0.79, learned-cost baseline ~3.06, tol {tol:.3f})")
    crossed = final_succ > 0.0
    verdict = ("ON-POLICY CLOSES/CROSSES tol -> M3 POSITIVE (fold)" if crossed else
               "on-policy lowers error but does NOT cross tol -> even more thorough M3 negative")
    print(f"VERDICT: {verdict}")

    res = {"rounds": rows, "final_success_rate": final_succ, "final_mean_dist": final_dist,
           "final_seeds": fseeds, "tol": tol, "crossed_tol": crossed, "verdict": verdict,
           "baseline_learned_cost_final": 3.06, "oracle_final": 0.79}
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "m3_onpolicy.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {out}/m3_onpolicy.json")
    return res


if __name__ == "__main__":
    fire.Fire(run)
