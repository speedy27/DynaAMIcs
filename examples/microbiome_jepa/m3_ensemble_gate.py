"""
M3 model-fidelity push — STEP 1 cheap diagnostic GATE: is the planning gap MODEL EXPLOITATION (epistemic)?

Diagnosis so far: the wall is a learned-model-vs-true-dynamics fidelity gap (oracle 0.79 vs learned ~3.06
at the same horizon). Hypothesis: the planner optimizes against the model and EXPLOITS its errors in
regions its action sequences make OOD. If so, an ENSEMBLE of predictors (trained on the model's data) will
DISAGREE exactly where the single model errs along the planner's trajectories — and a disagreement penalty
(pessimistic planning) should help. If the ensemble AGREES even where the model errs, the gap is
structural/aleatoric, not epistemic, and pessimism won't help — record that and stop.

GATE (this file, CPU — frozen encoder + small GRU predictors, no GPU needed):
  1. Freeze the weak-reg encoder + its trained predictor (the planner's model).
  2. Train an ENSEMBLE of K fresh GRU predictors on the SAME frozen-encoder latent transitions
     (z_t, a_t -> z_{t+1}) from random-dose trajectories (the model's training distribution), different seeds.
  3. Run the learned-cost MPPI planner (receding-horizon) on the true gLV; log per executed step
     (z_t, a_t, z_{t+1}^true, episode success).
  4. Per step: ensemble DISAGREEMENT = std over the K predictors' next-latent predictions; true-model
     ERROR = ||planner_predictor(z_t,a_t) - encode(true next state)||. Report Pearson/Spearman(disagreement,
     error) and mean disagreement on FAILED vs reached-best steps. Positive corr + high disagreement where
     plans fail => exploitation is real => Step 1 (pessimistic planning) is warranted.

INTEGRITY: all numbers measured here; ensemble + planner seeded. No fabrication.
Run: .venv-cpu/bin/python -m examples.microbiome_jepa.m3_ensemble_gate \
     --checkpoint checkpoints/plan_model_k24_lowreg/latest.pth.tar --device cpu \
     --overrides '{"data.n_candidate":24,"model.d_model":128,"model.regularizer.sim_coeff_t":4,"model.regularizer.cov_coeff":1,"model.regularizer.std_coeff":0.25}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import fire
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from eb_jepa.architectures import RNNPredictor
from eb_jepa.logging import get_logger
from examples.microbiome_jepa.plan_glv import (
    MPPIConfig, _greedy_action, build_glv_and_encoder, build_world_model, rollout_latent,
)
from examples.microbiome_jepa.plan_glv_learned import fit_rank_head, mppi_plan_learned

logger = get_logger(__name__)


@torch.no_grad()
def encode_transitions(sim, state_enc, jepa, n_traj, T, seed=7):
    """Random-dose trajectories -> (Zt[M,D], At[M,K], Zt1[M,D]) latent transitions (model train dist)."""
    data = sim.generate_trajectories(n=n_traj, T=T, action_policy="random", seed=seed)
    states, actions = data["states"], data["actions"]              # [n,T+1,S], [n,T,K]
    Zt, At, Zt1 = [], [], []
    for i in range(states.shape[0]):
        zs = [state_enc.encode(jepa, states[i, t]).flatten(1)[0] for t in range(states.shape[1])]
        for t in range(actions.shape[1]):
            Zt.append(zs[t]); At.append(torch.from_numpy(actions[i, t]).float()); Zt1.append(zs[t + 1])
    return torch.stack(Zt), torch.stack(At), torch.stack(Zt1)


def train_ensemble(Zt, At, Zt1, D, K, final_ln, n_models=5, steps=1500, batch=256, device="cpu", base_seed=100):
    """Train n_models fresh GRU predictors on (z_t,a_t)->z_{t+1} (MSE). Returns list of predictors."""
    dev = torch.device(device)
    M = len(Zt)
    Zt, At, Zt1 = Zt.to(dev), At.to(dev), Zt1.to(dev)
    models = []
    for m in range(n_models):
        torch.manual_seed(base_seed + m)
        pred = RNNPredictor(hidden_size=D, action_dim=K, final_ln=final_ln).to(dev)
        opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(base_seed + m)
        pred.train()
        for _ in range(steps):
            idx = torch.randint(0, M, (batch,), generator=g)
            zt = Zt[idx].view(batch, D, 1, 1, 1)
            at = At[idx].view(batch, K, 1)
            out = pred(zt, at).flatten(1)                          # [B,D]
            loss = torch.nn.functional.mse_loss(out, Zt1[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        pred.eval()
        models.append(pred)
        logger.info(f"[ensemble] trained predictor {m} (final MSE {loss.item():.4f})")
    return models


@torch.no_grad()
def disagreement_and_error(models, jepa, zt, at):
    """zt[1,D,1,1,1], at[1,K,1] -> (ensemble disagreement, true-vs-? n/a here). Returns ens preds + main pred."""
    main = jepa.predictor(zt, at).flatten(1)[0]                    # [D] the planner's model
    ens = torch.stack([m(zt, at).flatten(1)[0] for m in models])   # [n_models, D]
    disagreement = float(ens.std(dim=0).mean())                    # mean per-dim std across ensemble
    return main, disagreement


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: Optional[str] = None,
    seeds: str = "0,1,2",
    n_episodes: int = 12,
    mpc_steps: int = 20,
    horizon: int = 6,
    n_samples: int = 128,
    n_iters: int = 3,
    n_models: int = 5,
    ens_traj: int = 192,
    device: str = "cpu",
    out: str = "examples/microbiome_jepa/results",
    overrides: Optional[dict] = None,
):
    dev = torch.device(device)
    seed_list = ([int(s) for s in seeds] if isinstance(seeds, (list, tuple))
                 else [int(seeds)] if isinstance(seeds, int)
                 else [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip()])
    jepa, cfg, K = build_world_model(fname, checkpoint, dev, overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    D = jepa.predictor.rnn.hidden_size
    attr = sim.attractors; n_attr = int(attr.shape[0])
    inter = [float(np.linalg.norm(attr[i] - attr[j])) for i in range(n_attr) for j in range(n_attr) if i != j]
    tol = 0.15 * float(np.mean(inter))

    # 1) learned-cost head (the planner's cost) + 2) ensemble of predictors on the frozen encoder
    head, _ = fit_rank_head(jepa, state_enc, sim, T=int(cfg.data.T), device=dev)
    Zt, At, Zt1 = encode_transitions(sim, state_enc, jepa, ens_traj, int(cfg.data.T))
    models = train_ensemble(Zt, At, Zt1, D, K, jepa.encoder.final_ln, n_models=n_models, device=device)
    logger.info(f"[gate] ensemble n={n_models} | tol={tol:.3f} | transitions={len(Zt)}")

    # 3) run the learned-cost MPPI planner, logging per executed step
    mppi_cfg = MPPIConfig(horizon=horizon, n_samples=n_samples, n_elites=16, n_iters=n_iters,
                          init_std=0.25, cumulative=True)
    disag, err, step_fail = [], [], []
    n_succ = 0
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        tgen = torch.Generator(device=dev).manual_seed(seed)
        for _ in range(n_episodes):
            s = int(rng.integers(n_attr)); t = int(rng.integers(n_attr - 1)); t += int(t >= s)
            target_state = attr[t]
            z_tgt = state_enc.encode(jepa, target_state).flatten(1)
            x = sim.reset(attractor=s).astype(np.float32)
            warm = None
            reached = False
            for _ in range(mpc_steps):
                z0 = state_enc.encode(jepa, x)
                a_t, mean_plan = mppi_plan_learned(jepa.predictor, z0, z_tgt, head, K,
                                                   float(sim.config.action_max), mppi_cfg,
                                                   mean_init=warm, generator=tgen)
                warm = torch.zeros_like(mean_plan); warm[:-1] = mean_plan[1:]
                a_np = a_t.detach().cpu().numpy().astype(np.float32)
                # log disagreement + true 1-step error at this executed (z_t, a_t)
                zt5 = z0; at3 = a_t.view(1, K, 1)
                main_pred, dis = disagreement_and_error(models, jepa, zt5, at3)
                x_next = sim.step(a_np).astype(np.float32)
                z_next_true = state_enc.encode(jepa, x_next).flatten(1)[0]
                true_e = float(torch.linalg.norm(main_pred - z_next_true))
                disag.append(dis); err.append(true_e)
                d = float(np.linalg.norm(x_next - target_state))
                step_fail.append(d)  # true distance to target at this step (high = failing region)
                x = x_next
                if d < tol:
                    reached = True; break
            n_succ += int(reached)

    disag, err, step_fail = np.array(disag), np.array(err), np.array(step_fail)
    pear = float(pearsonr(disag, err)[0]); spear = float(spearmanr(disag, err)[0])
    # "where plans fail" = steps still far from target (top tercile of true distance)
    far = step_fail >= np.quantile(step_fail, 0.66)
    near = step_fail <= np.quantile(step_fail, 0.34)
    res = {
        "n_steps": int(len(disag)), "n_episodes_total": len(seed_list) * n_episodes,
        "planner_success_rate": n_succ / (len(seed_list) * n_episodes),
        "corr_disagreement_vs_error_pearson": pear,
        "corr_disagreement_vs_error_spearman": spear,
        "mean_disagreement": float(disag.mean()), "mean_true_error": float(err.mean()),
        "disagreement_far_from_target": float(disag[far].mean()),
        "disagreement_near_target": float(disag[near].mean()),
        "error_far_from_target": float(err[far].mean()),
        "error_near_target": float(err[near].mean()),
        "n_models": n_models, "tol": tol,
    }
    print("\n================ M3 ENSEMBLE GATE (is it model exploitation?) ================")
    print(f"steps={res['n_steps']} planner_success={res['planner_success_rate']:.3f} tol={tol:.3f}")
    print(f"corr(disagreement, true_error): Pearson {pear:.3f} / Spearman {spear:.3f}")
    print(f"disagreement  far-from-target {res['disagreement_far_from_target']:.4f}  vs  "
          f"near {res['disagreement_near_target']:.4f}")
    print(f"true 1-step err far-from-target {res['error_far_from_target']:.4f}  vs  "
          f"near {res['error_near_target']:.4f}  (mean disag {res['mean_disagreement']:.4f}, "
          f"mean err {res['mean_true_error']:.4f})")
    verdict = ("EXPLOITATION LIKELY: disagreement correlates with error -> pessimistic planning warranted"
               if (pear > 0.2 and res['disagreement_far_from_target'] > res['disagreement_near_target'])
               else "NOT epistemic exploitation: ensemble agrees even where the model errs -> gap is "
                    "structural/aleatoric; pessimism unlikely to help")
    print(f"VERDICT: {verdict}")
    res["verdict"] = verdict
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "m3_ensemble_gate.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {out}/m3_ensemble_gate.json")
    return res


if __name__ == "__main__":
    fire.Fire(run)
