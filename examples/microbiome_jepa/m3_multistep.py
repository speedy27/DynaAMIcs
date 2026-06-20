"""
M3 FINAL lever — MULTI-STEP FREE-RUNNING rollout loss for the latent dynamics predictor.

New evidence (Step 2): a fresh 1-step-MSE predictor has LOWER 1-step error (0.135 vs 0.264) yet plans
WORSE (3.49 vs 3.06). So 1-step accuracy is DECOUPLED from planning — the binding constraint is the
predictor's FREE-RUNNING multi-step rollout fidelity (an exposure-bias / train-inference mismatch: trained
teacher-forced 1-step, but MPPI unrolls it free-running, feeding its own predictions). This is distinct
from compounding-DIVERGENCE (ruled out by the horizon sweep). This lever targets exactly that gap.

ONE change only: retrain the latent predictor with a multi-step FREE-RUNNING objective (unroll k steps
feeding its own predicted latents, conditioned on the action sequence; penalize the accumulated rollout
error vs the true k-step latent trajectory). Encoder FROZEN, learned cost FROZEN, same data as the 1-step
baseline — only the training objective changes, so the effect is attributable. Scheduled sampling ramps the
free-running fraction for stability. Seeded.

GATE (cheap, first): does multi-step training REDUCE the free-running k-step rollout error (vs the 1-step
predictor) on held-out + planner trajectories? If yes the lever bites -> re-plan. If it cannot, that points
to a representational/capacity limit (the model can't capture the multi-step stiff dynamics regardless of
objective) -> record that. Then re-plan: success@tol + final vs baselines. Report 1-step err, free-running
multi-step err, and planning side by side (the decoupling story).

Fold to bnz only if it crosses tol=1.0. Else: final, most precise M3 capstone.
Run: .venv-cpu/bin/python -m examples.microbiome_jepa.m3_multistep \
     --checkpoint checkpoints/plan_model_k24_lowreg/latest.pth.tar --device cpu --k 6 \
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
from examples.microbiome_jepa.plan_glv_learned import fit_rank_head
from examples.microbiome_jepa.m3_onpolicy import train_predictor, plan_and_collect

logger = get_logger(__name__)


@torch.no_grad()
def encode_sequences(sim, state_enc, jepa, n_traj, T, seed=11):
    """Random-dose trajectories -> latent sequences Z[n,T+1,D], actions A[n,T,K]."""
    data = sim.generate_trajectories(n=n_traj, T=T, action_policy="random", seed=seed)
    states, actions = data["states"], data["actions"]
    Z = []
    for i in range(states.shape[0]):
        Z.append(torch.stack([state_enc.encode(jepa, states[i, t]).flatten(1)[0]
                              for t in range(states.shape[1])]))
    return torch.stack(Z), torch.from_numpy(actions).float()           # [n,T+1,D], [n,T,K]


@torch.no_grad()
def free_running_error(predictor, Z, A, k, n_windows=400, seed=0):
    """Mean per-step FREE-RUNNING k-step rollout error: unroll feeding own predictions, vs true latents."""
    dev = Z.device
    n, Tp1, D = Z.shape; K = A.shape[-1]
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_windows):
        i = int(rng.integers(n)); t0 = int(rng.integers(0, Tp1 - 1 - k))
        z = Z[i, t0].view(1, D, 1, 1, 1)
        e = 0.0
        for j in range(k):
            z = predictor(z, A[i, t0 + j].view(1, K, 1))
            e += float(torch.linalg.norm(z.flatten(1)[0] - Z[i, t0 + 1 + j]))
        errs.append(e / k)                                             # mean per-step rollout error
    return float(np.mean(errs))


@torch.no_grad()
def one_step_error(predictor, Z, A, n_windows=2000, seed=0):
    dev = Z.device
    n, Tp1, D = Z.shape; K = A.shape[-1]
    rng = np.random.default_rng(seed)
    zt, at, zt1 = [], [], []
    for _ in range(n_windows):
        i = int(rng.integers(n)); t = int(rng.integers(0, Tp1 - 1))
        zt.append(Z[i, t]); at.append(A[i, t]); zt1.append(Z[i, t + 1])
    zt, at, zt1 = torch.stack(zt), torch.stack(at), torch.stack(zt1)
    pred = predictor(zt.view(-1, D, 1, 1, 1), at.view(-1, K, 1)).flatten(1)
    return float(torch.linalg.norm(pred - zt1, dim=-1).mean())


def train_multistep(Z, A, D, K, final_ln, k, steps, seed, device, batch=128, free_ramp=0.6):
    """Train predictor with a FREE-RUNNING k-step rollout loss + scheduled sampling (free-run fraction
    ramps 0->1 over the first `free_ramp` of training). Seeded."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    pred = RNNPredictor(hidden_size=D, action_dim=K, final_ln=final_ln).to(dev)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, Tp1, _ = Z.shape
    Z, A = Z.to(dev), A.to(dev)
    pred.train()
    last = 0.0
    for step in range(steps):
        p_free = min(1.0, (step / max(1, int(free_ramp * steps))))     # scheduled sampling
        ii = torch.randint(0, n, (batch,), generator=g)
        t0 = torch.randint(0, Tp1 - 1 - k, (batch,), generator=g)
        z = Z[ii, t0].view(batch, D, 1, 1, 1)                          # start from true z_t0
        loss = 0.0
        for j in range(k):
            a = A[ii, t0 + j].view(batch, K, 1)
            z_pred = pred(z, a)
            tgt = Z[ii, t0 + 1 + j]
            loss = loss + torch.nn.functional.mse_loss(z_pred.flatten(1), tgt)
            # scheduled sampling: next input is the model's own prediction w.p. p_free, else the truth
            use_free = (torch.rand(batch, generator=g) < p_free).to(dev).view(batch, 1, 1, 1, 1)
            z = torch.where(use_free, z_pred, tgt.view(batch, D, 1, 1, 1)).detach()
        loss = loss / k
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    pred.eval()
    return pred, last


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: Optional[str] = None,
    k: int = 6,
    steps: int = 4000,
    n_traj: int = 192,
    seeds: str = "0,1,2",
    n_episodes: int = 12,
    mpc_steps: int = 20,
    horizon: int = 6,
    n_samples: int = 128,
    n_iters: int = 3,
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

    # encode train + held-out sequences (same data setup as the 1-step baseline)
    Ztr, Atr = encode_sequences(sim, state_enc, jepa, n_traj, int(cfg.data.T), seed=11)
    Zho, Aho = encode_sequences(sim, state_enc, jepa, 48, int(cfg.data.T), seed=999)

    # BASELINE: fresh 1-step-MSE predictor (same data) ; VARIANT: multi-step free-running predictor
    Ztr_t = Ztr[:, :-1].reshape(-1, D); Atr_t = Atr.reshape(-1, K); Ztr_t1 = Ztr[:, 1:].reshape(-1, D)
    pred_1step, mse1 = train_predictor(Ztr_t, Atr_t, Ztr_t1, D, K, final_ln, 2500, seed=300, device=device)
    pred_multi, lossm = train_multistep(Ztr, Atr, D, K, final_ln, k, steps, seed=300, device=device)

    # GATE: free-running k-step rollout error (held-out) + 1-step error, before(1step) vs after(multi)
    g_1s_1step = one_step_error(pred_1step, Zho, Aho)
    g_1s_multi = one_step_error(pred_multi, Zho, Aho)
    g_fr_1step = free_running_error(pred_1step, Zho, Aho, k)
    g_fr_multi = free_running_error(pred_multi, Zho, Aho, k)
    print("\n================ M3 MULTI-STEP ROLLOUT — GATE (free-running fidelity) ================")
    print(f"held-out 1-step error:        1step-pred {g_1s_1step:.4f}   multistep-pred {g_1s_multi:.4f}")
    print(f"held-out FREE-RUN {k}-step err: 1step-pred {g_fr_1step:.4f}   multistep-pred {g_fr_multi:.4f}  "
          f"({'REDUCED' if g_fr_multi < g_fr_1step else 'NOT reduced'})")
    gate_bites = g_fr_multi < 0.9 * g_fr_1step

    # PLAN with the multi-step predictor (learned cost), success@tol + final, over seeds
    mppi_cfg = MPPIConfig(horizon=horizon, n_samples=n_samples, n_elites=16, n_iters=n_iters,
                          init_std=0.25, cumulative=True)
    seed_list = [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip()]
    jepa.predictor = pred_multi
    succ, fin, err_traj = [], [], []
    for sd in seed_list:
        _, su, fi, et = plan_and_collect(jepa, head, sim, state_enc, K, tol, n_episodes, mpc_steps,
                                         mppi_cfg, sd, device)
        succ.append(su); fin.append(fi); err_traj.append(et)
    succ_m = float(np.mean(succ)); fin_m = float(np.mean(fin))
    crossed = succ_m > 0.0
    print(f"\n[multistep PLAN, {len(seed_list)} seeds] success={succ_m:.3f} ± {np.std(succ):.3f} "
          f"final={fin_m:.3f}  (oracle 0.79, learned-cost 1-step baseline ~3.06, tol {tol:.3f})")
    verdict = ("MULTI-STEP CLOSES tol -> M3 POSITIVE (fold)" if crossed else
               ("multi-step REDUCES free-running error but does NOT cross tol -> capacity-limited M3 negative"
                if gate_bites else
                "multi-step training cannot even reduce free-running error -> representational/capacity limit"))
    print(f"VERDICT: {verdict}")

    res = {"k": k, "tol": tol,
           "one_step_err_1steppred": g_1s_1step, "one_step_err_multipred": g_1s_multi,
           "freerun_err_1steppred": g_fr_1step, "freerun_err_multipred": g_fr_multi,
           "gate_bites_freerun_reduced": bool(gate_bites),
           "multistep_plan_success": succ_m, "multistep_plan_final": fin_m,
           "multistep_plan_err_on_traj": float(np.mean(err_traj)),
           "crossed_tol": crossed, "verdict": verdict,
           "baselines": {"oracle": 0.79, "learned_cost_1step": 3.06}}
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "m3_multistep.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {out}/m3_multistep.json")
    return res


if __name__ == "__main__":
    fire.Fire(run)
