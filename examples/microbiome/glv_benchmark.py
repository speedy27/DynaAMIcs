"""
gLV JEPA temporal benchmark — train then evaluate in one script.

Uses the SAME JEPA architecture as examples/microbiome/main.py (tristan branch):
  encoder   = SetEncoder           (permutation-invariant DeepSets)
  predictor = RNNPredictor         (action-conditioned GRU)
  regularizer = VC_IDM_Sim_Regularizer  (VICReg + IDM, the real anti-collapse recipe)
  predcost  = SquareLossSeq        (multi-step prediction energy)

Additional loss against temporal collapse:
  TemporalVarianceLoss  (forces latent to vary along T, not just across batch)

Trained on synthetic gLV trajectories (GLVTrajDataset).
Evaluated with MDSINE2-style hold-one-subject-out CLR-RMSE benchmark vs:
  persistence  |  gLV-L2  |  gLV-net  |  JEPA (ours)

Usage
-----
  # smoke test
  python -m examples.microbiome.glv_benchmark \
      --epochs 5 --n_traj 32 --n_subjects 8 --seeds 0 --horizons 1 5

  # full cluster run
  python -m examples.microbiome.glv_benchmark \
      --epochs 60 --n_traj 512 --n_subjects 30 --seeds 0 1 2 \
      --out /path/results.json --figs /path/figs/ --ckpt_out /path/jepa.pt
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clr(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lx = np.log(np.clip(x, eps, None))
    return lx - lx.mean(axis=-1, keepdims=True)


def _clr_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((_clr(pred) - _clr(true)) ** 2).mean()))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class PersistenceModel:
    def fit(self, states, actions): pass
    def predict_step(self, x, action): return x.copy()


class GLV_L2:
    def __init__(self, alpha=1.0): self.alpha = alpha

    def fit(self, states, actions):
        from sklearn.linear_model import Ridge
        N, T1, S = states.shape
        T = T1 - 1; cs = _clr(states)
        X, Y = [], []
        for i in range(N):
            for t in range(T):
                X.append(np.concatenate([cs[i, t], actions[i, t]]))
                Y.append(cs[i, t + 1])
        self._reg = Ridge(alpha=self.alpha).fit(np.array(X, np.float32), np.array(Y, np.float32))

    def predict_step(self, x, action):
        clr_p = self._reg.predict(np.concatenate([_clr(x), action])[None])[0]
        raw = np.exp(clr_p - clr_p.max())
        return raw / raw.sum() * max(x.sum(), 1e-8)


class GLV_Net:
    def __init__(self, hidden=64, max_iter=1000, alpha=1e-4):
        self.hidden = hidden; self.max_iter = max_iter; self.alpha = alpha

    def fit(self, states, actions):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        N, T1, S = states.shape; T = T1 - 1; cs = _clr(states)
        X, Y = [], []
        for i in range(N):
            for t in range(T):
                X.append(np.concatenate([cs[i, t], actions[i, t]]))
                Y.append(cs[i, t + 1] - cs[i, t])
        X, Y = np.array(X, np.float32), np.array(Y, np.float32)
        self._sc_x = StandardScaler().fit(X); self._sc_y = StandardScaler().fit(Y)
        self._mlp = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.hidden),
            max_iter=self.max_iter, alpha=self.alpha, random_state=0,
        ).fit(self._sc_x.transform(X), self._sc_y.transform(Y))

    def predict_step(self, x, action):
        feat = np.concatenate([_clr(x), action])[None]
        delta = self._sc_y.inverse_transform(self._mlp.predict(self._sc_x.transform(feat)))[0]
        clr_p = _clr(x) + delta
        raw = np.exp(clr_p - clr_p.max())
        return raw / raw.sum() * max(x.sum(), 1e-8)


# ---------------------------------------------------------------------------
# JEPA — build + checkpoint helpers (same architecture as main.py)
# ---------------------------------------------------------------------------

def _build_jepa(K: int, D: int, h_enc: int, device: torch.device):
    """Build JEPA with the canonical microbiome architecture.

    Identical to examples/microbiome/main.py::build_jepa, adapted for arbitrary K / D.
    """
    from eb_jepa.architectures import (
        InverseDynamicsModel, Projector, RNNPredictor, SetEncoder,
    )
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

    encoder   = SetEncoder(emb_dim=384, h_d=h_enc, out_d=D)
    predictor = RNNPredictor(hidden_size=D, action_dim=K, num_layers=1,
                             final_ln=nn.LayerNorm(D))
    idm       = InverseDynamicsModel(state_dim=D, hidden_dim=D * 2, action_dim=K)
    projector = Projector(f"{D}-{D * 4}-{D * 4}")
    reg = VC_IDM_Sim_Regularizer(
        cov_coeff=25.0, std_coeff=10.0, sim_coeff_t=0.0, idm_coeff=1.0,
        idm=idm, projector=projector, first_t_only=False,
    )
    predcost = SquareLossSeq()
    return JEPA(encoder, nn.Identity(), predictor, reg, predcost).to(device)


def _ckpt_save(blob: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    print(f"  checkpoint -> {path}")


# ---------------------------------------------------------------------------
# JEPA predictor for the benchmark phase
# ---------------------------------------------------------------------------

class JEPAPredictor:
    """Wraps a trained gLV JEPA for single-step rollout in the temporal benchmark.

    Encoding pipeline per step:
      x [S] -> relative-CLR -> concat species_emb -> z-score -> [1, F, 1, S, 1]
      -> jepa.encoder -> latent z [1, D, 1, 1, 1]
    Single-step dynamics:
      (z_t, action_t) -> jepa.predictor -> z_{t+1} -> linear Ridge readout -> x
    """

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        blob = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = blob["build_cfg"]

        self._jepa = _build_jepa(cfg["K"], cfg["D"], cfg["h_enc"], self.device)
        self._jepa.load_state_dict(blob["jepa_state"])
        self._jepa.eval()

        self._species_emb = blob["species_emb"].to(self.device)   # [S, 384]
        self._zscore_mean = blob["zscore_mean"].to(self.device)    # [F=385]
        self._zscore_std  = blob["zscore_std"].to(self.device)     # [F=385]
        self._readout: Optional[object] = None
        self._sc: Optional[object] = None
        self._z_state: Optional[torch.Tensor] = None  # carries latent across steps

    def _tokenize(self, x: np.ndarray) -> torch.Tensor:
        """x: [S] raw abundance -> obs [1, F, 1, S, 1]"""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        rel  = x_t / x_t.sum().clamp_min(1e-12)
        log_ab = torch.log(rel.clamp_min(1e-8))
        clr_ab = log_ab - log_ab.mean()                                        # [S]
        tok = torch.cat([self._species_emb, clr_ab.unsqueeze(-1)], dim=-1)    # [S, F]
        tok = (tok - self._zscore_mean) / self._zscore_std.clamp_min(1e-6)    # [S, F]
        # [S, F] -> [F, S] -> [1, F, 1, S, 1]   (B=1, C=F, T=1, H=S, W=1)
        return tok.T.unsqueeze(0).unsqueeze(2).unsqueeze(-1)

    @torch.no_grad()
    def _encode(self, x: np.ndarray) -> np.ndarray:
        obs = self._tokenize(x)                     # [1, F, 1, S, 1]
        z   = self._jepa.encoder(obs)               # [1, D, 1, 1, 1]
        return z[0, :, 0, 0, 0].cpu().numpy()

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        N, T1, S = states.shape
        Z, Y = [], []
        for i in range(N):
            for t in range(T1):
                Z.append(self._encode(states[i, t]))
                Y.append(_clr(states[i, t]))
        Z = np.array(Z, np.float32); Y = np.array(Y, np.float32)
        self._sc = StandardScaler().fit(Z)
        self._readout = Ridge(alpha=1.0).fit(self._sc.transform(Z), Y)
        self._z_state = None

    def reset(self) -> None:
        """Reset latent state — call at the start of each new trajectory."""
        self._z_state = None

    @torch.no_grad()
    def predict_step(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        a = torch.tensor(action, dtype=torch.float32, device=self.device
                         ).unsqueeze(0).unsqueeze(-1)    # [1, K, 1]

        # First step: encode the true observation to get z_0.
        # Subsequent steps: propagate z in latent space directly — no re-encoding.
        if self._z_state is None:
            obs = self._tokenize(x)                  # [1, F, 1, S, 1]
            self._z_state = self._jepa.encoder(obs)  # [1, D, 1, 1, 1]

        z_next = self._jepa.predictor(self._z_state, a)  # [1, D, 1, 1, 1]
        self._z_state = z_next                            # carry forward in latent space

        z_np  = z_next[0, :, 0, 0, 0].cpu().numpy()
        clr_p = self._readout.predict(self._sc.transform(z_np[None]))[0]
        raw   = np.exp(clr_p - clr_p.max())
        return raw / raw.sum()


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

def _rollout_errors(model, states: np.ndarray, actions: np.ndarray, horizons):
    T = states.shape[0] - 1
    max_h = max(horizons)
    errors = {h: [] for h in horizons}
    for t0 in range(T - max_h + 1):
        if hasattr(model, "reset"):
            model.reset()
        x_cur = states[t0].copy()
        preds = [x_cur]
        for k in range(max_h):
            idx = t0 + k
            a = actions[idx] if idx < actions.shape[0] else np.zeros_like(actions[0])
            preds.append(model.predict_step(x_cur, a))
            x_cur = preds[-1]
        for h in horizons:
            errors[h].append(_clr_rmse(preds[h], states[t0 + h]))
    return errors


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jepa(args, device: torch.device) -> dict:
    from eb_jepa.losses import TemporalVarianceLoss
    from eb_jepa.datasets.microbiome.traj import (
        GLVTrajConfig, GLVTrajDataset, init_microbiome_traj_data)
    from eb_jepa.schedulers import CosineWithWarmup

    D, K_arg = args.d_model, args.n_candidate
    W = args.n_window

    print(f"\n{'='*60}")
    print(f"JEPA Training  (canonical microbiome architecture)")
    print(f"  n_traj={args.n_traj}  T={args.T_train}  W={W}  epochs={args.epochs}")
    print(f"  SetEncoder(384->h{args.h_enc}->D{D}) + RNN + IDM + VICReg + TemporalVar")
    print(f"  device={device}")
    print(f"{'='*60}")

    # ── data ──────────────────────────────────────────────────────────────
    train_loader, _, dl_cfg, _ = init_microbiome_traj_data(
        cfg_data=dict(
            n_traj=args.n_traj,
            T=args.T_train,
            n_species=args.n_species,
            n_candidate=args.n_candidate,
            action_policy="random",
            sim_seed=0, emb_seed=0,
            batch_size=args.batch_size,
            num_workers=0,
            num_frames=W,
            frameskip=1,
            train_fraction=0.9,
        ),
        device=None,
    )
    K = dl_cfg.action_dim

    # Reference dataset for zscore + species embeddings (same seeds)
    ref_ds = GLVTrajDataset(GLVTrajConfig(
        n_traj=args.n_traj, T=args.T_train,
        n_species=args.n_species, n_candidate=args.n_candidate,
        sim_seed=0, emb_seed=0,
    ))
    species_emb = ref_ds._species_emb    # [S, 384]
    zscore_mean = ref_ds.zscore.mean     # [F=385]
    zscore_std  = ref_ds.zscore.std      # [F=385]

    # ── model ─────────────────────────────────────────────────────────────
    jepa     = _build_jepa(K, D, args.h_enc, device)
    tvar_loss = TemporalVarianceLoss(margin=1.0).to(device)

    params = list(jepa.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = CosineWithWarmup(opt, total_steps=total_steps, warmup_ratio=0.1,
                             min_lr=args.lr * 0.01)

    log_every = max(1, args.epochs // 10)
    nsteps = W - 1   # predict W-1 steps autoregressively from context step 0

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        jepa.train()
        agg = {"total": 0.0, "pred": 0.0, "reg": 0.0, "tvar": 0.0, "n": 0}

        for batch in train_loader:
            if batch is None: continue
            obs, act, *_ = batch
            # obs["otu"]: [B, W, N, F]   act: [B, W, K]
            otu   = obs["otu"].to(device)   # [B, W, N, F]
            act_t = act.to(device)          # [B, W, K]
            B, WW, N, Fv = otu.shape

            # Convert to JEPA 5-D format
            obs_5d   = otu.permute(0, 3, 1, 2).unsqueeze(-1)   # [B, F, W, N, 1]
            actions_t = act_t.permute(0, 2, 1)                  # [B, K, W]

            # JEPA.unroll: encode + VC_IDM regularizer + autoregressive prediction
            _, (loss, rloss, _, _, ploss) = jepa.unroll(
                obs_5d, actions_t, nsteps=nsteps,
                unroll_mode="autoregressive", compute_loss=True,
            )

            # Extra TemporalVarianceLoss (anti temporal-collapse)
            state = jepa.encoder(obs_5d)            # [B, D, W, 1, 1]
            l_tvar = tvar_loss(state)
            total = loss + args.tvar_coeff * l_tvar

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()

            b = otu.shape[0]
            ploss_v = ploss.item() if torch.is_tensor(ploss) else float(ploss or 0)
            agg["total"] += total.item() * b; agg["reg"]  += rloss.item() * b
            agg["pred"]  += ploss_v * b;      agg["tvar"] += l_tvar.item() * b
            agg["n"] += b

        if epoch % log_every == 0 or epoch == args.epochs:
            n = max(1, agg["n"])
            tvar_v = state[..., 0, 0].var(dim=2).mean().item()
            print(f"  epoch {epoch:3d}/{args.epochs}"
                  f"  total={agg['total']/n:.4f}"
                  f"  pred={agg['pred']/n:.4f}"
                  f"  reg={agg['reg']/n:.4f}"
                  f"  tvarL={agg['tvar']/n:.4f}"
                  f"  tvar(monitor)={tvar_v:.4f}")

    # ── checkpoint ────────────────────────────────────────────────────────
    blob = {
        "jepa_state":   jepa.cpu().state_dict(),
        "build_cfg":    {"K": K, "D": D, "h_enc": args.h_enc},
        "species_emb":  species_emb,
        "zscore_mean":  zscore_mean,
        "zscore_std":   zscore_std,
    }
    if args.ckpt_out:
        _ckpt_save(blob, args.ckpt_out)

    return blob


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args, jepa_blob: dict, device: torch.device) -> dict:
    from eb_jepa.datasets.microbiome.glv import GLVConfig, GLVSimulator

    horizons   = args.horizons
    model_names = ["persistence", "gLV-L2", "gLV-net", "JEPA (ours)"]
    all_results: dict = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Benchmark  seed={seed}  |  {args.n_subjects} subjects  T={args.T}")
        print(f"{'='*60}")

        glv  = GLVSimulator(GLVConfig(n_species=args.n_species,
                                      n_candidate=args.n_candidate, seed=42))
        data = glv.generate_trajectories(n=args.n_subjects, T=args.T,
                                         action_policy="random", seed=seed)
        states  = data["states"]    # [N, T+1, S]
        actions = data["actions"]   # [N, T,   K]

        # Materialise JEPA predictor via tmp checkpoint file
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(jepa_blob, tmp.name); tmp.close()
        jepa_pred = JEPAPredictor(tmp.name, device=str(device))
        os.unlink(tmp.name)

        fold_errors = {name: {h: [] for h in horizons} for name in model_names}

        for s in range(args.n_subjects):
            tr_idx = [i for i in range(args.n_subjects) if i != s]
            tr_s, tr_a = states[tr_idx], actions[tr_idx]
            ho_s, ho_a = states[s],      actions[s]

            models = {
                "persistence": PersistenceModel(),
                "gLV-L2":      GLV_L2(),
                "gLV-net":     GLV_Net(),
                "JEPA (ours)": jepa_pred,
            }
            for name, model in models.items():
                t0 = time.time()
                model.fit(tr_s, tr_a)
                errs = _rollout_errors(model, ho_s, ho_a, horizons)
                for h in horizons:
                    fold_errors[name][h].extend(errs[h])
                if s == 0:
                    h1 = np.mean(fold_errors[name][horizons[0]])
                    print(f"  [{name:14s}]  fold-0  {time.time()-t0:.1f}s  "
                          f"h={horizons[0]} RMSE={h1:.4f}")

        seed_res = {
            name: {h: {"mean": float(np.mean(fold_errors[name][h])),
                        "std":  float(np.std (fold_errors[name][h]))}
                   for h in horizons}
            for name in model_names
        }
        all_results[str(seed)] = seed_res
        _print_table(seed_res, horizons, args.n_subjects)

    summary = _aggregate(all_results, horizons, model_names)
    print(f"\n{'='*60}\nSUMMARY  (mean over {len(args.seeds)} seed(s))\n{'='*60}")
    _print_table(summary, horizons, args.n_subjects)
    _print_skill(summary, horizons)
    return summary


def _print_table(results: dict, horizons, n_subjects: int):
    col_w = 18
    hdr = f"{'model':17s}|" + "|".join(f"{'h='+str(h):^{col_w}}" for h in horizons)
    print(f"\nCLR-RMSE  --  {n_subjects} subjects, HOSO CV")
    print(hdr); print("-" * len(hdr))
    for name, data in results.items():
        cells = [f"{data[h]['mean']:.4f}+-{data[h]['std']:.4f}" for h in horizons]
        print(f"{name:17s}|" + "|".join(f"{c:^{col_w}}" for c in cells))


def _print_skill(summary: dict, horizons):
    pers = {h: summary["persistence"][h]["mean"] for h in horizons}
    print("\nSkill vs persistence (pers_RMSE / model_RMSE,  >1 beats no-change):")
    for name, data in summary.items():
        if name == "persistence": continue
        skills = [f"h={h}: {pers[h]/max(data[h]['mean'], 1e-9):.3f}x" for h in horizons]
        print(f"  {name:16s}: " + "  ".join(skills))


def _aggregate(all_results: dict, horizons, model_names) -> dict:
    buf = {n: {h: [] for h in horizons} for n in model_names}
    for sr in all_results.values():
        for n in model_names:
            if n not in sr: continue
            for h in horizons:
                buf[n][h].append(sr[n][h]["mean"])
    return {
        n: {h: {"mean": float(np.mean(buf[n][h])),
                "std":  float(np.std (buf[n][h]))}
            for h in horizons}
        for n in model_names
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figures(summary: dict, horizons, n_subjects: int, seeds, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    COLORS  = {"persistence": "#aaaaaa", "gLV-L2": "#2196F3",
                "gLV-net": "#FF9800",    "JEPA (ours)": "#E91E63"}
    MARKERS = {"persistence": "s", "gLV-L2": "o", "gLV-net": "^", "JEPA (ours)": "D"}
    models  = list(summary.keys())

    # ── Fig 1: CLR-RMSE vs horizon ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in models:
        ys = [summary[name][h]["mean"] for h in horizons]
        es = [summary[name][h]["std"]  for h in horizons]
        lw = 2.8 if name == "JEPA (ours)" else 1.8
        ax.errorbar(horizons, ys, yerr=es, label=name,
                    color=COLORS.get(name, "#555"), marker=MARKERS.get(name, "o"),
                    linewidth=lw, markersize=7, capsize=4,
                    zorder=5 if name == "JEPA (ours)" else 3)
    ax.set_xlabel("Prediction horizon (steps)", fontsize=12)
    ax.set_ylabel("CLR-RMSE (lower is better)", fontsize=12)
    ax.set_title(
        f"Temporal benchmark -- MDSINE2 protocol\n"
        f"gLV synthetic ({n_subjects} subjects, HOSO, {len(seeds)} seed(s))", fontsize=11)
    ax.set_xticks(horizons); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = f"{out_dir}/glv_benchmark_rmse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")

    # ── Fig 2: Skill vs persistence ─────────────────────────────────────────
    pers = [summary["persistence"][h]["mean"] for h in horizons]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in models:
        if name == "persistence": continue
        skills = [pers[i] / max(summary[name][h]["mean"], 1e-9) for i, h in enumerate(horizons)]
        lw = 2.8 if name == "JEPA (ours)" else 1.8
        ax.plot(horizons, skills, label=name,
                color=COLORS.get(name, "#555"), marker=MARKERS.get(name, "o"),
                linewidth=lw, markersize=7,
                zorder=5 if name == "JEPA (ours)" else 3)
    ax.axhline(1.0, color="#bbb", linestyle="--", linewidth=1.5, label="Persistence (=1)")
    ax.set_xlabel("Prediction horizon (steps)", fontsize=12)
    ax.set_ylabel("Skill vs persistence (>1 beats no-change)", fontsize=11)
    ax.set_title(
        f"Temporal skill -- MDSINE2 protocol\n"
        f"gLV synthetic ({n_subjects} subjects, HOSO, {len(seeds)} seed(s))", fontsize=11)
    ax.set_xticks(horizons)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = f"{out_dir}/glv_benchmark_skill.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")

    # ── Fig 3: Grouped bars per horizon ─────────────────────────────────────
    non_pers = [m for m in models if m != "persistence"]
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.2 * len(horizons), 4.2))
    if len(horizons) == 1: axes = [axes]
    for ax, h in zip(axes, horizons):
        for xi, name in enumerate(non_pers):
            v = summary[name][h]["mean"]; e = summary[name][h]["std"]
            ec = "black" if name == "JEPA (ours)" else "none"
            lw = 1.5 if name == "JEPA (ours)" else 0
            ax.bar(xi, v, yerr=e, color=COLORS.get(name, "#888"), capsize=5,
                   width=0.6, alpha=0.85, edgecolor=ec, linewidth=lw)
        ax.axhline(summary["persistence"][h]["mean"],
                   color="#aaa", linestyle="--", linewidth=1.5, label="Persistence")
        ax.set_xticks(range(len(non_pers)))
        ax.set_xticklabels([m.split(" ")[0] for m in non_pers], rotation=25, ha="right", fontsize=9)
        ax.set_title(f"h = {h}", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("CLR-RMSE", fontsize=10); ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"CLR-RMSE per horizon  ({n_subjects} subjects, HOSO)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = f"{out_dir}/glv_benchmark_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")

    # ── Fig 4: JEPA vs gLV-net with % improvement ────────────────────────────
    net_m  = [summary["gLV-net"][h]["mean"]     for h in horizons]
    jep_m  = [summary["JEPA (ours)"][h]["mean"] for h in horizons]
    pers_m = [summary["persistence"][h]["mean"] for h in horizons]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    w = 0.32; xs = np.arange(len(horizons))
    ax.bar(xs - w/2, net_m, w, label="gLV-net",     color="#FF9800", alpha=0.85)
    ax.bar(xs + w/2, jep_m, w, label="JEPA (ours)", color="#E91E63", alpha=0.85,
           edgecolor="black", linewidth=1.2)
    for i, (nv, jv, h) in enumerate(zip(net_m, jep_m, horizons)):
        gain = 100.0 * (nv - jv) / max(nv, 1e-9)
        sign = "v" if gain > 0 else "^"
        col  = "#2e7d32" if gain > 0 else "#c62828"
        ax.annotate(f"{sign}{abs(gain):.1f}%", xy=(i + w/2, jv),
                    ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
    ax.plot(xs, pers_m, color="#aaa", linestyle="--", marker="s",
            linewidth=1.5, markersize=5, label="Persistence")
    ax.set_xticks(xs); ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylabel("CLR-RMSE (lower is better)", fontsize=11)
    ax.set_title("JEPA vs strongest baseline (gLV-net)\n% improvement annotated", fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = f"{out_dir}/glv_benchmark_jepa_vs_net.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_species",   type=int,   default=32)
    p.add_argument("--n_candidate", type=int,   default=8)
    p.add_argument("--n_traj",      type=int,   default=256,  help="Training trajectories")
    p.add_argument("--T_train",     type=int,   default=40,   help="Steps per training traj")
    p.add_argument("--n_window",    type=int,   default=8,    help="Context window W")
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--d_model",     type=int,   default=128,  help="Latent state dim D")
    p.add_argument("--h_enc",       type=int,   default=256,  help="SetEncoder hidden dim")
    p.add_argument("--tvar_coeff",  type=float, default=1.0,  help="TemporalVarianceLoss weight")
    p.add_argument("--ckpt_out",    type=str,   default=None)
    p.add_argument("--n_subjects",  type=int,   default=30)
    p.add_argument("--T",           type=int,   default=60,   help="Benchmark traj length")
    p.add_argument("--seeds",       type=int,   nargs="+",    default=[0, 1, 2])
    p.add_argument("--horizons",    type=int,   nargs="+",    default=[1, 3, 5, 10])
    p.add_argument("--out",         type=str,   default=None)
    p.add_argument("--figs",        type=str,   default=None)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    jepa_blob = train_jepa(args, device)
    summary   = run_benchmark(args, jepa_blob, device)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "summary": summary,
            "config": {"n_subjects": args.n_subjects, "T": args.T,
                       "horizons": args.horizons, "seeds": args.seeds,
                       "epochs": args.epochs, "n_traj": args.n_traj,
                       "d_model": args.d_model, "h_enc": args.h_enc,
                       "tvar_coeff": args.tvar_coeff},
        }, open(args.out, "w"), indent=2)
        print(f"\nResults -> {args.out}")

    if args.figs:
        print(f"\nGenerating figures -> {args.figs}/")
        save_figures(summary, args.horizons, args.n_subjects, args.seeds, args.figs)


if __name__ == "__main__":
    main()
