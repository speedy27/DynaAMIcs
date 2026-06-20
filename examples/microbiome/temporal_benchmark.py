"""
Temporal benchmark — MDSINE2-style protocol on gLV synthetic trajectories.

Measures how well each model predicts future community states given past states
and interventions, using a hold-one-subject-out cross-validation scheme.

Protocol (faithful to MDSINE2 / Jones et al.)
----------------------------------------------
1. Generate N_SUBJECTS gLV synthetic subjects (each = one trajectory of T steps).
2. For each held-out subject s:
     - Fit/train each baseline on the remaining N-1 subjects.
     - Starting from the true initial window of the held-out subject, roll out each
       model autoregressively for K_STEPS steps.
     - Report CLR-RMSE at horizons 1, 3, 5, 10 steps.
3. Aggregate: mean ± std CLR-RMSE across subjects.

Metrics (all in CLR space, matching MDSINE2 log-abundance reporting)
--------------------------------------------------------------------
- CLR-RMSE(k): RMSE between predicted CLR(x_{t+k}) and true CLR(x_{t+k}),
  averaged over all rollout start-points in the held-out trajectory.

Baselines
---------
persistence   x_{t+1} = x_t (the "no-change" lower bound every model must beat)
gLV-L2        Ridge regression: CLR(x_{t+1}) = W @ [CLR(x_t), action_t] + b
              (linear GLV approximation, trained per fold on all other subjects)
gLV-net       2-layer MLP (64 hidden units) on same input → same output
              (nonlinear, captures interaction terms missed by the linear model)
JEPA*         Trained RNNPredictor rollout in latent space + linear readout → CLR
              (* optional, requires --ckpt)

Usage
-----
  # baselines only (no GPU, no checkpoint needed):
  python -m examples.microbiome.temporal_benchmark

  # with JEPA predictor:
  python -m examples.microbiome.temporal_benchmark --ckpt artifacts/ckpt/microbiome_jepa.pt

  # larger sweep:
  python -m examples.microbiome.temporal_benchmark --n_subjects 40 --T 60 --seeds 0 1 2

  # save results:
  python -m examples.microbiome.temporal_benchmark --out results/temporal_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# CLR transform helpers
# ---------------------------------------------------------------------------

def _clr(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Centered log-ratio transform on last axis (species axis)."""
    lx = np.log(np.clip(x, eps, None))
    return lx - lx.mean(axis=-1, keepdims=True)


def _clr_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error in CLR space. Both inputs: [..., S]."""
    p, t = _clr(pred), _clr(true)
    return float(np.sqrt(((p - t) ** 2).mean()))


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

class PersistenceModel:
    """x_{t+1} = x_t (zero-change baseline)."""

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        pass

    def predict_step(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        return x.copy()


class GLV_L2:
    """Ridge regression: CLR(x_{t+1}) = W @ [CLR(x_t), action_t] + b."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._reg = None

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        """
        states:  [N, T+1, S]
        actions: [N, T,   K]
        """
        from sklearn.linear_model import Ridge

        N, T1, S = states.shape
        T = T1 - 1
        clr_s = _clr(states)  # [N, T+1, S]

        X_list, Y_list = [], []
        for i in range(N):
            for t in range(T):
                x_in = np.concatenate([clr_s[i, t], actions[i, t]])  # [S+K]
                y_out = clr_s[i, t + 1]                              # [S]
                X_list.append(x_in)
                Y_list.append(y_out)

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)
        self._reg = Ridge(alpha=self.alpha, fit_intercept=True).fit(X, Y)
        self._S = S

    def predict_step(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        clr_x = _clr(x)
        feat = np.concatenate([clr_x, action])[None]      # [1, S+K]
        clr_pred = self._reg.predict(feat)[0]             # [S]
        # invert CLR: exp(clr_pred), renormalize to positive simplex
        raw = np.exp(clr_pred - clr_pred.max())
        return raw / raw.sum() * max(x.sum(), 1e-8)


class GLV_Net:
    """2-layer MLP: (CLR(x_t), action_t) → Δ CLR(x_t). Trained with sklearn MLPRegressor."""

    def __init__(self, hidden: int = 64, max_iter: int = 1000, alpha: float = 1e-4):
        self.hidden = hidden
        self.max_iter = max_iter
        self.alpha = alpha
        self._mlp = None

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        N, T1, S = states.shape
        T = T1 - 1
        clr_s = _clr(states)

        X_list, Y_list = [], []
        for i in range(N):
            for t in range(T):
                x_in = np.concatenate([clr_s[i, t], actions[i, t]])
                # predict delta CLR (easier to learn than absolute)
                y_out = clr_s[i, t + 1] - clr_s[i, t]
                X_list.append(x_in)
                Y_list.append(y_out)

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)
        self._sc_x = StandardScaler().fit(X)
        self._sc_y = StandardScaler().fit(Y)
        Xs, Ys = self._sc_x.transform(X), self._sc_y.transform(Y)
        self._mlp = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.hidden),
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=0,
        ).fit(Xs, Ys)
        self._S = S

    def predict_step(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        clr_x = _clr(x)
        feat = np.concatenate([clr_x, action])[None]
        feat_s = self._sc_x.transform(feat)
        delta_s = self._mlp.predict(feat_s)
        delta = self._sc_y.inverse_transform(delta_s)[0]
        clr_pred = clr_x + delta
        raw = np.exp(clr_pred - clr_pred.max())
        return raw / raw.sum() * max(x.sum(), 1e-8)


# ---------------------------------------------------------------------------
# JEPA predictor baseline (optional, requires checkpoint)
# ---------------------------------------------------------------------------

class JEPAPredictor:
    """Trained JEPA: encode x_t → z_t, roll out predictor, linear readout → CLR."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        import torch
        self.device = torch.device(device)
        self._load(ckpt_path)

    def _load(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._enc = ckpt["encoder"].to(self.device).eval()
        self._pred = ckpt["predictor"].to(self.device).eval()
        self._cfg = ckpt.get("cfg", {})
        self._readout = None  # fitted in fit()
        self._sc = None

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Fit linear CLR readout from latent z → CLR(x) on training subjects."""
        import torch
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        N, T1, S = states.shape
        T = T1 - 1
        clr_s = _clr(states)

        Z_list, Y_list = [], []
        with torch.no_grad():
            for i in range(N):
                x_t = torch.tensor(states[i], dtype=torch.float32)  # [T+1, S]
                # reshape to [1, S, T+1, 1, 1] — SetEncoder contract
                obs = x_t.T.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)
                z = self._enc(obs)                     # [1, D, T+1, 1, 1]
                z_np = z[0, :, :, 0, 0].T.cpu().numpy()  # [T+1, D]
                for t in range(T1):
                    Z_list.append(z_np[t])
                    Y_list.append(clr_s[i, t])

        Z = np.array(Z_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)
        self._sc = StandardScaler().fit(Z)
        self._readout = Ridge(alpha=1.0).fit(self._sc.transform(Z), Y)
        self._S = S

    def _encode_x(self, x: np.ndarray):
        import torch
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, S]
        # [1, S, 1, 1, 1]
        obs = x_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            z = self._enc(obs)  # [1, D, 1, 1, 1]
        return z[0, :, 0, 0, 0].cpu().numpy()  # [D]

    def _decode_z(self, z: np.ndarray) -> np.ndarray:
        clr_pred = self._readout.predict(self._sc.transform(z[None]))[0]
        raw = np.exp(clr_pred - clr_pred.max())
        return raw / raw.sum()

    def predict_step(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        import torch
        z = self._encode_x(x)  # [D]
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, K]
        z_t = torch.tensor(z, dtype=torch.float32).reshape(1, -1, 1, 1, 1).to(self.device)
        with torch.no_grad():
            z_pred = self._pred(z_t, a)  # [1, D, 1, 1, 1] or similar
            z_pred_np = z_pred.reshape(-1).cpu().numpy()
        return self._decode_z(z_pred_np)


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

def _rollout_errors(model, states: np.ndarray, actions: np.ndarray,
                    horizons: list[int]) -> dict[int, list[float]]:
    """
    Roll out `model` from each valid start in the trajectory and collect CLR-RMSE
    at each horizon. Returns {horizon: [rmse_per_start]}.
    """
    T = states.shape[0] - 1   # number of transitions
    max_h = max(horizons)
    errors: dict[int, list[float]] = {h: [] for h in horizons}

    for t_start in range(T - max_h + 1):
        x_cur = states[t_start].copy()
        preds = [x_cur]
        for k in range(max_h):
            a = actions[t_start + k] if t_start + k < actions.shape[0] else np.zeros_like(actions[0])
            x_next = model.predict_step(x_cur, a)
            preds.append(x_next)
            x_cur = x_next

        for h in horizons:
            errors[h].append(_clr_rmse(preds[h], states[t_start + h]))

    return errors


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    n_subjects: int = 20,
    T: int = 50,
    glv_seed: int = 42,
    seeds: list[int] | None = None,
    horizons: list[int] | None = None,
    ckpt: str | None = None,
    device: str = "cpu",
    out: str | None = None,
    verbose: bool = True,
) -> dict:
    from eb_jepa.datasets.microbiome.glv import GLVConfig, GLVSimulator

    if seeds is None:
        seeds = [0]
    if horizons is None:
        horizons = [1, 3, 5, 10]

    all_results = {}

    for seed in seeds:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Seed {seed} | {n_subjects} subjects, T={T} steps each")
            print(f"{'='*60}")

        cfg = GLVConfig(n_species=32, n_candidate=8, seed=glv_seed)
        sim = GLVSimulator(cfg)
        batch = sim.generate_trajectories(n=n_subjects, T=T,
                                          action_policy="random", seed=seed)
        states  = batch["states"]   # [N, T+1, S]
        actions = batch["actions"]  # [N, T,   K]

        models: dict[str, object] = {
            "persistence": PersistenceModel(),
            "gLV-L2":      GLV_L2(alpha=1.0),
            "gLV-net":     GLV_Net(hidden=64, max_iter=300),
        }
        if ckpt is not None:
            try:
                models["JEPA"] = JEPAPredictor(ckpt, device=device)
            except Exception as e:
                print(f"[WARN] Could not load JEPA checkpoint: {e}", file=sys.stderr)

        # hold-one-subject-out
        fold_errors: dict[str, dict[int, list[float]]] = {
            name: {h: [] for h in horizons} for name in models
        }

        for s in range(n_subjects):
            train_idx = [i for i in range(n_subjects) if i != s]
            tr_states  = states[train_idx]
            tr_actions = actions[train_idx]
            ho_states  = states[s]   # [T+1, S]
            ho_actions = actions[s]  # [T,   K]

            for name, model in models.items():
                t0 = time.time()
                model.fit(tr_states, tr_actions)
                errs = _rollout_errors(model, ho_states, ho_actions, horizons)
                for h in horizons:
                    fold_errors[name][h].extend(errs[h])
                if verbose and s == 0:
                    elapsed = time.time() - t0
                    print(f"  [{name:12s}] fit+eval fold 0: {elapsed:.1f}s")

        # aggregate
        seed_result: dict[str, dict] = {}
        for name in models:
            seed_result[name] = {}
            for h in horizons:
                vals = fold_errors[name][h]
                seed_result[name][h] = {"mean": float(np.mean(vals)),
                                        "std":  float(np.std(vals))}

        all_results[str(seed)] = seed_result

        if verbose:
            _print_table(seed_result, horizons, n_subjects)

    # aggregate across seeds
    summary = _aggregate_seeds(all_results, horizons, list(models.keys()))

    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY (mean ± std across {len(seeds)} seed(s))")
        print(f"{'='*60}")
        _print_table(summary, horizons, n_subjects, is_summary=True)

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"per_seed": all_results, "summary": summary,
                       "config": {"n_subjects": n_subjects, "T": T,
                                  "horizons": horizons, "seeds": seeds}}, f, indent=2)
        print(f"\nResults saved to {out}")

    return summary


def _print_table(results: dict, horizons: list[int], n_subjects: int,
                 is_summary: bool = False) -> None:
    col_w = 14
    h_labels = [f"h={h}" for h in horizons]
    header = f"{'model':12s} | " + " | ".join(f"{h:^{col_w}}" for h in h_labels)
    print(f"\nCLR-RMSE (mean ± std, {n_subjects} subjects, hold-one-subject-out)")
    print(header)
    print("-" * len(header))
    for name, data in results.items():
        row = f"{name:12s} | "
        cells = []
        for h in horizons:
            m = data[h]["mean"]
            s = data[h]["std"]
            cells.append(f"{m:.4f}±{s:.4f}")
        row += " | ".join(f"{c:^{col_w}}" for c in cells)
        print(row)


def _aggregate_seeds(all_results: dict, horizons: list[int],
                     model_names: list[str]) -> dict:
    """Average mean/std across seeds."""
    agg = {name: {h: {"mean": [], "std": []} for h in horizons}
           for name in model_names}
    for seed_res in all_results.values():
        for name in model_names:
            if name not in seed_res:
                continue
            for h in horizons:
                agg[name][h]["mean"].append(seed_res[name][h]["mean"])
                agg[name][h]["std"].append(seed_res[name][h]["std"])

    summary = {}
    for name in model_names:
        summary[name] = {}
        for h in horizons:
            means = agg[name][h]["mean"]
            stds  = agg[name][h]["std"]
            summary[name][h] = {
                "mean": float(np.mean(means)),
                "std":  float(np.mean(stds)),  # avg within-fold std
                "seed_std": float(np.std(means)),  # across-seed variance
            }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_subjects", type=int, default=20,
                   help="Number of synthetic gLV subjects (default 20)")
    p.add_argument("--T", type=int, default=50,
                   help="Trajectory length in steps (default 50)")
    p.add_argument("--glv_seed", type=int, default=42,
                   help="Seed for gLV simulator construction (default 42)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Data seeds for the trajectory batch (default [0])")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10],
                   help="Prediction horizons in steps (default 1 3 5 10)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to microbiome JEPA checkpoint (optional)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device for JEPA inference (default cpu)")
    p.add_argument("--out", type=str, default=None,
                   help="Save results JSON to this path")
    args = p.parse_args()

    run_benchmark(
        n_subjects=args.n_subjects,
        T=args.T,
        glv_seed=args.glv_seed,
        seeds=args.seeds,
        horizons=args.horizons,
        ckpt=args.ckpt,
        device=args.device,
        out=args.out,
    )


if __name__ == "__main__":
    main()
