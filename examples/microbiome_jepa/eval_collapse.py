"""
Collapse metric for the Layer-B IDM ablation (the headline analysis).

Sobal et al. (arXiv 2211.10831): JEPAs preferentially encode SLOW features. On gLV trajectories the
slow feature is the trajectory's static identity (its initial composition / basin); the FAST signal is
the time-varying community state and its change under actions. Collapse onto slow features =
high slow-decodability, low fast-decodability. The IDM term should RESTORE fast decodability.

We freeze the trained encoder, encode held-out gLV trajectories, and fit linear probes (Ridge),
splitting BY TRAJECTORY (no timepoint leakage):
  fast_r2_state : z_t            -> x_t            (current abundances)        higher = better (dynamics)
  fast_r2_delta : [z_t, z_{t+1}] -> x_{t+1} - x_t  (one-step change)           higher = better (dynamics)
  slow_r2_init  : z_t            -> x_0            (trajectory initial state)  higher = more slow-feature
  feat_std      : mean per-dim std of z across all (traj,t)                    -> 0 = representational collapse
Headline contrast: fast_r2_delta (and fast_r2_state) should be HIGHER with IDM on; slow_r2_init
should not need IDM. A collapsed (idm-off) model shows fast << slow.

This module is import-safe (no training side effects) and reused by run_ablation.py.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


@torch.no_grad()
def encode_trajectories(encoder, dataset, device, max_traj=None):
    """Encode every (full) gLV trajectory in `dataset` to latents.

    dataset[i] -> (obs{"otu":[T,N,F],"mask":[T,N]}, act[T,K], state[T,S], reward, extra)
    Returns numpy arrays: Z [N,T,D], states [N,T,S], actions [N,T,K].
    """
    encoder.eval()
    n = len(dataset) if max_traj is None else min(max_traj, len(dataset))
    otu = torch.stack([dataset[i][0]["otu"] for i in range(n)]).to(device)   # [N,T,Nmax,F]
    mask = torch.stack([dataset[i][0]["mask"] for i in range(n)]).to(device)  # [N,T,Nmax]
    states = torch.stack([dataset[i][2] for i in range(n)]).cpu().numpy()      # [N,T,S]
    actions = torch.stack([dataset[i][1] for i in range(n)]).cpu().numpy()     # [N,T,K]

    state = encoder({"otu": otu, "mask": mask})  # [N, D, T, 1, 1]
    Z = state.squeeze(-1).squeeze(-1).permute(0, 2, 1).float().cpu().numpy()  # [N, T, D]
    return Z, states, actions


def _ridge_r2(X_tr, y_tr, X_te, y_te, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return float(r2_score(y_te, model.predict(X_te), multioutput="uniform_average"))


def collapse_probes(Z, states, actions=None, train_frac=0.7, seed=0, alpha=1.0):
    """Fit fast/slow linear probes with a trajectory-level train/test split.

    Z: [N,T,D], states: [N,T,S]. Returns a dict of R^2 values + feat_std.
    """
    N, T, D = Z.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr = max(1, int(round(train_frac * N)))
    tr, te = perm[:n_tr], perm[n_tr:]
    if len(te) == 0:  # tiny-N smoke fallback
        te = tr

    def flat(idx, arr):
        return arr[idx].reshape(len(idx) * arr.shape[1], -1)

    # fast: z_t -> x_t
    fast_r2_state = _ridge_r2(flat(tr, Z), flat(tr, states), flat(te, Z), flat(te, states), alpha)

    # fast: [z_t, z_{t+1}] -> (x_{t+1} - x_t)
    Zcat = np.concatenate([Z[:, :-1], Z[:, 1:]], axis=-1)        # [N,T-1,2D]
    dstate = states[:, 1:] - states[:, :-1]                       # [N,T-1,S]
    fast_r2_delta = _ridge_r2(flat(tr, Zcat), flat(tr, dstate), flat(te, Zcat), flat(te, dstate), alpha)

    # fast: [z_t, z_{t+1}] -> a_t  (the intervention that drove the transition) — MOST DIRECT IDM signal.
    # A FRESH linear probe (not the trained IDM): measures whether the frozen encoder's latent carries
    # action/fast info. Sobal-style collapse => low; IDM should force this UP (the recovery).
    fast_r2_action = float("nan")
    if actions is not None:
        a = np.asarray(actions)[:, :-1]                           # [N,T-1,K]
        fast_r2_action = _ridge_r2(flat(tr, Zcat), flat(tr, a), flat(te, Zcat), flat(te, a), alpha)

    # slow: z_t -> x_0 (trajectory initial state, broadcast over t)
    x0 = np.repeat(states[:, :1, :], T, axis=1)                  # [N,T,S]
    slow_r2_init = _ridge_r2(flat(tr, Z), flat(tr, x0), flat(te, Z), flat(te, x0), alpha)

    feat_std = float(Z.reshape(-1, D).std(axis=0).mean())
    return {
        "fast_r2_action": fast_r2_action,  # headline: IDM should raise this most
        "fast_r2_state": fast_r2_state,
        "fast_r2_delta": fast_r2_delta,
        "slow_r2_init": slow_r2_init,
        "fast_minus_slow": fast_r2_delta - slow_r2_init,  # >0 = dynamics-dominant (healthy)
        "feat_std": feat_std,
    }


def probe_encoder(encoder, dataset, device, **kw):
    """Convenience: encode `dataset` with `encoder` and run the collapse probes."""
    Z, states, actions = encode_trajectories(encoder, dataset, device, max_traj=kw.pop("max_traj", None))
    return collapse_probes(Z, states, actions, **kw)


if __name__ == "__main__":
    # Smoke: a RANDOM-INIT encoder should already linearly carry some state info (sanity that the
    # probes run and return finite numbers); the science is the idm-on vs idm-off CONTRAST (run_ablation).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eb_jepa.architectures import SetTransformerEncoder
    from eb_jepa.datasets.microbiome.traj import GLVTrajConfig, GLVTrajDataset

    dev = torch.device("cpu")
    ds = GLVTrajDataset(GLVTrajConfig(n_traj=24, T=16, n_species=16, n_candidate=4, sim_seed=7))
    enc = SetTransformerEncoder(token_dim=ds.token_dim, d_model=64, n_heads=4, n_layers=2).to(dev)
    out = probe_encoder(enc, ds, dev, max_traj=24)
    print("collapse probes (random-init encoder):")
    for k, v in out.items():
        print(f"  {k:16s}= {v:.4f}")
    print("OK: probes ran and returned finite numbers." if all(np.isfinite(list(out.values()))) else "NON-FINITE!")
