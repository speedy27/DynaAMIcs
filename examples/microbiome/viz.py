"""
viz.py - Plots to understand the Microbiome-JEPA.

Two figures:
  1) training curves      (from a training .out/.log): loss terms + age_r2 + tvar + skill
  2) latent-space panels  (from a checkpoint + cache):  PCA colored by age / diversity /
                          feeding, and a predicted-vs-true age scatter (the microbiome clock)

Usage:
  python -m examples.microbiome.viz --log artifacts/train_log.txt --curves-out artifacts/curves.png
  python -m examples.microbiome.viz --ckpt <ckpt.pt> --latent-out artifacts/latent.png
"""

import argparse
import os
import re

import numpy as np


# --------------------------------------------------------------------------- #
def plot_curves(log_path, out_path):
    """Parse '[ep NNN] ... key=val ...' lines and plot the training dynamics."""
    pat_ep = re.compile(r"\[ep\s+(\d+)\]")
    keys = ["pred", "div", "phylo", "reg", "skill", "tvar", "age_r2", "t1d_auroc"]
    rows = {k: [] for k in keys}
    eps = []
    with open(log_path) as f:
        for line in f:
            m = pat_ep.search(line)
            if not m:
                continue
            eps.append(int(m.group(1)))
            for k in keys:
                mm = re.search(rf"{k}=([0-9.]+)", line)
                rows[k].append(float(mm.group(1)) if mm else np.nan)
    if not eps:
        print("no epoch lines found in", log_path)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # (a) training loss components
    for k in ["pred", "div", "phylo", "reg"]:
        ax[0].plot(eps, rows[k], label=k)
    ax[0].set_title("training loss components"); ax[0].set_xlabel("epoch")
    ax[0].set_yscale("log"); ax[0].legend()
    # (b) downstream probe = the positive result
    ax[1].plot(eps, rows["age_r2"], "-o", c="tab:green", label="age R² (microbiome clock)")
    ax[1].plot(eps, rows["t1d_auroc"], "-o", c="tab:red", alpha=.6, label="T1D AUROC")
    ax[1].axhline(0.5, ls="--", c="gray", lw=1, label="chance (AUROC)")
    ax[1].set_title("downstream probes (held-out subjects)"); ax[1].set_xlabel("epoch")
    ax[1].set_ylim(-0.05, 1.0); ax[1].legend()
    # (c) the temporal-collapse finding
    ax2 = ax[2].twinx()
    ax[2].plot(eps, rows["skill"], "-o", c="tab:blue", label="skill vs identity")
    ax[2].axhline(1.0, ls="--", c="tab:blue", lw=1, alpha=.5)
    ax2.plot(eps, rows["tvar"], "-o", c="tab:orange", label="tvar (temporal var)")
    ax[2].set_title("world-model dynamics (collapse monitor)"); ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("skill (>1 = beats no-change)", color="tab:blue")
    ax2.set_ylabel("tvar (0 = temporal collapse)", color="tab:orange")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print("saved", out_path)


# --------------------------------------------------------------------------- #
def plot_latent(ckpt_path, out_path):
    import torch
    from omegaconf import OmegaConf
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders
    from examples.microbiome.main import build_jepa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(blob["cfg"])
    dcfg = MicrobiomeConfig(cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
                            n_max=cfg.data.n_max, emb_dim=cfg.model.emb_dim,
                            val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed)
    train_ds, val_ds, train_loader, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    jepa = build_jepa(cfg, train_ds.action_dim, device); jepa.load_state_dict(blob["jepa"]); jepa.eval()

    def gather(loader):
        Z, A, D, M = [], [], [], []
        with torch.no_grad():
            for b in loader:
                z = jepa.encoder(b["observations"].to(device)).mean(2)[..., 0, 0]
                Z.append(z.cpu().numpy()); A.append(b["age"].numpy())
                D.append(b["diversity"].mean(1).numpy())
                M.append(b["actions"][:, :-1].amax(2).argmax(1).numpy())  # feeding category idx
        return np.concatenate(Z), np.concatenate(A), np.concatenate(D), np.concatenate(M)

    Ztr, Atr, _, _ = gather(train_loader)
    Zv, Av, Dv, Mv = gather(val_loader)
    reg = Ridge(1.0).fit(Ztr, Atr); pred = reg.predict(Zv); r2 = r2_score(Av, pred)

    mu = Ztr.mean(0); _, _, Vt = np.linalg.svd(Ztr - mu, full_matrices=False)
    P = (Zv - mu) @ Vt[:2].T

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    s0 = ax[0].scatter(P[:, 0], P[:, 1], c=Av, cmap="viridis", s=16)
    ax[0].set_title("latent PCA — colored by AGE (microbiome clock)"); plt.colorbar(s0, ax=ax[0], label="age (yr)")
    s1 = ax[1].scatter(P[:, 0], P[:, 1], c=Dv, cmap="magma", s=16)
    ax[1].set_title("latent PCA — colored by Shannon α-diversity"); plt.colorbar(s1, ax=ax[1])
    ax[2].scatter(Av, pred, s=16, alpha=.6); lim = [min(Av.min(), pred.min()), max(Av.max(), pred.max())]
    ax[2].plot(lim, lim, "--", c="gray"); ax[2].set_xlabel("true age (yr)"); ax[2].set_ylabel("predicted age")
    ax[2].set_title(f"microbiome aging clock — held-out R²={r2:.2f}")
    for a in ax[:2]:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print("saved", out_path, "| held-out age R²=", round(r2, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--curves-out", default="artifacts/curves.png")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--latent-out", default="artifacts/latent.png")
    args = ap.parse_args()
    if args.log:
        plot_curves(args.log, args.curves_out)
    if args.ckpt:
        plot_latent(args.ckpt, args.latent_out)
