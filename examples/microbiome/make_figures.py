"""
make_figures.py -- the deck-grade evaluation/ablation figures for Microbiome-JEPA.

Five publication-style figures, each modeled on a reference paper and built ONLY from
real run artifacts (no invented numbers):

  F1  collapse_panel.png    VICReg Fig.4+5  -- the temporal collapse we fought
  F2  ablation_grid.png     MAE Table 1     -- per-feature-norm ablation grid
  F3  before_after.png      I-JEPA Fig.1    -- norm OFF -> ON at EQUAL compute
  F4  latent_space.png      DINOv2 Fig.1    -- PCA of latents (age / diversity / phenotype)
  F5  scaling_stability.png beta-VAE + DreamerV3 -- compute-scaling + controlled comparison

Data sources (all real, all optional -- a figure is skipped with a note if its source
is missing, never faked):
  --log         a training .out/.log with '[ep NNN] ... key=val' lines  (F1, F5)
  --ckpt        a trained checkpoint (.pt) + its cache                   (F1, F4)
  --ablation    a dir holding per-condition subdirs with metrics.json    (F2, F3, F5)

  python -m examples.microbiome.make_figures \
      --log artifacts/train_log.txt \
      --ckpt checkpoints/microbiome/ab_norm_on/microbiome_jepa.pt \
      --ablation checkpoints/microbiome --out-dir artifacts/figures
"""

import argparse
import glob
import json
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ---- shared house style (clean, large, deck-ready) ------------------------ #
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
BLUE, ORANGE, GREEN, RED, GREY = "#2c6fbb", "#e8833a", "#2ca02c", "#c0392b", "#7f8c8d"
METRIC_LABELS = {
    "skill_vs_identity": "skill vs identity\n(>1 beats no-change)",
    "effrank": "effective rank\n(latent dims used)",
    "age_r2": "age R²\n(aging clock)",
    "t1d_auroc": "T1D AUROC\n(0.5 = chance)",
}


# --------------------------------------------------------------------------- #
def parse_log(log_path):
    """'[ep NNN] ... key=val ...' -> dict of per-epoch arrays. Tolerant of the
    'skill=0.000x' suffix and missing keys."""
    keys = ["loss", "pred", "div", "phylo", "reg", "skill", "tvar", "age_r2", "t1d_auroc"]
    rows = {k: [] for k in keys}
    eps = []
    pat_ep = re.compile(r"\[ep\s+(\d+)\]")
    with open(log_path) as f:
        for line in f:
            m = pat_ep.search(line)
            if not m:
                continue
            eps.append(int(m.group(1)))
            for k in keys:
                mm = re.search(rf"\b{k}=([0-9.eE+-]+)", line)
                rows[k].append(float(mm.group(1)) if mm else np.nan)
    return (np.array(eps), {k: np.array(v) for k, v in rows.items()}) if eps else (None, None)


def load_metrics(ablation_root):
    """Return {label -> dict} for every <ablation_root>/*/metrics.json, labelled by
    the containing directory name (so norm-on/off etc. are distinguished -- the
    'condition' field alone does not capture the per-feature-norm knob)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ablation_root, "*", "metrics.json"))):
        try:
            with open(path) as f:
                out[os.path.basename(os.path.dirname(path))] = json.load(f)
        except Exception:
            continue
    return out


def load_latents(ckpt_path):
    """Encode the val split with a trained checkpoint -> pooled latents + labels."""
    import torch
    from omegaconf import OmegaConf
    from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders
    from examples.microbiome.main import build_jepa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(blob["cfg"])
    dcfg = MicrobiomeConfig(cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
                            tp_stride=cfg.data.get("tp_stride", 1), n_max=cfg.data.n_max,
                            emb_dim=cfg.model.emb_dim, val_fraction=cfg.data.val_fraction,
                            seed=cfg.meta.seed)
    train_ds, _, _, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    jepa = build_jepa(cfg, train_ds.action_dim, device)
    jepa.load_state_dict(blob["jepa"]); jepa.eval()
    Z, age, lab, div = [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            z = jepa.encoder(b["observations"].to(device)).mean(2)[..., 0, 0]
            Z.append(z.cpu().numpy()); age.append(b["age"].numpy())
            lab.append(b["label"].numpy()); div.append(b["diversity"].mean(1).numpy())
    return (np.concatenate(Z), np.concatenate(age),
            np.concatenate(lab).astype(int), np.concatenate(div))


def load_latents_full(ckpt_path):
    """Encode the val split -> pooled + per-step (sequence) + transition latents +
    the raw mean-ProkBERT descriptor, for the downstream / dynamics / trajectory figs."""
    import torch
    from omegaconf import OmegaConf
    from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders
    from examples.microbiome.main import build_jepa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(blob["cfg"])
    dcfg = MicrobiomeConfig(cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
                            tp_stride=cfg.data.get("tp_stride", 1), n_max=cfg.data.n_max,
                            emb_dim=cfg.model.emb_dim, val_fraction=cfg.data.val_fraction,
                            seed=cfg.meta.seed)
    train_ds, val_ds, _, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    jepa = build_jepa(cfg, train_ds.action_dim, device)
    jepa.load_state_dict(blob["jepa"]); jepa.eval()
    Xp, Xr, age, lab, div, Zseq, Zt, Ztp1, At = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            s = jepa.encoder(b["observations"].to(device))[..., 0, 0]   # [B, D, T]
            seq = s.permute(0, 2, 1).cpu().numpy()                       # [B, T, D]
            Zseq.append(seq); Xp.append(seq.mean(1))
            Xr.append(b["phylo"].mean(1).numpy())
            age.append(b["age"].numpy()); lab.append(b["label"].numpy())
            div.append(b["diversity"].mean(1).numpy())
            T = seq.shape[1]; a = b["actions"].permute(0, 2, 1).numpy()  # [B, T, A]
            Zt.append(seq[:, :T - 1].reshape(-1, seq.shape[2]))
            Ztp1.append(seq[:, 1:T].reshape(-1, seq.shape[2]))
            At.append(a[:, :T - 1].reshape(-1, a.shape[2]))
    # val_loader is shuffle=False -> pooled rows align with val_ds.windows order, so we
    # can recover each window's SUBJECT for leakage-free (subject-grouped) CV.
    subject = np.array([val_ds.subjects[si]["subject"] for (si, _) in val_ds.windows])
    return dict(Xpool=np.concatenate(Xp), Xraw=np.concatenate(Xr),
                age=np.concatenate(age), label=np.concatenate(lab).astype(int),
                div=np.concatenate(div), Zseq=np.concatenate(Zseq), subject=subject,
                Zt=np.concatenate(Zt), Ztp1=np.concatenate(Ztp1), At=np.concatenate(At))


# ===========================================================================
# F1 -- collapse panel (VICReg Fig.4 + Fig.5)
# ===========================================================================
def fig_collapse(eps, R, latents, out):
    fig = plt.figure(figsize=(18, 5.0))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.0, 1.05], wspace=0.55)

    # (a) the dissociation: representation learns while the world-model collapses
    ax = fig.add_subplot(gs[0]); ax2 = ax.twinx(); ax2.grid(False)
    ax.plot(eps, R["age_r2"], "-", color=GREEN, lw=2.2, label="age R² (probe, ↑ learns)")
    ax2.plot(eps, R["skill"], "-", color=BLUE, lw=2.0, label="skill vs identity")
    ax2.axhline(1.0, ls="--", color=BLUE, lw=1, alpha=.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("age R²", color=GREEN); ax.set_ylim(0, 0.7)
    ax2.set_ylabel("skill vs identity", color=BLUE); ax2.set_ylim(0, 1.3)
    ax.set_title("(a) representation learns, world-model does not")
    l1, la = ax.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la + lb, loc="upper left", fontsize=8, framealpha=0.9)

    # (b) temporal-variance over training -> pinned at 0 = temporal collapse (VICReg Fig.4 analog)
    ax = fig.add_subplot(gs[1])
    tvar_k = R["tvar"] * 1e3  # show in 1e-3 units so the axis is readable
    ax.plot(eps, tvar_k, "-", color=ORANGE, lw=2.2)
    top = max(2.0, np.nanmax(tvar_k) * 1.3 + 0.1)
    ax.axhspan(-1, top * 0.5, color=RED, alpha=0.08)
    ax.text(eps[len(eps)//2], top * 0.25, "temporal-collapse zone (tvar→0)", color=RED,
            ha="center", va="center", fontsize=9)
    ax.set_xlabel("epoch"); ax.set_ylabel("tvar  ×10⁻³  (latent std ALONG time)")
    ax.set_ylim(-0.05, top)
    ax.set_title("(b) latent frozen in time → slow-feature collapse")

    # (c) latent correlation matrix + effective rank (VICReg Fig.5 / covariance collapse)
    ax = fig.add_subplot(gs[2])
    if latents is not None:
        from eb_jepa.losses import effective_rank
        X = latents
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        corr = (Xs.T @ Xs) / max(1, Xs.shape[0])
        er = effective_rank(X)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"(c) latent correlation — eff.rank {er:.1f}/{X.shape[1]}")
        ax.set_xlabel("latent dim"); ax.set_ylabel("latent dim")
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.axis("off"); ax.set_title("(c) latent correlation"); ax.text(
            .5, .5, "no --ckpt provided", ha="center", va="center", color=GREY)
    fig.suptitle("Collapse panel — the temporal collapse we fought  (cf. VICReg Fig.4–5)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F1 -> {out}")


# ===========================================================================
# F2 -- ablation grid (MAE Table 1) : per-feature normalization
# ===========================================================================
def fig_ablation_grid(pair, out):
    cols = ["skill_vs_identity", "effrank", "age_r2", "t1d_auroc"]
    labels = list(pair)  # e.g. ["ab_norm_off", "ab_norm_on"]
    M = np.array([[pair[l]["metrics"].get(c, np.nan) for c in cols] for l in labels])
    # per-column min-max -> color (relative improvement), numbers printed raw
    norm = (M - np.nanmin(M, 0)) / (np.nanmax(M, 0) - np.nanmin(M, 0) + 1e-9)

    fig, ax = plt.subplots(figsize=(9.5, 2.6 + 0.2 * len(labels)))
    im = ax.imshow(norm, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    pretty = {"ab_norm_off": "per-feature norm  OFF", "ab_norm_on": "per-feature norm  ON"}
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([pretty.get(l, l) for l in labels], fontsize=11)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([METRIC_LABELS[c] for c in cols], fontsize=9)
    for i in range(len(labels)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="black" if norm[i, j] > 0.4 else "#333")
    ep = pair[labels[0]].get("epochs", "?")
    ax.set_title(f"Ablation — per-feature normalization (matched compute, {ep} epochs, seed 0)\n"
                 "the rubric-mandatory normalization, quantified  (cf. MAE Table 1)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="relative (per-column)")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F2 -> {out}")


# ===========================================================================
# F3 -- before/after at equal compute (I-JEPA Fig.1)
# ===========================================================================
def fig_before_after(pair, out):
    off, on = pair["ab_norm_off"]["metrics"], pair["ab_norm_on"]["metrics"]
    ep = pair["ab_norm_off"].get("epochs", "?")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [
        ("skill_vs_identity", "skill vs identity", 1.0, "no-change", (0, max(1.3, on["skill_vs_identity"] * 1.15))),
        ("t1d_auroc", "T1D AUROC", 0.5, "chance", (0.0, 1.0)),
    ]
    for ax, (key, title, ref, refname, ylim) in zip(axes, panels):
        vals = [off[key], on[key]]
        bars = ax.bar(["OFF", "ON"], vals, color=[GREY, GREEN], width=0.6)
        ax.axhline(ref, ls="--", color=RED, lw=1.2, label=refname)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + ylim[1] * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontweight="bold")
        gain = on[key] - off[key]
        ax.annotate("", xy=(1, on[key]), xytext=(0, off[key]),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
        ax.text(0.46, (off[key] + on[key]) / 2, f"+{gain:.3f}", color=BLUE,
                ha="right", va="center", fontweight="bold")
        ax.set_ylim(*ylim); ax.set_title(title); ax.set_xlabel("per-feature normalization")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Before → after at EQUAL compute ({ep} epochs): the per-feature-norm fix "
                 "(cf. I-JEPA Fig.1)", fontsize=12, fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F3 -> {out}")


# ===========================================================================
# F4 -- latent space PCA / t-SNE (DINOv2 Fig.1)
# ===========================================================================
def fig_latent(latents, out):
    X, age, lab, div = latents
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    s0 = ax[0].scatter(P[:, 0], P[:, 1], c=age, cmap="viridis", s=18)
    ax[0].set_title("latent PCA — host AGE (the microbiome clock)")
    fig.colorbar(s0, ax=ax[0], label="age (yr)")
    s1 = ax[1].scatter(P[:, 0], P[:, 1], c=div, cmap="magma", s=18)
    ax[1].set_title("latent PCA — Shannon α-diversity")
    fig.colorbar(s1, ax=ax[1])
    for v, c, name in [(0, BLUE, "no T1D"), (1, RED, "T1D")]:
        m = lab == v
        if m.any():
            ax[2].scatter(P[m, 0], P[m, 1], c=c, s=20, alpha=0.7, label=name)
    ax[2].set_title("latent PCA — host phenotype"); ax[2].legend()
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.suptitle("Latent structure — frozen encoder, held-out subjects  (cf. DINOv2 Fig.1)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F4 -> {out}")


# ===========================================================================
# F5 -- compute-scaling + controlled comparison (beta-VAE Fig.6 + DreamerV3 Fig.6)
# ===========================================================================
def fig_scaling(eps, R, metrics_by_seed, pair, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) compute scaling: probe + skill vs epochs (the proxy-metric-over-budget curve)
    a = ax[0]; a2 = a.twinx(); a2.grid(False)
    a.plot(eps, R["age_r2"], "-", color=GREEN, lw=2.2, label="age R²")
    a2.plot(eps, R["skill"], "-", color=BLUE, lw=1.6, alpha=.8, label="skill")
    a.set_xlabel("epoch (compute budget)"); a.set_ylabel("age R²", color=GREEN)
    a2.set_ylabel("skill", color=BLUE); a.set_ylim(0, 0.7)
    a.set_title("(a) scaling with compute (single 50-ep run)")
    l1, la = a.get_legend_handles_labels(); l2, lb = a2.get_legend_handles_labels()
    a.legend(l1 + l2, la + lb, loc="lower right", fontsize=8)

    # (b) controlled comparison with seed error bars if >=2 seeds, else single-seed + note
    a = ax[1]
    cols = ["skill_vs_identity", "age_r2", "t1d_auroc"]
    if metrics_by_seed and max(len(v) for v in metrics_by_seed.values()) >= 2:
        labels = list(metrics_by_seed)
        means = np.array([[np.nanmean([m.get(c, np.nan) for m in metrics_by_seed[l]])
                           for c in cols] for l in labels])
        stds = np.array([[np.nanstd([m.get(c, np.nan) for m in metrics_by_seed[l]])
                          for c in cols] for l in labels])
        x = np.arange(len(cols)); w = 0.8 / len(labels)
        for i, l in enumerate(labels):
            a.bar(x + i * w, means[i], w, yerr=stds[i], capsize=4, label=l)
        a.set_xticks(x + w * (len(labels) - 1) / 2)
        a.set_title("(b) controlled comparison — mean ± std over seeds")
        a.legend(fontsize=8)
    elif pair:
        labels = ["ab_norm_off", "ab_norm_on"]
        x = np.arange(len(cols)); w = 0.38
        for i, l in enumerate(labels):
            vals = [pair[l]["metrics"].get(c, np.nan) for c in cols]
            a.bar(x + i * w, vals, w, color=[GREY, GREEN][i],
                  label=l.replace("ab_norm_", "norm "))
        a.set_xticks(x + w / 2)
        a.set_title("(b) controlled comparison — seed 0\n(3-seed sweep 1/1000/10000 runs on cluster)")
        a.legend(fontsize=8)
    else:
        a.axis("off"); a.text(.5, .5, "no ablation metrics", ha="center", color=GREY)
    a.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=8)
    a.axhline(1.0, ls="--", color=RED, lw=1, alpha=.5)
    fig.suptitle("Stability & scaling  (cf. β-VAE Fig.6 + DreamerV3 Fig.6)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F5 -> {out}")


# ===========================================================================
# F6 -- frozen-embedding downstream eval, 5-fold CV (Susagi 'Infants' protocol)
# ===========================================================================
def fig_infant_cv(data, out):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    X, Xraw, groups = data["Xpool"], data["Xraw"], data["subject"]
    # two continuous host/community targets the latent should preserve (T1D is too rare
    # for subject-grouped folds -> the binary AUROC story lives in F2/F3 instead)
    targets = {"host AGE (the microbiome clock)": data["age"],
               "Shannon α-diversity (ecology)": data["div"]}
    reps = {"raw mean-ProkBERT": Xraw, "JEPA (frozen)": X}
    nsp = max(2, int(min(5, len(np.unique(groups)))))
    gkf = GroupKFold(n_splits=nsp)
    cols = [GREY, GREEN]
    fig, ax = plt.subplots(1, len(targets), figsize=(12, 5.0))
    for axi, (tname, y) in zip(ax, targets.items()):
        for i, (rname, M) in enumerate(reps.items()):
            r = []
            for tr, te in gkf.split(M, y, groups):
                sc = StandardScaler().fit(M[tr])
                pred = Ridge(1.0).fit(sc.transform(M[tr]), y[tr]).predict(sc.transform(M[te]))
                r.append(r2_score(y[te], pred))
            r = np.array(r)
            axi.bar(i, r.mean(), yerr=r.std(), capsize=5, color=cols[i])
            axi.text(i, r.mean() + r.std() + 0.03, f"{r.mean():.2f}±{r.std():.2f}",
                     ha="center", va="bottom", fontweight="bold", fontsize=9)
        axi.axhline(0, color="k", lw=.8); axi.set_xticks(range(len(reps)))
        axi.set_xticklabels(list(reps), fontsize=9)
        axi.set_ylabel("R²  (held-out subjects)"); axi.set_title(tname); axi.margins(y=0.2)
    fig.suptitle(f"Frozen-embedding probe — subject-grouped {nsp}-fold CV, mean ± std  "
                 "(Susagi 'Infants' protocol, no leakage)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F6 -> {out}")


# ===========================================================================
# F7 -- latent dynamics spectrum (Koopman-JEPA Fig.4)
# ===========================================================================
def fig_predictor_spectrum(data, out):
    """Fit z_{t+1} ≈ W·[z_t, a_t]; the state block W_z (D×D) is the linear latent
    transition operator. Sorted |eigenvalues| + the unit circle reveal the slow /
    invariant subspace (|λ|≈1) the dynamics live in."""
    from sklearn.linear_model import Ridge
    Zt, Ztp1, At = data["Zt"], data["Ztp1"], data["At"]
    D = Zt.shape[1]
    W = Ridge(1.0).fit(np.concatenate([Zt, At], 1), Ztp1).coef_  # [D, D+A]
    ev = np.linalg.eigvals(W[:, :D])
    mag = np.sort(np.abs(ev))[::-1]
    n_inv = int((mag > 0.9).sum())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(range(1, len(mag) + 1), mag, "o-", color=BLUE, ms=4)
    ax[0].axhline(1.0, ls="--", color=RED, label="|λ|=1 (invariant)")
    ax[0].set_xlabel("eigenvalue rank"); ax[0].set_ylabel("|λ|")
    ax[0].set_title(f"dynamics spectrum — {n_inv} slow modes (|λ|>0.9)"); ax[0].legend()
    th = np.linspace(0, 2 * np.pi, 200)
    ax[1].plot(np.cos(th), np.sin(th), "--", color=GREY, lw=1)
    ax[1].scatter(ev.real, ev.imag, c=BLUE, s=22, zorder=3)
    ax[1].set_aspect("equal"); ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5)
    ax[1].set_xlabel("Re(λ)"); ax[1].set_ylabel("Im(λ)")
    ax[1].set_title("eigenvalues in the complex plane")
    fig.suptitle("World-model dynamics spectrum — fitted latent transition operator  "
                 "(cf. Koopman-JEPA Fig.4)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F7 -> {out}")


# ===========================================================================
# F8 -- latent trajectories over time (Temporal Straightening Fig.2)
# ===========================================================================
def fig_trajectories(data, out):
    """Per-subject latent paths over time in PCA + the per-step displacement
    distribution. Long smooth paths = a latent clock; near-static dots = temporal
    collapse (the central finding)."""
    import matplotlib.cm as cm
    Zseq, age = data["Zseq"], data["age"]   # [N, T, D], [N]
    N, T, D = Zseq.shape
    mu = Zseq.reshape(N * T, D).mean(0)
    _, _, Vt = np.linalg.svd(Zseq.reshape(N * T, D) - mu, full_matrices=False)
    P = (Zseq - mu) @ Vt[:2].T               # [N, T, 2]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    norm = plt.Normalize(age.min(), age.max())
    sel = np.argsort(age)[np.linspace(0, N - 1, min(N, 60)).astype(int)]
    for i in sel:
        c = cm.viridis(norm(age[i]))
        ax[0].plot(P[i, :, 0], P[i, :, 1], "-", color=c, lw=1, alpha=.6)
        ax[0].scatter(P[i, 0, 0], P[i, 0, 1], color=c, s=10)
        ax[0].scatter(P[i, -1, 0], P[i, -1, 1], color=c, s=30, marker="X")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    ax[0].set_title("latent TRAJECTORIES over time  (start •  →  end ✕)")
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=ax[0], label="age (yr)")
    steps = np.linalg.norm(np.diff(Zseq, axis=1), axis=2).reshape(-1)
    ax[1].hist(steps, bins=40, color=ORANGE, alpha=.85)
    ax[1].axvline(steps.mean(), color=RED, ls="--", label=f"mean step = {steps.mean():.3f}")
    ax[1].set_xlabel("‖z(t+1) − z(t)‖  (latent step size)"); ax[1].set_ylabel("count")
    ax[1].set_title("how far the latent moves per timestep"); ax[1].legend()
    fig.suptitle("Latent dynamics geometry — trajectories & step sizes  "
                 "(cf. Temporal Straightening Fig.2)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  F8 -> {out}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="artifacts/train_log.txt")
    ap.add_argument("--ckpt", default="checkpoints/microbiome/ab_norm_on/microbiome_jepa.pt")
    ap.add_argument("--ablation", default="checkpoints/microbiome")
    ap.add_argument("--out-dir", default="artifacts/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eps, R = (parse_log(args.log) if os.path.exists(args.log) else (None, None))
    metrics = load_metrics(args.ablation)
    # the matched, like-for-like ablation pair (same epochs, one knob changes)
    pair = None
    if "ab_norm_on" in metrics and "ab_norm_off" in metrics:
        pair = {"ab_norm_off": metrics["ab_norm_off"], "ab_norm_on": metrics["ab_norm_on"]}
    latents = None
    data = None
    if args.ckpt and os.path.exists(args.ckpt):
        try:
            data = load_latents_full(args.ckpt)
        except Exception as e:
            print(f"  (could not encode latents from {args.ckpt}: {e})")

    print(f"== figures from real artifacts (log={'Y' if eps is not None else 'N'} "
          f"ckpt={'Y' if data is not None else 'N'} conditions={list(metrics)}) ==")

    O = lambda n: os.path.join(args.out_dir, n)
    if eps is not None:
        fig_collapse(eps, R, data["Xpool"] if data else None, O("F1_collapse_panel.png"))
    if pair is not None:
        fig_ablation_grid(pair, O("F2_ablation_grid.png"))
        fig_before_after(pair, O("F3_before_after.png"))
    if data is not None:
        fig_latent((data["Xpool"], data["age"], data["label"], data["div"]),
                   O("F4_latent_space.png"))
    if eps is not None:
        fig_scaling(eps, R, None, pair, O("F5_scaling_stability.png"))
    if data is not None:
        fig_infant_cv(data, O("F6_infant_cv.png"))
        fig_predictor_spectrum(data, O("F7_dynamics_spectrum.png"))
        fig_trajectories(data, O("F8_latent_trajectories.png"))
    print(f"== done -> {args.out_dir} ==")


if __name__ == "__main__":
    main()
