"""
Baselines for Microbiome-JEPA -- the controlled comparison every representation
paper needs. We score three representations of a microbiome community on the
SAME subject-disjoint linear probes (host-age regression + T1D classification),
so the only thing that changes is the representation:

  1. raw             abundance-weighted mean ProkBERT descriptor (NO learning)
                     -> "does the JEPA beat trivial sequence+abundance features?"
  2. random-encoder  an UNTRAINED SetEncoder, same architecture, random weights
                     (averaged over a few seeds) -> "is it the training that
                     helps, or just the architecture / set-pooling inductive bias?"
  3. jepa            the trained encoder from a checkpoint (optional)

All three go through one identical pipeline: subject-pool the per-timestep
representation over the window, standardize features on TRAIN, fit Ridge (age)
and balanced LogisticRegression (T1D) on TRAIN, score on VAL. Subjects in
train/val are disjoint (handled by the dataset split), so there is no leakage.

  # raw + random baselines only (no trained model needed):
  python -m examples.microbiome.baselines

  # add the trained JEPA column from a checkpoint:
  python -m examples.microbiome.baselines --ckpt checkpoints/microbiome/microbiome_jepa.pt
"""

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.architectures import SetEncoder
from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders


def _gather_raw(loader):
    """Subject-pooled RAW descriptor: the abundance-weighted mean ProkBERT
    embedding of the community, averaged over the window. No model involved."""
    X, age, lab = [], [], []
    for b in loader:
        X.append(b["phylo"].mean(dim=1).numpy())  # [B, T, E] -> [B, E]
        age.append(b["age"].numpy())
        lab.append(b["label"].numpy())
    return np.concatenate(X), np.concatenate(age), np.concatenate(lab).astype(int)


@torch.no_grad()
def _gather_encoder(encoder, loader, device):
    """Subject-pooled latent from an encoder (trained or random), pooled over the
    window exactly like main.py's probe (state.mean over T)."""
    X, age, lab = [], [], []
    for b in loader:
        state = encoder(b["observations"].to(device))  # [B, D, T, 1, 1]
        X.append(state.mean(dim=2)[..., 0, 0].cpu().numpy())  # [B, D]
        age.append(b["age"].numpy())
        lab.append(b["label"].numpy())
    return np.concatenate(X), np.concatenate(age), np.concatenate(lab).astype(int)


def _probe(Xt, at, yt, Xv, av, yv):
    """One standardized linear probe. Fit on train, score on val."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xt)
    Xt, Xv = sc.transform(Xt), sc.transform(Xv)
    out = {}
    out["age_r2"] = float(r2_score(av, Ridge(alpha=1.0).fit(Xt, at).predict(Xv)))
    if len(np.unique(yt)) == 2 and len(np.unique(yv)) == 2:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt, yt)
        out["t1d_auroc"] = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
    else:
        out["t1d_auroc"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="trained JEPA checkpoint to add as the 'jepa' row")
    ap.add_argument("--cfg", default="examples/microbiome/cfgs/train.yaml",
                    help="config used when no --ckpt is given")
    ap.add_argument("--rand-seeds", type=int, nargs="+", default=[1, 1000, 10000],
                    help="seeds for the untrained random-encoder baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt is not None:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(blob["cfg"])
    else:
        blob = None
        cfg = OmegaConf.load(args.cfg)

    dcfg = MicrobiomeConfig(
        cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
        n_max=cfg.data.n_max, emb_dim=cfg.model.emb_dim,
        val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed,
    )
    _, _, train_loader, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)

    rows = {}  # name -> {age_r2, t1d_auroc} or (mean, std) dict

    # ---- 1. raw (non-learned) ---------------------------------------------
    Xt, at, yt = _gather_raw(train_loader)
    Xv, av, yv = _gather_raw(val_loader)
    rows["raw (mean ProkBERT)"] = _probe(Xt, at, yt, Xv, av, yv)

    # ---- 2. random untrained encoder (averaged over seeds) ----------------
    rand = {"age_r2": [], "t1d_auroc": []}
    for s in args.rand_seeds:
        torch.manual_seed(s)
        enc = SetEncoder(emb_dim=cfg.model.emb_dim, h_d=cfg.model.henc,
                         out_d=cfg.model.dstc).to(device).eval()
        Xt, at, yt = _gather_encoder(enc, train_loader, device)
        Xv, av, yv = _gather_encoder(enc, val_loader, device)
        r = _probe(Xt, at, yt, Xv, av, yv)
        rand["age_r2"].append(r["age_r2"])
        rand["t1d_auroc"].append(r["t1d_auroc"])
    rows[f"random encoder (n={len(args.rand_seeds)})"] = {
        "age_r2": (float(np.mean(rand["age_r2"])), float(np.std(rand["age_r2"]))),
        "t1d_auroc": (float(np.nanmean(rand["t1d_auroc"])), float(np.nanstd(rand["t1d_auroc"]))),
    }

    # ---- 3. trained JEPA encoder (optional) -------------------------------
    if blob is not None:
        from examples.microbiome.main import build_jepa
        jepa = build_jepa(cfg, train_loader.dataset.action_dim, device)
        jepa.load_state_dict(blob["jepa"])
        jepa.eval()
        Xt, at, yt = _gather_encoder(jepa.encoder, train_loader, device)
        Xv, av, yv = _gather_encoder(jepa.encoder, val_loader, device)
        rows["JEPA (trained)"] = _probe(Xt, at, yt, Xv, av, yv)

    # ---- print the comparison table ---------------------------------------
    def _fmt(v):
        if isinstance(v, tuple):
            return f"{v[0]:.3f}+/-{v[1]:.3f}"
        return f"{v:.3f}"

    print("\n== Microbiome representation baselines "
          "(subject-disjoint standardized linear probe) ==")
    print(f"{'representation':<26}{'age_r2':>14}{'t1d_auroc':>14}")
    print("-" * 54)
    for name, m in rows.items():
        print(f"{name:<26}{_fmt(m['age_r2']):>14}{_fmt(m['t1d_auroc']):>14}")
    if blob is None:
        print("\n(no --ckpt given: 'JEPA (trained)' row omitted. Pass "
              "--ckpt <path> to add it.)")
    print("\nRead: age_r2 = host-age R^2 (higher=better, 0=no signal); "
          "t1d_auroc = T1D AUROC (0.5=chance).")


if __name__ == "__main__":
    main()
