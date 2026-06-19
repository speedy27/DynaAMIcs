"""
Tahoe-JEPA: a cell-state Joint-Embedding Predictive model on Tahoe-100M.

A scaled, differentiated take on GeneJEPA (Litman et al. 2025). Same JEPA premise
(predict in representation space, no count reconstruction) but THREE deliberate
differences we ablate:
  1. SIGReg (BCS) regularizer        instead of GeneJEPA's VICReg
  2. PathwayCoherenceLoss            a gene-program structural prior they don't have
  3. biological two-view augmentation (gene dropout + noise) instead of masked-prediction

Encoder = MLP over a top-K gene panel; two augmented views are pulled together by
the regularizer; a linear probe on frozen embeddings is compared to raw-expression
and PCA baselines (the GeneJEPA evaluation protocol).

Training curves (loss + probe F1 per epoch) are saved to:
  <ckpt>/tahoe_jepa_curves.json   (incremental, one entry per epoch)
  <ckpt>/tahoe_jepa_curves.png    (figure generated at end of training)

  python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml optim.epochs=250
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import Projector
from eb_jepa.datasets.tahoe.dataset import TahoeConfig, make_loaders
from eb_jepa.losses import BCS, VICRegLoss, PathwayCoherenceLoss
from eb_jepa.schedulers import CosineWithWarmup


class CellEncoder(nn.Module):
    """MLP cell-state encoder over a fixed gene panel -> latent z."""
    def __init__(self, k, h, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, d),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def embed(enc, X, device, bs=4096):
    enc.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(enc(X[i:i+bs].to(device)).cpu())
    return torch.cat(out).numpy()


def probe(name, Xtr, ytr, Xva, yva):
    """Linear probe (logreg) macro-F1 + accuracy; drops classes absent from train."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    keep = ytr >= 0
    if keep.sum() < 10 or len(np.unique(ytr[keep])) < 2:
        return None
    clf = LogisticRegression(max_iter=300, n_jobs=-1).fit(Xtr[keep], ytr[keep])
    m = yva >= 0
    pred = clf.predict(Xva[m])
    return dict(task=name, macro_f1=float(f1_score(yva[m], pred, average="macro")),
                acc=float(accuracy_score(yva[m], pred)))


def save_curves_plot(curves, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [c["epoch"] for c in curves]
        reg_loss = [c["train_reg"] for c in curves]
        path_loss = [c["train_pathway"] for c in curves]

        probe_epochs = [c["epoch"] for c in curves if "probe_drug_f1" in c]
        tasks = ["drug", "moa", "cell_line"]
        colors = {"drug": "#009688", "moa": "#7e57c2", "cell_line": "#0f2d50"}
        has_probes = len(probe_epochs) > 0

        fig, axes = plt.subplots(1, 3 if has_probes else 2,
                                 figsize=(15 if has_probes else 10, 4.5))
        PRIM = "#0f2d50"

        # training losses (linear scale)
        ax = axes[0]
        ax.plot(epochs, reg_loss, color=PRIM, lw=1.5, label="reg loss (SIGReg/VICReg)")
        ax.plot(epochs, path_loss, color="#009688", lw=1.5, label="pathway coherence")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title("Training losses", color=PRIM)
        ax.legend(); ax.grid(True, alpha=0.3)

        # training losses (log scale)
        ax = axes[1]
        ax.semilogy(epochs, reg_loss, color=PRIM, lw=1.5, label="reg loss")
        ax.semilogy(epochs, path_loss, color="#009688", lw=1.5, label="pathway")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)")
        ax.set_title("Training losses (log scale)", color=PRIM)
        ax.legend(); ax.grid(True, alpha=0.3)

        # probe F1 over epochs (only on probe checkpoints)
        if has_probes:
            ax = axes[2]
            for t in tasks:
                key = f"probe_{t}_f1"
                vals = [c[key] for c in curves if key in c]
                if vals:
                    ax.plot(probe_epochs[:len(vals)], vals, color=colors[t],
                            marker="o", markersize=3, lw=1.5, label=t)
            ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1")
            ax.set_title("Linear probe F1 (frozen encoder)", color=PRIM)
            ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        fig.suptitle("Tahoe-JEPA training curves", color=PRIM, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"curves plot -> {out_path}")
    except Exception as e:
        print(f"[warn] could not save curves plot: {e}")


def run(fname, overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed); np.random.seed(cfg.meta.seed)

    dcfg = TahoeConfig(cache_path=cfg.data.cache_path, drop_frac=cfg.data.drop_frac,
                       noise_std=cfg.data.noise_std, val_fraction=cfg.data.val_fraction,
                       seed=cfg.meta.seed)
    tr, va, train_loader, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    K, D = tr.K, cfg.model.dstc
    print(f"== tahoe-jepa | device={device} | cells train={len(tr)} val={len(va)} | "
          f"genes={K} modules={tr.n_modules} reg={cfg.loss.reg} epochs={cfg.optim.epochs} ==")

    enc = CellEncoder(K, cfg.model.henc, D).to(device)
    proj = Projector(f"{D}-{cfg.model.proj}-{cfg.model.proj}").to(device)
    if cfg.loss.reg == "sigreg":
        reg = BCS(num_slices=cfg.loss.num_slices, lmbd=cfg.loss.lmbd).to(device)
    else:
        reg = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff).to(device)
    pathway = PathwayCoherenceLoss().to(device)
    pw = cfg.loss.pathway_coeff

    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = CosineWithWarmup(opt, total_steps=max(1, len(train_loader) * cfg.optim.epochs),
                             warmup_ratio=0.1, min_lr=cfg.optim.lr * 0.01)

    ckpt_dir = os.environ.get("EBJEPA_CKPTS", "checkpoints/tahoe")
    os.makedirs(ckpt_dir, exist_ok=True)
    curves_json = os.path.join(ckpt_dir, "tahoe_jepa_curves.json")
    curves_png = os.path.join(ckpt_dir, "tahoe_jepa_curves.png")
    curves = []
    best_f1 = -1.0

    for ep in range(1, cfg.optim.epochs + 1):
        enc.train(); proj.train()
        agg = {"reg": 0.0, "path": 0.0, "n": 0}
        for v1, v2, P, *_ in train_loader:
            v1, v2, P = v1.to(device), v2.to(device), P.to(device)
            z1, z2 = enc(v1), enc(v2)
            rl = reg(proj(z1), proj(z2))["loss"]
            pl = pathway(z1, P)
            total = rl + pw * pl
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            b = v1.shape[0]; agg["reg"] += rl.item()*b; agg["path"] += pl.item()*b; agg["n"] += b
        n = max(1, agg["n"])
        row = {"epoch": ep, "train_reg": agg["reg"]/n, "train_pathway": agg["path"]/n}
        msg = f"[ep {ep:03d}/{cfg.optim.epochs}] reg={row['train_reg']:.4f} pathway={row['train_pathway']:.4f}"

        if ep % cfg.optim.probe_every == 0 or ep == cfg.optim.epochs:
            Etr = embed(enc, tr.X[tr.ids], device); Eva = embed(enc, va.X[va.ids], device)
            for tname, arr in [("drug", "drug"), ("moa", "moa"), ("cell_line", "cell_line")]:
                ytr = getattr(tr, arr)[tr.ids].numpy(); yva = getattr(va, arr)[va.ids].numpy()
                if tname == "moa":
                    unclear = tr.moa_names.index("unclear") if "unclear" in tr.moa_names else -999
                    ytr = np.where(ytr == unclear, -1, ytr); yva = np.where(yva == unclear, -1, yva)
                r = probe(tname, Etr, ytr, Eva, yva)
                if r:
                    msg += f" | {tname}: F1={r['macro_f1']:.3f} acc={r['acc']:.3f}"
                    row[f"probe_{tname}_f1"] = r["macro_f1"]
                    row[f"probe_{tname}_acc"] = r["acc"]
            # checkpoint best by drug F1
            drug_f1 = row.get("probe_drug_f1", -1.0)
            if drug_f1 > best_f1:
                best_f1 = drug_f1
                msg += " *BEST*"
                torch.save({"enc": enc.state_dict(), "cfg": OmegaConf.to_container(cfg), "epoch": ep},
                           os.path.join(ckpt_dir, "tahoe_jepa_best.pt"))

        print(msg, flush=True)
        curves.append(row)
        # incremental write so training is inspectable without waiting for the end
        with open(curves_json, "w") as f:
            json.dump(curves, f, indent=2)

    # final checkpoint
    torch.save({"enc": enc.state_dict(), "cfg": OmegaConf.to_container(cfg),
                "epoch": cfg.optim.epochs},
               os.path.join(ckpt_dir, "tahoe_jepa.pt"))
    print(f"saved final -> {os.path.join(ckpt_dir, 'tahoe_jepa.pt')}")

    save_curves_plot(curves, curves_png)

    # final baselines vs ours (GeneJEPA protocol: linear probe on frozen features)
    print("\n== final linear-probe comparison (macro-F1) ==")
    Xtr_raw = tr.X[tr.ids].numpy(); Xva_raw = va.X[va.ids].numpy()
    Etr = embed(enc, tr.X[tr.ids], device); Eva = embed(enc, va.X[va.ids], device)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(D, 50)).fit(Xtr_raw)
    Ptr, Pva = pca.transform(Xtr_raw), pca.transform(Xva_raw)
    for tname, arr in [("cell_line", "cell_line"), ("drug", "drug"), ("moa", "moa")]:
        ytr = getattr(tr, arr)[tr.ids].numpy(); yva = getattr(va, arr)[va.ids].numpy()
        if tname == "moa":
            u = tr.moa_names.index("unclear") if "unclear" in tr.moa_names else -999
            ytr = np.where(ytr == u, -1, ytr); yva = np.where(yva == u, -1, yva)
        for feat, Xt, Xv in [("raw", Xtr_raw, Xva_raw), ("pca50", Ptr, Pva), ("JEPA(ours)", Etr, Eva)]:
            r = probe(tname, Xt, ytr, Xv, yva)
            if r: print(f"  {tname:10s} {feat:11s} F1={r['macro_f1']:.3f} acc={r['acc']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/train.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
