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

Training curves saved to:
  <ckpt>/tahoe_jepa_curves.json   (incremental, one entry per epoch)
  <ckpt>/tahoe_jepa_curves.png    (train / val / test losses + probe F1 per task)

Split: val_fraction=0.2 -> train 80%, then val split 50/50 -> val 10%, test 10%.
SSL losses (reg + pathway) logged every epoch on train, val, test.
Linear probes logged every probe_every epochs.
Early stopping on drug probe F1 (patience_probes consecutive probe evaluations).

  python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml optim.epochs=250
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
def compute_ssl_loss(enc, proj, reg_fn, pathway_fn, loader, device):
    """SSL loss (reg + pathway) on a dataloader — no gradient, eval mode."""
    enc.eval(); proj.eval()
    agg = {"reg": 0.0, "path": 0.0, "n": 0}
    for v1, v2, P, *_ in loader:
        v1, v2, P = v1.to(device), v2.to(device), P.to(device)
        z1, z2 = enc(v1), enc(v2)
        rl = reg_fn(proj(z1), proj(z2))["loss"]
        pl = pathway_fn(z1, P)
        b = v1.shape[0]
        agg["reg"] += rl.item() * b; agg["path"] += pl.item() * b; agg["n"] += b
    n = max(1, agg["n"])
    return {"reg": agg["reg"] / n, "pathway": agg["path"] / n}


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
        PRIM = "#0f2d50"; VAL_C = "#009688"; TEST_C = "#e67e22"
        probe_epochs = [c["epoch"] for c in curves if "probe_drug_f1" in c]
        has_probes = len(probe_epochs) > 0

        fig, axes = plt.subplots(1, 3 if has_probes else 2, figsize=(15 if has_probes else 10, 4.5))

        # panel 1: reg loss on train / val / test
        ax = axes[0]
        ax.plot(epochs, [c["train_reg"] for c in curves], color=PRIM, lw=1.5, label="train")
        if any("val_reg" in c for c in curves):
            ax.plot(epochs, [c.get("val_reg", float("nan")) for c in curves],
                    color=VAL_C, lw=1.5, ls="--", label="val")
        if any("test_reg" in c for c in curves):
            ax.plot(epochs, [c.get("test_reg", float("nan")) for c in curves],
                    color=TEST_C, lw=1.5, ls=":", label="test")
        ax.set_xlabel("epoch"); ax.set_ylabel("reg loss")
        ax.set_title("Reg loss (train / val / test)", color=PRIM)
        ax.legend(); ax.grid(True, alpha=0.3)

        # panel 2: pathway loss on train / val / test (log scale)
        ax = axes[1]
        ax.semilogy(epochs, [c["train_pathway"] for c in curves], color=PRIM, lw=1.5, label="train")
        if any("val_pathway" in c for c in curves):
            ax.semilogy(epochs, [c.get("val_pathway", float("nan")) for c in curves],
                        color=VAL_C, lw=1.5, ls="--", label="val")
        if any("test_pathway" in c for c in curves):
            ax.semilogy(epochs, [c.get("test_pathway", float("nan")) for c in curves],
                        color=TEST_C, lw=1.5, ls=":", label="test")
        ax.set_xlabel("epoch"); ax.set_ylabel("pathway loss (log)")
        ax.set_title("Pathway loss train/val/test", color=PRIM)
        ax.legend(); ax.grid(True, alpha=0.3)

        # panel 3: probe F1 per task
        if has_probes:
            tasks = ["drug", "moa", "cell_line"]
            colors = {"drug": "#009688", "moa": "#7e57c2", "cell_line": "#0f2d50"}
            ax = axes[2]
            for t in tasks:
                key = f"probe_{t}_f1"
                vals = [c[key] for c in curves if key in c]
                if vals:
                    ax.plot(probe_epochs[:len(vals)], vals, color=colors[t],
                            marker="o", markersize=3, lw=1.5, label=t)
            stopped = next((c for c in reversed(curves) if c.get("early_stop")), None)
            if stopped:
                ax.axvline(stopped["epoch"], ls="--", color="crimson", lw=1.2,
                           label=f"early stop (ep {stopped['epoch']})")
            ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1")
            ax.set_title("Linear probe F1 (frozen encoder)", color=PRIM)
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

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
    tr, va, train_loader, _ = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    K, D = tr.K, cfg.model.dstc
    patience_probes = int(cfg.optim.get("patience_probes", 5))

    # split val 50/50 -> val (10%) and test (10%) of total
    n_half = len(va) // 2
    val_loader = DataLoader(Subset(va, range(n_half)), batch_size=cfg.data.batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(Subset(va, range(n_half, len(va))), batch_size=cfg.data.batch_size,
                             shuffle=False, drop_last=False)

    print(f"== tahoe-jepa | device={device} | train={len(tr)} val={n_half} test={len(va)-n_half} | "
          f"genes={K} modules={tr.n_modules} reg={cfg.loss.reg} "
          f"epochs={cfg.optim.epochs} patience_probes={patience_probes} ==")

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
    probes_since_best = 0
    t0 = time.time()

    for ep in range(1, cfg.optim.epochs + 1):
        enc.train(); proj.train()
        ep_t0 = time.time()
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

        # val / test SSL losses every epoch (fast forward-only passes)
        val_m = compute_ssl_loss(enc, proj, reg, pathway, val_loader, device)
        test_m = compute_ssl_loss(enc, proj, reg, pathway, test_loader, device)
        enc.train(); proj.train()

        ep_time = time.time() - ep_t0
        elapsed = time.time() - t0
        eta_s = (elapsed / ep) * (cfg.optim.epochs - ep)
        row = {
            "epoch": ep,
            "train_reg": agg["reg"] / n,  "train_pathway": agg["path"] / n,
            "val_reg": val_m["reg"],       "val_pathway": val_m["pathway"],
            "test_reg": test_m["reg"],     "test_pathway": test_m["pathway"],
            "epoch_time_s": round(ep_time, 2),
        }
        msg = (f"[ep {ep:03d}/{cfg.optim.epochs}] "
               f"reg tr={row['train_reg']:.4f} va={row['val_reg']:.4f} te={row['test_reg']:.4f} | "
               f"path tr={row['train_pathway']:.4f} va={row['val_pathway']:.4f} te={row['test_pathway']:.4f} | "
               f"t={ep_time:.1f}s eta={eta_s/60:.0f}m")

        do_probe = (ep % cfg.optim.probe_every == 0 or ep == cfg.optim.epochs)
        if do_probe:
            # probe on full val set (indices 0..n_half) using the underlying va dataset
            Etr = embed(enc, tr.X[tr.ids], device)
            Eva = embed(enc, va.X[va.ids[:n_half]], device)
            for tname, arr in [("drug", "drug"), ("moa", "moa"), ("cell_line", "cell_line")]:
                ytr_full = getattr(tr, arr)[tr.ids].numpy()
                yva_full = getattr(va, arr)[va.ids[:n_half]].numpy()
                if tname == "moa":
                    unclear = tr.moa_names.index("unclear") if "unclear" in tr.moa_names else -999
                    ytr_full = np.where(ytr_full == unclear, -1, ytr_full)
                    yva_full = np.where(yva_full == unclear, -1, yva_full)
                r = probe(tname, Etr, ytr_full, Eva, yva_full)
                if r:
                    msg += f" | {tname}: F1={r['macro_f1']:.3f}"
                    row[f"probe_{tname}_f1"] = r["macro_f1"]
                    row[f"probe_{tname}_acc"] = r["acc"]
            drug_f1 = row.get("probe_drug_f1", -1.0)
            if drug_f1 > best_f1:
                best_f1 = drug_f1; probes_since_best = 0; msg += " *BEST*"
                torch.save({"enc": enc.state_dict(), "cfg": OmegaConf.to_container(cfg), "epoch": ep},
                           os.path.join(ckpt_dir, "tahoe_jepa_best.pt"))
            else:
                probes_since_best += 1

        print(msg, flush=True)
        curves.append(row)
        with open(curves_json, "w") as f:
            json.dump(curves, f, indent=2)

        if do_probe and probes_since_best >= patience_probes:
            print(f"[early stop] no improvement for {probes_since_best} probe evals "
                  f"({probes_since_best * cfg.optim.probe_every} epochs) — stopping at ep {ep}.",
                  flush=True)
            curves[-1]["early_stop"] = True
            with open(curves_json, "w") as f:
                json.dump(curves, f, indent=2)
            break

    torch.save({"enc": enc.state_dict(), "cfg": OmegaConf.to_container(cfg),
                "epoch": curves[-1]["epoch"]},
               os.path.join(ckpt_dir, "tahoe_jepa.pt"))
    total_time = time.time() - t0
    print(f"saved -> {os.path.join(ckpt_dir, 'tahoe_jepa.pt')} | total: {total_time/60:.1f} min")
    save_curves_plot(curves, curves_png)

    print("\n== final linear-probe comparison (macro-F1) ==")
    Xtr_raw = tr.X[tr.ids].numpy(); Xva_raw = va.X[va.ids[:n_half]].numpy()
    Etr = embed(enc, tr.X[tr.ids], device); Eva = embed(enc, va.X[va.ids[:n_half]], device)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(D, 50)).fit(Xtr_raw)
    Ptr, Pva = pca.transform(Xtr_raw), pca.transform(Xva_raw)
    for tname, arr in [("cell_line", "cell_line"), ("drug", "drug"), ("moa", "moa")]:
        ytr = getattr(tr, arr)[tr.ids].numpy(); yva = getattr(va, arr)[va.ids[:n_half]].numpy()
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
