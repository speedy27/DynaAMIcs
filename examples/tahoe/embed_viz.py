"""
embed_viz.py — 2D embedding of the perturbation world-model OUTPUTS, colored by drug.

Loads the trained checkpoint (predictor), runs the action-conditioned prediction
ẑ_pert = g_φ(z_ctrl, drug) over a sample of cells, and projects to 2D (UMAP, else
t-SNE, else PCA). Two panels:
  left : predicted state ẑ_pert   (dominated by cell identity)
  right: predicted SHIFT Δ = ẑ_pert − z_ctrl   (drug-specific → should cluster by drug)

  python -m examples.tahoe.embed_viz --cache $WORK/tahoe/cache_pert_small.pt \
      --ckpt $WORK/checkpoints/tahoe_perturb.pt --out artifacts/tahoe/umap_by_drug.png
"""
import argparse, os
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from eb_jepa.architectures import RNNPredictor
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq
from eb_jepa.datasets.tahoe.pert_dataset import PertConfig, make_loaders
from examples.tahoe.perturb import FrozenIdentityEncoder, NoReg, _states


def embed2d(X):
    try:
        import umap
        return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, random_state=0).fit_transform(X), "UMAP"
    except Exception:
        try:
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, init="pca", random_state=0).fit_transform(X), "t-SNE"
        except Exception:
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=0).fit_transform(X), "PCA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="artifacts/tahoe/umap_by_drug.png")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blob = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    cfg = OmegaConf.create(blob["cfg"])
    dcfg = PertConfig(cache_path=args.cache, val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed)
    tr, va, _, val_loader = make_loaders(dcfg, batch_size=512, num_workers=0)
    D, A = tr.D, tr.action_dim

    enc = FrozenIdentityEncoder()
    nl = int(cfg.model.get("layers", 1)) if "model" in cfg else 1
    pred = RNNPredictor(hidden_size=D, action_dim=A, num_layers=nl, final_ln=nn.LayerNorm(D))
    jepa = JEPA(enc, nn.Identity(), pred, NoReg(), SquareLossSeq()).to(device)
    jepa.load_state_dict(blob["jepa"]); jepa.eval()

    Ps, Ss, Ds = [], [], []
    with torch.no_grad():
        for b in val_loader:
            obs = b["observations"].to(device); act = b["actions"].to(device)
            z_ctrl, _ = _states(obs)
            preds, _ = jepa.unroll(obs, act, nsteps=1, unroll_mode="autoregressive", compute_loss=False)
            p = preds[:, :, -1, 0, 0]
            Ps.append(p.cpu().numpy()); Ss.append((p - z_ctrl).cpu().numpy()); Ds.append(b["drug"].numpy())
            if sum(len(x) for x in Ds) >= args.n:
                break
    P = np.concatenate(Ps)[:args.n]; S = np.concatenate(Ss)[:args.n]; drug = np.concatenate(Ds)[:args.n]

    # keep the top-K most frequent drugs for a legible plot
    uniq, cnt = np.unique(drug, return_counts=True)
    top = uniq[np.argsort(-cnt)[:args.topk]]
    keep = np.isin(drug, top)
    P, S, drug = P[keep], S[keep], drug[keep]
    names = tr.drug_names

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, X, title in [(axes[0], P, "predicted state  ẑ_pert"), (axes[1], S, "predicted shift  Δ = ẑ_pert − z_ctrl")]:
        emb, algo = embed2d(X)
        for d in top:
            m = drug == d
            ax.scatter(emb[m, 0], emb[m, 1], s=7, alpha=0.7, label=str(names[int(d)])[:16])
        ax.set_title(f"{title}  ({algo}, colored by drug)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(markerscale=2, fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    fig.suptitle("World-model outputs in 2D — do same-drug cells group?", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved -> {args.out}  | cells={len(drug)} drugs(top)={len(top)}")


if __name__ == "__main__":
    main()
