"""UMAP exploration of the Tahoe-100M cells (MosaicFM-3B embeddings) for the
data-preprocessing section. Saves a multi-panel figure + individual panels.

  python examples/tahoe/slides/make_umap.py --cache $WORK/tahoe/cache_pert.pt --n 40000 --out examples/tahoe/slides
"""
import argparse, os, collections
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIM = "#0f2d50"; SEC = "#009688"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    b = torch.load(args.cache, weights_only=False)
    X = b["X"].float().numpy()
    drug = b["drug"].numpy(); cl = b["cell_line"].numpy()
    is_ctrl = b["is_control"].numpy() if "is_control" in b else np.zeros(len(X), bool)
    dn = b["drug_names"]; cln = b["cl_names"]
    n = min(args.n, len(X))
    idx = np.random.default_rng(0).choice(len(X), n, replace=False)
    X, drug, cl, is_ctrl = X[idx], drug[idx], cl[idx], is_ctrl[idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    print(f"UMAP on {n} cells x {X.shape[1]} ...")

    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(X)

    def scatter(ax, c, title, cmap="tab20", cbar=False, disc=True):
        s = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap=cmap, s=2, alpha=0.5,
                       rasterized=True)
        ax.set_title(title, color=PRIM, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        if cbar:
            plt.colorbar(s, ax=ax, fraction=0.046)
        return s

    # top-12 drugs, rest grey
    topd = [d for d, _ in collections.Counter(drug.tolist()).most_common(12)]
    dmap = {d: i for i, d in enumerate(topd)}
    dcol = np.array([dmap.get(d, -1) for d in drug])

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    scatter(ax[0, 0], cl, f"by cell line ({len(cln)} lines)", "tab20")
    sc = ax[0, 1].scatter(emb[:, 0], emb[:, 1], c=np.where(is_ctrl, 1, 0),
                          cmap="coolwarm", s=2, alpha=0.5, rasterized=True)
    ax[0, 1].set_title("control (DMSO) vs treated", color=PRIM); ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])
    m = dcol >= 0
    ax[1, 0].scatter(emb[~m, 0], emb[~m, 1], c="lightgrey", s=2, alpha=0.3, rasterized=True)
    ax[1, 0].scatter(emb[m, 0], emb[m, 1], c=dcol[m], cmap="tab20", s=3, alpha=0.7, rasterized=True)
    ax[1, 0].set_title("by drug (top 12 colored)", color=PRIM); ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
    scatter(ax[1, 1], np.linalg.norm(X, axis=1), "embedding magnitude", "viridis", cbar=True, disc=False)
    fig.suptitle("Tahoe-100M (MosaicFM-3B embeddings) — UMAP of the training cells",
                 color=PRIM, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "umap_panel.png"), dpi=150)
    print("saved", os.path.join(args.out, "umap_panel.png"))

    # standalone: cell-line UMAP (clean, for the deck)
    fig2, a2 = plt.subplots(figsize=(7, 6))
    scatter(a2, cl, f"Tahoe cells by cell line ({len(cln)})", "tab20")
    fig2.tight_layout(); fig2.savefig(os.path.join(args.out, "umap_celllines.png"), dpi=150)
    print("saved umap_celllines.png")


if __name__ == "__main__":
    main()
