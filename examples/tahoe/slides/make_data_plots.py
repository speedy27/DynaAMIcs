"""Exploratory data plots for the preprocessing section (from the gene-panel cache)."""
import os, collections
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
PRIM = "#0f2d50"; SEC = "#009688"
b = torch.load(os.path.join(HERE, "..", "..", "..", "artifacts", "tahoe", "cache.pt"), weights_only=False)
X = b["X"].float().numpy(); drug = b["drug"].numpy(); cl = b["cell_line"].numpy(); moa = b["moa"].numpy()
dn = b["drug_names"]; cln = b["cl_names"]; mn = b["moa_names"]; mods = b["modules"].numpy()
print("cells", len(X), "genes", X.shape[1])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# cells per cell line (top 20)
cc = collections.Counter(cl.tolist()).most_common(20)
ax[0, 0].barh([cln[i][:14] for i, _ in cc][::-1], [v for _, v in cc][::-1], color=PRIM)
ax[0, 0].set_title("cells per cell line (top 20)", color=PRIM); ax[0, 0].tick_params(labelsize=7)

# cells per drug (top 20)
dc = collections.Counter(drug.tolist()).most_common(20)
ax[0, 1].barh([dn[i][:16] for i, _ in dc][::-1], [v for _, v in dc][::-1], color=SEC)
ax[0, 1].set_title("cells per compound (top 20)", color=PRIM); ax[0, 1].tick_params(labelsize=7)

# expressed genes per cell (sparsity)
nnz = (X != 0).sum(1)
ax[1, 0].hist(nnz, bins=40, color=PRIM, alpha=0.85)
ax[1, 0].set_title(f"expressed genes / cell (panel={X.shape[1]})", color=PRIM)
ax[1, 0].set_xlabel("# non-zero genes"); ax[1, 0].set_ylabel("cells")

# MoA distribution (top, excluding unclear)
mc = [(mn[i], v) for i, v in collections.Counter(moa.tolist()).most_common() if mn[i] != "unclear"][:10]
ax[1, 1].barh([m[:22] for m, _ in mc][::-1], [v for _, v in mc][::-1], color="#7e57c2")
ax[1, 1].set_title("cells per mechanism-of-action (top 10)", color=PRIM); ax[1, 1].tick_params(labelsize=7)

fig.suptitle("Tahoe-100M subset — data composition", color=PRIM, fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "data_composition.png"), dpi=150)
print("saved data_composition.png")

# co-expression module sizes (the pathway prior)
fig2, a2 = plt.subplots(figsize=(7, 3.6))
sz = collections.Counter(mods.tolist())
a2.bar(range(len(sz)), [sz[i] for i in range(len(sz))], color=SEC)
a2.set_title("Co-expression module sizes (PathwayCoherenceLoss prior)", color=PRIM)
a2.set_xlabel("module"); a2.set_ylabel("# genes")
fig2.tight_layout(); fig2.savefig(os.path.join(HERE, "modules.png"), dpi=150)
print("saved modules.png")
