"""
precompute_pbmc.py - Build a cache from the standard PBMC3k benchmark (the dataset
GeneJEPA/scGPT report cell-type linear-probe Macro-F1 on).

Loads scanpy's annotated pbmc3k (louvain immune cell types), and writes a cache in
our Tahoe-JEPA format with the cell TYPE placed in the `cell_line` label slot, so
examples/tahoe/main.py probes cell-type classification unchanged. Lets us situate
our cell-state JEPA on a literature benchmark (GeneJEPA 0.69 / scGPT 0.23).

NOTE: this is an IN-DOMAIN run (we train on PBMC3k), whereas GeneJEPA's number is
frozen TRANSFER from Tahoe pretraining -- not apples-to-apples, but the same dataset.

  python -m eb_jepa.datasets.tahoe.precompute_pbmc --out artifacts/pbmc/cache.pt
"""

import argparse
import os
import numpy as np
import torch


def main():
    import scanpy as sc
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/pbmc/cache.pt")
    ap.add_argument("--modules", type=int, default=24)
    args = ap.parse_args()

    ad = sc.datasets.pbmc3k_processed()           # annotated: .obs['louvain'] cell types
    X = np.asarray(ad.X, dtype=np.float32)         # already log-normalized + scaled
    ctype = ad.obs["louvain"].astype(str).to_numpy()
    cl_names = sorted(set(ctype)); ci = {c: i for i, c in enumerate(cl_names)}
    cell_line = np.array([ci[c] for c in ctype], np.int64)
    n, K = X.shape
    print(f"PBMC3k: cells={n} genes={K} cell_types={len(cl_names)} -> {cl_names}")

    from sklearn.cluster import KMeans
    gp = X.T; gp = (gp - gp.mean(1, keepdims=True)) / (gp.std(1, keepdims=True) + 1e-6)
    n_mod = min(args.modules, K)
    modules = KMeans(n_clusters=n_mod, n_init=4, random_state=0).fit_predict(gp).astype(np.int64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(
        panel=torch.arange(K), X=torch.from_numpy(X.astype(np.float16)), is_embedding=True,
        modules=torch.from_numpy(modules), n_modules=n_mod,
        drug=torch.zeros(n, dtype=torch.long), moa=torch.zeros(n, dtype=torch.long),
        cell_line=torch.from_numpy(cell_line),
        drug_names=["na"], moa_names=["unclear"], cl_names=cl_names,
        drug_fp=torch.zeros(1, 256),
    ), args.out)
    print(f"saved {args.out} | X={X.shape}")


if __name__ == "__main__":
    main()
