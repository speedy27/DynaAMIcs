"""
precompute_emb.py - Same Tahoe-JEPA pipeline, but the encoder INPUT is the frozen
PRETRAINED MosaicFM-3B embedding (Tahoe-x1) instead of a raw gene panel.

Reads a Tahoe-x1-embeddings parquet shard (each row already carries a 2560-d
MosaicFM embedding + drug + cell_line) and writes a cache in the SAME format as
precompute.py, so examples/tahoe/main.py runs UNCHANGED. The only differences:
  - X = pretrained embedding (real-valued) -> is_embedding=True (dataset skips log1p)
  - "modules" = clusters of embedding dimensions (program-coherence prior)
  - moa mapped from drug via drug_metadata.parquet if present

The main.py "raw" baseline then = the frozen MosaicFM embedding (foundation model),
and "JEPA(ours)" = our head trained on top with SIGReg + PathwayCoherenceLoss.

  python -m eb_jepa.datasets.tahoe.precompute_emb --shard $WORK/tahoe_emb/emb0.parquet \
      --out $WORK/tahoe/cache_emb.pt --max-cells 300000 --modules 32
"""

import argparse
import os
import numpy as np
import torch

EMB_COL = "mosaicfm-3b-prod-cont-MFMv2"


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="artifacts/tahoe/cache_emb.pt")
    ap.add_argument("--drug-meta", default="", help="optional drug_metadata.parquet for MoA labels")
    ap.add_argument("--max-cells", type=int, default=300000)
    ap.add_argument("--modules", type=int, default=32)
    args = ap.parse_args()

    pf = pq.ParquetFile(args.shard)
    Xs, drug, cl = [], [], []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=["drug", "cell_line", EMB_COL]).to_pandas()
        Xs.append(np.stack(df[EMB_COL].to_numpy()).astype(np.float16))
        drug += df["drug"].tolist(); cl += df["cell_line"].tolist()
        if args.max_cells and sum(len(x) for x in Xs) >= args.max_cells:
            break
    X = np.concatenate(Xs)
    if args.max_cells:
        X = X[: args.max_cells]; drug = drug[: len(X)]; cl = cl[: len(X)]
    n, K = X.shape
    print(f"cells={n} emb_dim={K}")

    # drug -> MoA (optional)
    moa_of = {}
    moa_names = ["unclear"]
    if args.drug_meta and os.path.exists(args.drug_meta):
        dm = pq.read_table(args.drug_meta).to_pandas()
        col = "moa-fine" if "moa-fine" in dm.columns else None
        if col:
            moa_of = {str(r["drug"]): str(r[col]) for _, r in dm.iterrows()}
            moa_names = sorted(set(moa_of.values()) | {"unclear"})

    drug_names = sorted(set(drug)); cl_names = sorted(set(cl))
    di = {d: i for i, d in enumerate(drug_names)}
    ci = {c: i for i, c in enumerate(cl_names)}
    mi = {m: i for i, m in enumerate(moa_names)}
    drug_id = np.array([di[d] for d in drug], np.int64)
    cl_id = np.array([ci[c] for c in cl], np.int64)
    moa_id = np.array([mi.get(moa_of.get(d, "unclear"), 0) for d in drug], np.int64)

    # "modules" = clusters of embedding dimensions (program-coherence prior)
    from sklearn.cluster import KMeans
    sub = X[np.random.default_rng(0).choice(n, size=min(n, 5000), replace=False)].astype(np.float32)
    gp = sub.T  # [K, n_sub]
    gp = (gp - gp.mean(1, keepdims=True)) / (gp.std(1, keepdims=True) + 1e-6)
    n_mod = min(args.modules, K)
    modules = KMeans(n_clusters=n_mod, n_init=4, random_state=0).fit_predict(gp).astype(np.int64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(
        panel=torch.arange(K), X=torch.from_numpy(X), is_embedding=True,
        modules=torch.from_numpy(modules), n_modules=n_mod,
        drug=torch.from_numpy(drug_id), moa=torch.from_numpy(moa_id), cell_line=torch.from_numpy(cl_id),
        drug_names=drug_names, moa_names=moa_names, cl_names=cl_names,
        drug_fp=torch.zeros(len(drug_names), 256),
    ), args.out)
    print(f"saved {args.out} | X={X.shape} drugs={len(drug_names)} lines={len(cl_names)} moa={len(moa_names)}"
          f" ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
