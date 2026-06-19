"""
precompute_pert.py - Cache for the action-conditioned cell perturbation WORLD MODEL
(frozen MosaicFM-3B encoder + drug-conditioned predictor).

From a Tahoe-x1-embeddings shard (2560-d MosaicFM embedding + drug + cell_line per
cell) it builds:
  X         : float16 [N, 2560]  frozen pretrained cell embedding (the JEPA state)
  drug, cell_line : int64 [N]
  is_control: bool  [N]          DMSO/vehicle cells (auto-detected); else all False
  centroid  : float32 [n_lines, 2560]  per-cell-line mean (pseudo-control fallback)
  drug_fp   : float32 [n_drugs, F]     Morgan fingerprint per drug (the ACTION)
  modules   : int64  [2560]            embedding-dim clusters (pathway-coherence prior)

If real DMSO controls exist we pair treated<-control within a cell line; otherwise
the cell-line centroid is the control state. The action a = drug fingerprint.

  python -m eb_jepa.datasets.tahoe.precompute_pert --shard $WORK/tahoe_emb/emb0.parquet \
      --drug-meta $WORK/tahoe_data/drug_meta.parquet --out $WORK/tahoe/cache_pert.pt --max-cells 300000
"""

import argparse
import os
import numpy as np
import torch

EMB_COL = "mosaicfm-3b-prod-cont-MFMv2"


def morgan(smiles, nb=256, r=2):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import ConvertToNumpyArray
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return np.zeros(nb, np.float32)
        a = np.zeros(nb, np.float32)
        ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=nb), a)
        return a
    except Exception:
        return np.zeros(nb, np.float32)


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--drug-meta", default="")
    ap.add_argument("--out", default="artifacts/tahoe/cache_pert.pt")
    ap.add_argument("--max-cells", type=int, default=300000)
    ap.add_argument("--modules", type=int, default=32)
    ap.add_argument("--fp-bits", type=int, default=256)
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

    drug_names = sorted(set(drug)); cl_names = sorted(set(cl))
    di = {d: i for i, d in enumerate(drug_names)}; ci = {c: i for i, c in enumerate(cl_names)}
    drug_id = np.array([di[d] for d in drug], np.int64)
    cl_id = np.array([ci[c] for c in cl], np.int64)

    # auto-detect control (DMSO / vehicle / control)
    def is_ctrl(name):
        s = str(name).upper()
        return ("DMSO" in s) or ("VEHICLE" in s) or ("CONTROL" in s)
    is_control = np.array([is_ctrl(d) for d in drug], bool)
    print(f"control cells: {int(is_control.sum())}  ({'DMSO found' if is_control.any() else 'NONE -> centroid fallback'})")

    # cell-line centroids (pseudo-control fallback)
    Xf = X.astype(np.float32)
    centroid = np.zeros((len(cl_names), K), np.float32)
    for c in range(len(cl_names)):
        m = cl_id == c
        if m.any():
            centroid[c] = Xf[m].mean(0)

    # drug -> SMILES -> fingerprint (action)
    smiles = {}
    if args.drug_meta and os.path.exists(args.drug_meta):
        dm = pq.read_table(args.drug_meta).to_pandas()
        smiles = {str(r["drug"]): str(r.get("canonical_smiles", "")) for _, r in dm.iterrows()}
    drug_fp = np.stack([morgan(smiles.get(d, ""), args.fp_bits) for d in drug_names], 0)
    n_with_fp = int((drug_fp.sum(1) > 0).sum())
    print(f"drugs with Morgan fingerprint: {n_with_fp}/{len(drug_names)}")
    if n_with_fp == 0:  # no RDKit / no SMILES -> one-hot drug action (still informative)
        drug_fp = np.eye(len(drug_names), dtype=np.float32)
        print(f"  -> fallback: one-hot drug action (dim={len(drug_names)})")

    # embedding-dim modules (pathway-coherence prior)
    from sklearn.cluster import KMeans
    sub = Xf[np.random.default_rng(0).choice(n, size=min(n, 5000), replace=False)]
    gp = sub.T; gp = (gp - gp.mean(1, keepdims=True)) / (gp.std(1, keepdims=True) + 1e-6)
    modules = KMeans(n_clusters=min(args.modules, K), n_init=4, random_state=0).fit_predict(gp).astype(np.int64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(
        X=torch.from_numpy(X), drug=torch.from_numpy(drug_id), cell_line=torch.from_numpy(cl_id),
        is_control=torch.from_numpy(is_control), centroid=torch.from_numpy(centroid),
        drug_fp=torch.from_numpy(drug_fp.astype(np.float32)),
        modules=torch.from_numpy(modules), n_modules=int(min(args.modules, K)),
        drug_names=drug_names, cl_names=cl_names, emb_dim=K,
    ), args.out)
    print(f"saved {args.out} | X={X.shape} drugs={len(drug_names)} lines={len(cl_names)} "
          f"({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
