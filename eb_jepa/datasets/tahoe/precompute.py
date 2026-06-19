"""
precompute.py - Build a compact cache for the Tahoe-100M cell-state JEPA.

Reads a directory of Tahoe-100M parquet shards (tahoebio/Tahoe-100M), fixes a
top-K highly-expressed gene panel, and stores each cell as a dense K-vector over
that panel + its labels (drug, MoA, cell line). Also builds a per-drug Morgan
fingerprint table (from canonical_smiles) for the action-conditioned variant.

Output cache (torch.save):
  panel      : int64 [K]            gene ids defining the panel
  X          : float16 [N, K]       per-cell expression over the panel
  drug       : int64 [N]            drug label id
  moa        : int64 [N]            MoA label id ('unclear' -> -1 mapped to 0)
  cell_line  : int64 [N]            cell-line label id
  drug_names, moa_names, cl_names   : label vocabularies
  drug_fp    : float32 [n_drugs, F] Morgan fingerprint per drug (action)

Usage:
  python -m eb_jepa.datasets.tahoe.precompute --shards artifacts/tahoe --out artifacts/tahoe/cache.pt --panel 2000
"""

import argparse
import glob
import os

import numpy as np
import torch


def morgan_fp(smiles, n_bits=256, radius=2):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return np.zeros(n_bits, np.float32)
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        a = np.zeros(n_bits, np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(bv, a)
        return a
    except Exception:
        return np.zeros(n_bits, np.float32)


def main():
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="artifacts/tahoe", help="dir with train-*.parquet")
    ap.add_argument("--out", default="artifacts/tahoe/cache.pt")
    ap.add_argument("--panel", type=int, default=2000, help="top-K gene panel size")
    ap.add_argument("--max-cells", type=int, default=0, help="cap total cells (0=all)")
    ap.add_argument("--modules", type=int, default=32, help="co-expression modules (pathway proxy)")
    ap.add_argument("--fp-bits", type=int, default=256)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.shards, "*.parquet")))
    files = [f for f in files if "meta" not in os.path.basename(f).lower()]
    if not files:
        raise SystemExit(f"no parquet shards in {args.shards}")
    print(f"shards: {len(files)}")

    # pass 1: gene frequency -> panel, and collect rows
    freq = {}
    rows = []  # (genes, expr, drug, moa, cl, smiles)
    n = 0
    for fp in files:
        df = pq.read_table(fp, columns=["genes", "expressions", "drug", "moa-fine",
                                        "cell_line_id", "canonical_smiles"]).to_pandas()
        for g, e, dr, mo, cl, sm in zip(df["genes"], df["expressions"], df["drug"],
                                        df["moa-fine"], df["cell_line_id"], df["canonical_smiles"]):
            g = np.asarray(g, dtype=np.int64)
            for gi in g:
                freq[gi] = freq.get(gi, 0) + 1
            rows.append((g, np.asarray(e, dtype=np.float32), str(dr), str(mo), str(cl), str(sm)))
            n += 1
            if args.max_cells and n >= args.max_cells:
                break
        if args.max_cells and n >= args.max_cells:
            break
    print(f"cells: {n}  unique genes seen: {len(freq)}")

    panel = np.array([g for g, _ in sorted(freq.items(), key=lambda kv: -kv[1])[: args.panel]],
                     dtype=np.int64)
    panel.sort()
    gpos = {int(g): i for i, g in enumerate(panel)}
    K = len(panel)

    # vocabularies
    drug_names = sorted({r[2] for r in rows})
    moa_names = sorted({r[3] for r in rows})
    cl_names = sorted({r[4] for r in rows})
    di = {d: i for i, d in enumerate(drug_names)}
    mi = {m: i for i, m in enumerate(moa_names)}
    ci = {c: i for i, c in enumerate(cl_names)}
    smiles_of = {}
    for g, e, dr, mo, cl, sm in rows:
        smiles_of.setdefault(dr, sm)

    # pass 2: dense panel vectors
    X = np.zeros((n, K), dtype=np.float16)
    drug = np.zeros(n, np.int64); moa = np.zeros(n, np.int64); cell_line = np.zeros(n, np.int64)
    for j, (g, e, dr, mo, cl, sm) in enumerate(rows):
        keep = [(gpos[int(gi)], ev) for gi, ev in zip(g, e) if int(gi) in gpos]
        if keep:
            idx, vals = zip(*keep)
            X[j, list(idx)] = np.asarray(vals, np.float16)
        drug[j] = di[dr]; moa[j] = mi[mo]; cell_line[j] = ci[cl]

    drug_fp = np.stack([morgan_fp(smiles_of.get(d, ""), args.fp_bits) for d in drug_names], 0)

    # co-expression MODULES (proxy pathways): cluster panel genes by their
    # expression profile across cells -> per-gene module id. Used by the
    # PathwayCoherenceLoss to inject gene-program structure into the latent.
    from sklearn.cluster import KMeans
    Xl = np.log1p(np.clip(X.astype(np.float32), 0, None))
    sub = Xl[np.random.default_rng(0).choice(n, size=min(n, 5000), replace=False)]
    gp = sub.T  # [K, n_sub] gene profiles
    gp = (gp - gp.mean(1, keepdims=True)) / (gp.std(1, keepdims=True) + 1e-6)
    n_mod = min(args.modules, K)
    modules = KMeans(n_clusters=n_mod, n_init=4, random_state=0).fit_predict(gp).astype(np.int64)
    print(f"co-expression modules: {n_mod} (panel genes clustered)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dict(
        panel=torch.from_numpy(panel), X=torch.from_numpy(X),
        modules=torch.from_numpy(modules), n_modules=n_mod,
        drug=torch.from_numpy(drug), moa=torch.from_numpy(moa), cell_line=torch.from_numpy(cell_line),
        drug_names=drug_names, moa_names=moa_names, cl_names=cl_names,
        drug_fp=torch.from_numpy(drug_fp.astype(np.float32)),
    ), args.out)
    print(f"saved {args.out}  | X={X.shape}  drugs={len(drug_names)} moa={len(moa_names)} lines={len(cl_names)}"
          f"  ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
