"""
probe_emb.py - Use a PRETRAINED cell encoder (MosaicFM-3B, Tahoe-x1 embeddings)
instead of training one from scratch.

Each row of the Tahoe-x1-embeddings parquet already carries a 2560-d MosaicFM
embedding + drug + cell_line (no join needed). We:
  1. linear-probe the FROZEN pretrained embedding   -> the foundation-model baseline
  2. train a light JEPA refinement on top (two-view Gaussian-noise SSL + SIGReg)
     and linear-probe that                           -> does our objective add signal?

  python -m examples.tahoe.probe_emb --shard $WORK/tahoe_emb/emb0.parquet --max-cells 200000
"""

import argparse
import numpy as np
import torch
import torch.nn as nn


EMB_COL = "mosaicfm-3b-prod-cont-MFMv2"


def load(shard, max_cells):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(shard)
    need = ["drug", "cell_line", EMB_COL]
    X, drug, cl = [], [], []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=need).to_pandas()
        X.append(np.stack(df[EMB_COL].to_numpy()).astype(np.float32))
        drug += df["drug"].tolist(); cl += df["cell_line"].tolist()
        if max_cells and sum(len(x) for x in X) >= max_cells:
            break
    X = np.concatenate(X)[:max_cells] if max_cells else np.concatenate(X)
    drug, cl = drug[:len(X)], cl[:len(X)]
    di = {d: i for i, d in enumerate(sorted(set(drug)))}
    ci = {c: i for i, c in enumerate(sorted(set(cl)))}
    y_d = np.array([di[d] for d in drug]); y_c = np.array([ci[c] for c in cl])
    return X, y_d, y_c


def probe(Xtr, Xva, ytr, yva, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    clf = LogisticRegression(max_iter=300, n_jobs=-1).fit(Xtr, ytr)
    p = clf.predict(Xva)
    return f"{name:16s} F1={f1_score(yva,p,average='macro'):.3f} acc={accuracy_score(yva,p):.3f}"


def jepa_refine(Xtr, device, d=128, epochs=30, noise=0.3, bs=2048, lmbd=10.0):
    from eb_jepa.architectures import Projector
    from eb_jepa.losses import BCS
    K = Xtr.shape[1]
    enc = nn.Sequential(nn.Linear(K, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Linear(512, d)).to(device)
    proj = Projector(f"{d}-{4*d}-{4*d}").to(device)
    reg = BCS(num_slices=256, lmbd=lmbd).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=device)
    for ep in range(epochs):
        enc.train(); proj.train(); perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i+bs]]
            if xb.shape[0] < 4: continue
            v1 = xb + noise * torch.randn_like(xb); v2 = xb + noise * torch.randn_like(xb)
            loss = reg(proj(enc(v1)), proj(enc(v2)))["loss"]
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        return lambda X: enc(torch.tensor(X, device=device)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--max-cells", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {args.shard} ...")
    X, y_d, y_c = load(args.shard, args.max_cells)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    print(f"cells={len(X)} dim={X.shape[1]} drugs={y_d.max()+1} cell_lines={y_c.max()+1}")

    rng = np.random.default_rng(args.seed); idx = rng.permutation(len(X))
    nval = len(X) // 5; va, tr = idx[:nval], idx[nval:]

    print("\n== PRETRAINED MosaicFM-3B embedding (frozen) vs our JEPA-refinement ==")
    for tname, y in [("cell_line", y_c), ("drug", y_d)]:
        print(f"-- {tname} --")
        print("  " + probe(X[tr], X[va], y[tr], y[va], "MosaicFM(frozen)"))
        from sklearn.decomposition import PCA
        P = PCA(n_components=128).fit(X[tr])
        print("  " + probe(P.transform(X[tr]), P.transform(X[va]), y[tr], y[va], "PCA-128"))
        enc = jepa_refine(X[tr], device)
        print("  " + probe(enc(X[tr]), enc(X[va]), y[tr], y[va], "JEPA-refine(ours)"))


if __name__ == "__main__":
    main()
