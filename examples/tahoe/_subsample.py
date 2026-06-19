import sys, torch, numpy as np
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
b = torch.load(src, weights_only=False)
N = b["X"].shape[0]
idx = np.random.default_rng(0).choice(N, min(n, N), replace=False)
idx = np.sort(idx)
out = dict(b)
for k in ["X", "drug", "cell_line", "is_control"]:
    out[k] = b[k][idx]
torch.save(out, dst)
print("subsampled", len(idx), "cells ->", dst)
print("n_drugs", len(b["drug_names"]), "controls", int(out["is_control"].sum()))
