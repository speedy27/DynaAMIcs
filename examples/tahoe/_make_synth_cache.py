"""
_make_synth_cache.py — tiny SYNTHETIC Tahoe cache for LOCAL smoke runs.

⚠️ This is NOT real Tahoe-100M data. The real cache (artifacts/tahoe/cache.pt)
is built on the cluster from the raw parquet atlas. This script fabricates a
small cache with the SAME schema + a deliberately learnable structure
(cell-line / drug mean-shifts) so the full training + SIGReg + PathwayCoherence
+ linear-probe code path runs end-to-end on a laptop. Metrics are illustrative
of the PIPELINE, not biology.

    python examples/tahoe/_make_synth_cache.py
"""
import os
import numpy as np
import torch

rng = np.random.default_rng(0)
torch.manual_seed(0)

N, K, M = 3000, 256, 16          # cells, genes, co-expression modules
n_cl, n_drug, n_moa = 6, 10, 4   # cell lines, drugs, real MoA classes (+ 'unclear')

# gene -> module assignment
modules = torch.tensor(rng.integers(0, M, size=K), dtype=torch.long)

# class-specific expression patterns (the learnable signal)
cl_mean = torch.tensor(rng.normal(0, 1.0, size=(n_cl, K)), dtype=torch.float)
drug_mean = torch.tensor(rng.normal(0, 0.6, size=(n_drug, K)), dtype=torch.float)

cell_line = torch.tensor(rng.integers(0, n_cl, size=N), dtype=torch.long)
drug = torch.tensor(rng.integers(0, n_drug, size=N), dtype=torch.long)

# drug -> MoA (a few drugs are 'unclear', mirroring the real label noise)
drug_to_moa = rng.integers(0, n_moa, size=n_drug)
unclear_drugs = rng.choice(n_drug, size=3, replace=False)
moa_np = drug_to_moa[drug.numpy()]
moa_np = np.where(np.isin(drug.numpy(), unclear_drugs), n_moa, moa_np)  # n_moa == 'unclear'
moa = torch.tensor(moa_np, dtype=torch.long)

# count-like, non-negative expression matrix with structure + noise
noise = torch.tensor(rng.normal(0, 0.5, size=(N, K)), dtype=torch.float)
signal = 2.0 + cl_mean[cell_line] + drug_mean[drug] + noise
X = torch.clamp(signal, min=0.0) * 5.0

blob = dict(
    X=X, is_embedding=False,
    drug=drug, moa=moa, cell_line=cell_line,
    drug_names=[f"drug{i}" for i in range(n_drug)],
    moa_names=[f"moa{i}" for i in range(n_moa)] + ["unclear"],
    cl_names=[f"cl{i}" for i in range(n_cl)],
    drug_fp=torch.tensor(rng.integers(0, 2, size=(n_drug, 256)), dtype=torch.float),
    modules=modules, n_modules=M,
)

os.makedirs("artifacts/tahoe", exist_ok=True)
out = "artifacts/tahoe/cache.pt"
torch.save(blob, out)
print(f"[SYNTHETIC] wrote {out}: N={N} K={K} modules={M} "
      f"cell_lines={n_cl} drugs={n_drug} moas={n_moa}(+unclear)")
