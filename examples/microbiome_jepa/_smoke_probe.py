"""WS4 synthetic smoke test (CPU, no real data).

Exercises the full downstream-eval path end to end with SYNTHETIC communities and
RANDOM labels, so it needs none of the cluster-only files. Every metric printed here
is a genuine sklearn fit on random features (~= chance); it is NOT a microbiome result
and must not be reported as one.

Checks:
  1. random SetTransformerEncoder + synthetic OTUSampleDataset(mode="single") + fake
     (label, group, tech) -> encode_samples -> Z [N, D].
  2. linear_probe (LogReg + MLP, stratified AND grouped) returns FINITE acc/AUC.
  3. tech_invariance returns FINITE accuracy + a majority floor.
  4. baselines_port (Susagi LogReg + MLP + grouped MLP) on synthetic raw abundances
     returns FINITE metrics.
  5. run() fire entry (no checkpoint -> random encoder) returns finite probe metrics.

Run:
  /Users/bnz/DynaAMIcs/.venv-cpu/bin/python examples/microbiome_jepa/_smoke_probe.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

# Make `examples...` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.microbiome_jepa import baselines_port as bp  # noqa: E402
from examples.microbiome_jepa.probe_downstream import (  # noqa: E402
    encode_samples,
    linear_probe,
    run,
    tech_invariance,
)
from eb_jepa.architectures import SetTransformerEncoder  # noqa: E402
from eb_jepa.datasets.microbiome.otu_data import (  # noqa: E402
    OTUDatasetConfig,
    OTUSampleDataset,
)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _assert_finite(d: dict, keys, ctx: str):
    for k in keys:
        v = d.get(k)
        assert _finite(v), f"[{ctx}] metric {k!r} not finite: {v!r}"


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cpu")
    failures = []

    # --- 1. synthetic single-mode dataset + random encoder ---
    n_samples, n_max, n_classes, n_tech = 200, 48, 3, 2
    cfg = OTUDatasetConfig(
        synthetic=True, mode="single", n_max=n_max,
        synth_n_samples=n_samples, synth_seed=0,
    )
    ds = OTUSampleDataset(cfg)
    assert len(ds) == n_samples, f"dataset size {len(ds)} != {n_samples}"
    F = ds.token_dim
    enc = SetTransformerEncoder(
        token_dim=F, d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, pool="mean"
    ).to(dev)

    Z, metas = encode_samples(enc, ds, dev, batch_size=64)
    print(f"[1] encode_samples -> Z {Z.shape}, metas[0]={metas[0]}")
    assert Z.ndim == 2 and Z.shape[0] == n_samples, f"bad Z shape {Z.shape}"
    assert np.isfinite(Z).all(), "Z contains non-finite values"

    rng = np.random.default_rng(0)
    y = rng.integers(0, n_classes, size=n_samples)
    tech = rng.integers(0, n_tech, size=n_samples)
    groups = rng.integers(0, n_samples // 8, size=n_samples)

    # --- 2. linear_probe: stratified LogReg, stratified MLP, grouped LogReg ---
    m_lin = linear_probe(Z, y, task="clf", probe="linear")
    print(f"[2a] linear_probe (stratified, LogReg): {m_lin}")
    _assert_finite(m_lin, ["acc_mean", "acc_std", "auc_macro_mean"], "linear_probe/linear")

    m_mlp = linear_probe(Z, y, task="clf", probe="mlp")
    print(f"[2b] linear_probe (stratified, MLP):    {m_mlp}")
    _assert_finite(m_mlp, ["acc_mean", "auc_macro_mean"], "linear_probe/mlp")

    m_grp = linear_probe(Z, y, groups=groups, task="clf", probe="linear")
    print(f"[2c] linear_probe (grouped, LogReg):    {m_grp}")
    _assert_finite(m_grp, ["acc_mean", "auc_macro_mean"], "linear_probe/grouped")
    assert m_grp["cv"] == "grouped", "grouped CV not selected when groups passed"

    # --- 3. tech_invariance ---
    t = tech_invariance(Z, tech)
    print(f"[3] tech_invariance: {t}")
    _assert_finite(t, ["tech_acc_mean", "tech_majority_rate"], "tech_invariance")

    # --- 4. baselines_port on synthetic raw abundances ---
    # Raw-feature surrogate: per-sample masked mean of the raw z-scored OTU tokens.
    X_raw = np.stack([
        (ds[i][0]["otu"][0] * ds[i][0]["mask"][0].float().unsqueeze(-1)).sum(0).numpy()
        / ds[i][0]["mask"][0].float().sum().clamp(min=1.0).item()
        for i in range(n_samples)
    ])
    print(f"[4] raw feature matrix X_raw {X_raw.shape}")

    b_lr = bp.run_baseline(X_raw, y, model="logreg")
    print(f"[4a] susagi baseline LogReg:      {b_lr}")
    _assert_finite(b_lr, ["acc_mean", "auc_macro_mean"], "baseline/logreg")

    b_mlp = bp.run_baseline(X_raw, y, model="mlp")
    print(f"[4b] susagi baseline MLP:         {b_mlp}")
    _assert_finite(b_mlp, ["acc_mean", "auc_macro_mean"], "baseline/mlp")

    b_grp = bp.run_baseline(X_raw, y, groups=groups, model="mlp_grouped")
    print(f"[4c] susagi baseline MLP grouped: {b_grp}")
    _assert_finite(b_grp, ["acc_mean", "auc_macro_mean"], "baseline/mlp_grouped")

    # --- 5. run() fire entry, synthetic (no checkpoint -> random encoder) ---
    res = run(checkpoint=None, task="synthetic", probe="linear",
              synth_n_samples=160, n_max=n_max, device="cpu", run_baseline_too=True)
    print("[5] run() jepa_probe:", res["jepa_probe"])
    print("[5] run() susagi_baseline:", res.get("susagi_baseline"))
    _assert_finite(res["jepa_probe"], ["acc_mean", "auc_macro_mean"], "run/jepa_probe")

    print("\nSMOKE OK: all downstream-eval paths ran and returned finite metrics "
          "(synthetic, random labels -> ~chance; NOT a microbiome result).")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except AssertionError as e:
        print(f"\nSMOKE FAILED: {e}")
        rc = 1
    sys.exit(rc)
