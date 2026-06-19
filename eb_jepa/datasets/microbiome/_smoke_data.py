"""WS1 smoke test: microbiome OTU-set + gLV-trajectory datasets, NO real data.

Run:
    /Users/bnz/DynaAMIcs/.venv-cpu/bin/python eb_jepa/datasets/microbiome/_smoke_data.py

Checks (per tasks/01-data.md):
  1. OTUSampleDataset(mode="two_view", synthetic=True) -> DataLoader(bs=4): shapes
     view1["otu"]==[4,1,N_max,F], view1["mask"]==[4,1,N_max], F==385, float32/bool,
     and view1 != view2 (augmentation differs).
  2. CLR+z-score: fitted train features have per-dim mean ~= 0, std ~= 1.
  3. GLVTrajDataset through TrajSlicerDataset(num_frames=4): obs["otu"]==[4,N_max,F],
     act==[4,K].
  4. init_data("microbiome", cfg_data={...}) returns a 4-tuple with working loaders.
"""

import sys
import traceback

import torch

# Make the repo root importable when run as a script.
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from eb_jepa.datasets.microbiome.otu_data import (  # noqa: E402
    OTUSampleDataset,
    OTUDatasetConfig,
    TOKEN_DIM,
)
from eb_jepa.datasets.microbiome.traj import GLVTrajDataset, GLVTrajConfig  # noqa: E402
from eb_jepa.datasets.traj_dset import TrajSlicerDataset  # noqa: E402
from eb_jepa.datasets.utils import init_data  # noqa: E402


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    ok = True

    # ----------------------------------------------------------------- check 1
    section("CHECK 1 — OTUSampleDataset two_view (synthetic) + DataLoader")
    n_max = 64
    ds = OTUSampleDataset(
        OTUDatasetConfig(mode="two_view", n_max=n_max, synth_n_samples=64),
        synthetic=True,
    )
    print(f"dataset len={len(ds)}  token_dim(F)={ds.token_dim}  n_max={ds.n_max}  "
          f"is_synthetic={ds.is_synthetic}")
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    view1, view2 = next(iter(loader))
    print("view1['otu'] :", tuple(view1["otu"].shape), view1["otu"].dtype)
    print("view1['mask']:", tuple(view1["mask"].shape), view1["mask"].dtype)
    print("view2['otu'] :", tuple(view2["otu"].shape), view2["otu"].dtype)
    assert view1["otu"].shape == (4, 1, n_max, TOKEN_DIM), view1["otu"].shape
    assert view1["mask"].shape == (4, 1, n_max), view1["mask"].shape
    assert TOKEN_DIM == 385, TOKEN_DIM
    assert view1["otu"].dtype == torch.float32
    assert view1["mask"].dtype == torch.bool
    diff = (view1["otu"] - view2["otu"]).abs().sum().item()
    mask_diff = (view1["mask"] ^ view2["mask"]).sum().item()
    print(f"view1 vs view2: sum|otu diff|={diff:.4f}   mask XOR count={mask_diff}")
    assert diff > 0.0, "augmentation produced identical views (otu)"
    print("CHECK 1 PASS")

    # ----------------------------------------------------------------- check 2
    section("CHECK 2 — CLR + per-dim z-score statistics")
    # Apply the FITTED z-score to all RAW (pre-zscore) tokens, masked to real OTUs.
    raw_tokens = torch.cat([s["tokens"] for s in ds._raw], dim=0)   # [sum_n, F]
    raw_mask = torch.cat([s["mask"] for s in ds._raw], dim=0)       # [sum_n]
    z = ds.zscore.transform(raw_tokens)
    real = z[raw_mask]                                              # only real tokens
    per_dim_mean = real.mean(dim=0)
    per_dim_std = real.std(dim=0, unbiased=False)
    print(f"real-token rows used: {real.shape[0]}  (F={real.shape[1]})")
    print(f"per-dim mean: min={per_dim_mean.min():.3e} max={per_dim_mean.max():.3e} "
          f"abs-max={per_dim_mean.abs().max():.3e}")
    print(f"per-dim std : min={per_dim_std.min():.4f} max={per_dim_std.max():.4f} "
          f"mean={per_dim_std.mean():.4f}")
    print(f"abundance dim (last, idx {TOKEN_DIM - 1}): "
          f"mean={per_dim_mean[-1]:.3e}  std={per_dim_std[-1]:.4f}")
    assert per_dim_mean.abs().max() < 1e-4, per_dim_mean.abs().max().item()
    assert torch.allclose(per_dim_std, torch.ones_like(per_dim_std), atol=1e-3), \
        per_dim_std
    print("CHECK 2 PASS  (mean ~= 0, std ~= 1 across all F dims incl. abundance)")

    # ----------------------------------------------------------------- check 3
    section("CHECK 3 — GLVTrajDataset through TrajSlicerDataset(num_frames=4)")
    num_frames = 4
    gds = GLVTrajDataset(GLVTrajConfig(n_traj=8, T=16, n_species=12, n_candidate=4))
    print(f"GLVTrajDataset: len={len(gds)}  used_stub_glv={gds.used_stub}  "
          f"n_max(S)={gds.n_max}  action_dim(K)={gds.action_dim}  "
          f"state_dim(S)={gds.state_dim}  proprio_dim={gds.proprio_dim}  "
          f"seq_len={gds.get_seq_length(0)}")
    # Sanity-check the raw 5-tuple before slicing.
    obs0, act0, state0, reward0, extra0 = gds[0]
    print("raw obs['otu']:", tuple(obs0["otu"].shape), obs0["otu"].dtype,
          " obs['mask']:", tuple(obs0["mask"].shape), obs0["mask"].dtype)
    print("raw act:", tuple(act0.shape), " state:", tuple(state0.shape),
          " reward:", tuple(reward0.shape))
    slicer = TrajSlicerDataset(gds, num_frames=num_frames, frameskip=1)
    print(f"slicer len={len(slicer)}  action_dim={slicer.action_dim}  "
          f"state_dim={slicer.state_dim}  proprio_dim={slicer.proprio_dim}")
    obs, act, state, reward = slicer[0]
    print("sliced obs['otu'] :", tuple(obs["otu"].shape), obs["otu"].dtype)
    print("sliced obs['mask']:", tuple(obs["mask"].shape), obs["mask"].dtype)
    print("sliced act:", tuple(act.shape), " state:", tuple(state.shape),
          " reward:", tuple(reward.shape))
    K = gds.action_dim
    assert obs["otu"].shape == (num_frames, gds.n_max, TOKEN_DIM), obs["otu"].shape
    assert obs["mask"].shape == (num_frames, gds.n_max), obs["mask"].shape
    assert act.shape == (num_frames, K), act.shape
    assert obs["otu"].dtype == torch.float32 and obs["mask"].dtype == torch.bool
    # Confirm it also collates through a DataLoader.
    gloader = torch.utils.data.DataLoader(slicer, batch_size=4, shuffle=False)
    b_obs, b_act, b_state, b_reward = next(iter(gloader))
    print("batched obs['otu']:", tuple(b_obs["otu"].shape),
          " act:", tuple(b_act.shape))
    assert b_obs["otu"].shape == (4, num_frames, gds.n_max, TOKEN_DIM)
    assert b_act.shape == (4, num_frames, K)
    print("CHECK 3 PASS")

    # ----------------------------------------------------------------- check 4
    section("CHECK 4 — init_data('microbiome', ...) returns working loaders")
    # 4a: static OTU task
    tr, va, cfg, mgr = init_data(
        "microbiome",
        cfg_data={"task": "otu", "synthetic": True, "batch_size": 4,
                  "n_max": 32, "synth_n_samples": 40},
    )
    print(f"[otu]  4-tuple: train={type(tr).__name__} val={type(va).__name__} "
          f"cfg.size={cfg.size} cfg.token_dim={cfg.token_dim} "
          f"cfg.n_max={cfg.n_max} mgr={mgr}")
    assert mgr is None
    v1, v2 = next(iter(tr))
    print("       train batch view1['otu']:", tuple(v1["otu"].shape))
    assert v1["otu"].shape[-1] == 385 and v1["otu"].shape[1] == 1
    nv = next(iter(va))
    print("       val batch ok:", tuple(nv[0]["otu"].shape))

    # 4b: gLV temporal task
    tr2, va2, cfg2, mgr2 = init_data(
        "microbiome",
        cfg_data={"task": "glv", "batch_size": 4, "num_frames": 4,
                  "n_traj": 8, "T": 16, "n_species": 12, "n_candidate": 4},
    )
    print(f"[glv]  4-tuple: cfg.size={cfg2.size} action_dim={cfg2.action_dim} "
          f"state_dim={cfg2.state_dim} n_max={cfg2.n_max} mgr={mgr2} "
          f"used_stub={cfg2.extra.get('used_stub_glv')}")
    assert mgr2 is None
    gobs, gact, gstate, grew = next(iter(tr2))
    print("       train batch obs['otu']:", tuple(gobs["otu"].shape),
          " act:", tuple(gact.shape))
    assert gobs["otu"].shape == (4, 4, 12, 385), gobs["otu"].shape
    assert gact.shape == (4, 4, cfg2.action_dim)
    print("CHECK 4 PASS")

    section("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\nSMOKE FAILED with exception:")
        traceback.print_exc()
        sys.exit(1)
