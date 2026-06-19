"""WS2 smoke test: SetTransformerEncoder + ImposterRepulsionLoss.

Run:
    /Users/bnz/DynaAMIcs/.venv-cpu/bin/python eb_jepa/_smoke_encoder.py

Verifies (per tasks/02-encoder.md):
  1. obs dict -> output shape EXACTLY [B, D, T, 1, 1]
  2. permutation-invariance over N_max (permute otu+mask together): max-abs-diff < 1e-5
  3. mask-invariance (randomize padded slots): max-abs-diff < 1e-5
  4. output `state` composes with VC_IDM_Sim_Regularizer (finite weighted/unweighted/dict),
     and a per-timestep [B,D,1,1,1] slice composes with RNNPredictor(hidden_size=D)
  5. ImposterRepulsionLoss on random reps -> finite scalar
  + extras: variable N_max, all-but-one-masked community (no NaN).
"""

import os
import sys
import warnings

# When run as a script (`python eb_jepa/_smoke_encoder.py`), Python puts this
# file's dir (eb_jepa/) on sys.path[0], where eb_jepa/logging.py shadows the
# stdlib `logging` and breaks `import torch`. Drop that dir and ensure the repo
# root is importable so `import eb_jepa...` works exactly like under pytest.
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Cosmetic: nn.TransformerEncoder warns that it disables an (unused) nested-tensor
# fast path because norm_first=True. Harmless; silence for clean smoke output.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from eb_jepa.architectures import RNNPredictor, SetTransformerEncoder  # noqa: E402
from eb_jepa.losses import ImposterRepulsionLoss, VC_IDM_Sim_Regularizer  # noqa: E402


def main():
    torch.manual_seed(0)
    ok = True

    # ----- 1. build encoder + shape -----------------------------------------
    B, T, N_max, Fdim = 4, 3, 16, 385
    D = 128
    enc = SetTransformerEncoder(token_dim=Fdim, d_model=D)
    enc.eval()  # deterministic (dropout=0.0 anyway)

    otu = torch.randn(B, T, N_max, Fdim)
    # random mask with >= 1 real OTU per (B, T) row
    mask = torch.rand(B, T, N_max) > 0.5
    mask[..., 0] = True  # guarantee at least one real OTU per row
    obs = {"otu": otu, "mask": mask}

    with torch.no_grad():
        out = enc(obs)
    print(f"[1] output shape           : {tuple(out.shape)}  (expected (4, 128, 3, 1, 1))")
    print(f"    encoder.mlp_output_dim : {enc.mlp_output_dim}")
    print(f"    output finite          : {torch.isfinite(out).all().item()}")
    shape_ok = tuple(out.shape) == (B, D, T, 1, 1)
    ok &= shape_ok and bool(torch.isfinite(out).all())
    assert shape_ok, f"shape mismatch: {tuple(out.shape)}"

    # ----- 2. permutation-invariance over N_max -----------------------------
    perm = torch.randperm(N_max)
    otu_perm = otu[:, :, perm, :]
    mask_perm = mask[:, :, perm]
    with torch.no_grad():
        out_perm = enc({"otu": otu_perm, "mask": mask_perm})
    perm_diff = (out - out_perm).abs().max().item()
    print(f"[2] perm-invariance max|diff|: {perm_diff:.3e}  (expected < 1e-5)")
    ok &= perm_diff < 1e-5

    # ----- 3. mask-invariance (randomize padded slots) ----------------------
    otu_corrupt = otu.clone()
    pad = ~mask  # True where padded
    noise = torch.randn_like(otu_corrupt) * 100.0  # large junk in pad slots
    otu_corrupt = torch.where(pad.unsqueeze(-1), noise, otu_corrupt)
    with torch.no_grad():
        out_corrupt = enc({"otu": otu_corrupt, "mask": mask})
    mask_diff = (out - out_corrupt).abs().max().item()
    print(f"[3] mask-invariance max|diff|: {mask_diff:.3e}  (expected < 1e-5)")
    ok &= mask_diff < 1e-5

    # ----- 4a. compose with VC_IDM_Sim_Regularizer --------------------------
    reg = VC_IDM_Sim_Regularizer(cov_coeff=1, std_coeff=1, sim_coeff_t=1)
    state = out  # [B, D, T, 1, 1]
    weighted, unweighted, loss_dict = reg(state, actions=None)
    w_fin = bool(torch.isfinite(weighted).all())
    u_fin = bool(torch.isfinite(unweighted).all())
    print(
        f"[4a] regularizer weighted={weighted.item():.4f} finite={w_fin} | "
        f"unweighted={unweighted.item():.4f} finite={u_fin} | dict={loss_dict}"
    )
    ok &= w_fin and u_fin

    # ----- 4b. compose with RNNPredictor(hidden_size=D) ---------------------
    action_dim = 4
    pred = RNNPredictor(hidden_size=D, action_dim=action_dim, final_ln=nn.Identity())
    state_t = state[:, :, 0:1]  # [B, D, 1, 1, 1]
    print(f"[4b] state_t (predictor input) shape: {tuple(state_t.shape)}")
    action = torch.randn(B, action_dim, 1)
    with torch.no_grad():
        next_state = pred(state_t, action)
    pred_shape_ok = tuple(next_state.shape) == (B, D, 1, 1, 1)
    pred_fin = bool(torch.isfinite(next_state).all())
    print(
        f"     RNNPredictor out shape: {tuple(next_state.shape)} "
        f"(expected (4, 128, 1, 1, 1)) finite={pred_fin}"
    )
    ok &= pred_shape_ok and pred_fin

    # ----- 5. ImposterRepulsionLoss -----------------------------------------
    irl = ImposterRepulsionLoss(margin=1.0)
    N, Dr = 32, 64
    pred_rep = torch.randn(N, Dr)
    real_rep = torch.randn(N, Dr)
    imposter_rep = torch.randn(N, Dr)
    irl_val = irl(pred_rep, real_rep, imposter_rep)
    irl_fin = bool(torch.isfinite(irl_val).all())
    print(
        f"[5] ImposterRepulsionLoss   : {irl_val.item():.4f} "
        f"scalar={irl_val.dim() == 0} finite={irl_fin}"
    )
    # cosine variant too
    irl_cos = ImposterRepulsionLoss(margin=0.5, distance="cosine")
    irl_cos_val = irl_cos(pred_rep, real_rep, imposter_rep)
    print(f"    (cosine variant)        : {irl_cos_val.item():.4f}")
    ok &= irl_fin and irl_val.dim() == 0 and bool(torch.isfinite(irl_cos_val))

    # ----- extras: variable N_max + all-but-one-masked (no NaN) --------------
    enc2 = SetTransformerEncoder(token_dim=Fdim, d_model=64, pool="attention")
    enc2.eval()
    otu2 = torch.randn(2, 2, 7, Fdim)  # different N_max
    mask2 = torch.zeros(2, 2, 7, dtype=torch.bool)
    mask2[..., 0] = True  # exactly ONE real OTU per row (extreme case)
    with torch.no_grad():
        out2 = enc2({"otu": otu2, "mask": mask2})
    extras_ok = tuple(out2.shape) == (2, 64, 2, 1, 1) and bool(torch.isfinite(out2).all())
    print(
        f"[extra] variable N_max + 1-real-OTU (attention pool): shape={tuple(out2.shape)} "
        f"finite={bool(torch.isfinite(out2).all())}"
    )
    ok &= extras_ok

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
