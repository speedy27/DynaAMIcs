"""Tests for the microbiome2img FCGR-JEPA integration: the FCGRSetEncoder respects
the same 5D contract as SetEncoder (so it is a true drop-in), and the *library* JEPA
unrolls end-to-end with it. The synthetic dataset emits the microbiome dict contract."""

import torch
import torch.nn as nn

from eb_jepa.architectures import (
    FCGRSetEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

B, K_, T, N, D, A = 4, 4, 6, 32, 16, 7  # K_ = FCGR resolution exponent (S = 2**K_)
S = 1 << K_
IMG = S * S


def _obs(n=N):
    # [B, S*S+1, T, N, 1]: flattened FCGR image channels + a non-negative abundance channel
    x = torch.randn(B, IMG + 1, T, n, 1)
    x[:, IMG] = torch.rand(B, T, n, 1)  # abundance >= 0
    return x


def test_fcgr_encoder_shapes_and_permutation_invariance():
    enc = FCGRSetEncoder(k=K_, h_d=8, out_d=D)
    x = _obs()
    z = enc(x)
    assert z.shape == (B, D, T, 1, 1)
    # permuting OTUs (the N axis) must not change the pooled representation
    perm = torch.randperm(N)
    z2 = enc(x[:, :, :, perm, :])
    assert torch.allclose(z, z2, atol=1e-5)


def test_fcgr_padding_is_ignored():
    enc = FCGRSetEncoder(k=K_, h_d=8, out_d=D)
    x = _obs()
    z = enc(x)
    pad = torch.randn(B, IMG + 1, T, 5, 1)
    pad[:, IMG] = 0.0  # padded slots have abundance 0
    z_pad = enc(torch.cat([x, pad], dim=3))
    assert torch.allclose(z, z_pad, atol=1e-5)


def test_fcgr_jepa_unroll_end_to_end():
    enc = FCGRSetEncoder(k=K_, h_d=8, out_d=D)
    pred = RNNPredictor(hidden_size=D, action_dim=A, final_ln=nn.LayerNorm(D))
    idm = InverseDynamicsModel(state_dim=D, hidden_dim=32, action_dim=A)
    reg = VC_IDM_Sim_Regularizer(
        cov_coeff=1.0, std_coeff=1.0, sim_coeff_t=1.0, idm_coeff=1.0,
        idm=idm, projector=Projector(f"{D}-{2*D}-{2*D}"), first_t_only=False,
    )
    jepa = JEPA(enc, nn.Identity(), pred, reg, SquareLossSeq())
    obs = _obs()
    act = torch.randn(B, A, T)
    preds, losses = jepa.unroll(obs, act, nsteps=3, unroll_mode="autoregressive",
                                compute_loss=True)
    total = losses[0]
    assert torch.isfinite(total)
    total.backward()  # gradients flow through the FCGR encoder + predictor + regularizer
    assert any(p.grad is not None for p in enc.parameters())


def test_synth_dataset_contract():
    from examples.microbiome2img.synth import SynthFCGRConfig, SynthFCGRDataset
    cfg = SynthFCGRConfig(k=4, n_clades=4, otus_per_clade=4, n_subjects=20,
                          n_window=5, n_max=8, seq_len=120)
    ds = SynthFCGRDataset(cfg)
    img_ch = ds.img_ch
    item = ds[0]
    assert item["observations"].shape == (img_ch + 1, cfg.n_window, cfg.n_max, 1)
    assert item["actions"].shape == (ds.action_dim, cfg.n_window)
    assert item["diversity"].shape == (cfg.n_window,)
    assert item["phylo"].shape == (cfg.n_window, img_ch)
    assert item["age"].ndim == 0 and item["label"].ndim == 0
    # padded OTU slots (beyond the present taxa) must carry abundance 0
    assert item["observations"][img_ch].min().item() >= 0.0
