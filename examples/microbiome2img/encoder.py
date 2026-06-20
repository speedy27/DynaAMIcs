"""
encoder.py -- the image encoder for "DNA as picture".

`FCGREncoder` maps an OTU's FCGR image [B, 1, 2^k, 2^k] to an embedding [B, D] with
a small conv net -- the drop-in replacement for the ProkBERT token embedding. Used
two ways:
  * RANDOM (untrained) -> a fair "is it the imaging + conv inductive bias?" baseline,
    mirroring the random-encoder baseline of the main microbiome example.
  * TRAINED -> plugged under the set-transformer + temporal JEPA (integration step).

`FCGRSetEncoder` shows the intended swap: per-OTU FCGR -> conv -> abundance-weighted
pool, the FCGR analogue of the ProkBERT `SetEncoder`.
"""

import torch
import torch.nn as nn


class FCGREncoder(nn.Module):
    """Small CNN: FCGR image [B, 1, S, S] (S = 2^k) -> embedding [B, out_dim]."""

    def __init__(self, k: int = 6, out_dim: int = 128, width: int = 32):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=2, padding=1), nn.GELU(),        # S/2
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.GELU(),  # S/4
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), nn.GELU(),  # S/8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * width, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        if x.dim() == 3:          # [B, S, S] -> add channel
            x = x.unsqueeze(1)
        return self.net(x)


class FCGRSetEncoder(nn.Module):
    """Community encoder over a SET of OTU FCGR images, abundance-weighted.

    Input  : images [B, N, S, S] + abundance weights [B, N]
    Output : community embedding [B, out_dim]
    This is the FCGR counterpart of the ProkBERT-based `SetEncoder`: permutation
    invariant over OTUs, padded slots (weight 0) ignored for free.
    """

    def __init__(self, k: int = 6, out_dim: int = 128, width: int = 32):
        super().__init__()
        self.token = FCGREncoder(k=k, out_dim=out_dim, width=width)
        self.out_dim = out_dim

    def forward(self, images, weights):
        b, n = images.shape[:2]
        s = images.shape[-1]
        z = self.token(images.reshape(b * n, 1, s, s)).reshape(b, n, -1)  # [B, N, D]
        w = weights.clamp_min(0)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        return (z * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
