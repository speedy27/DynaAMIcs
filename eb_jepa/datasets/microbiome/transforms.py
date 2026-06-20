"""Transforms for the microbiome modality (WS1).

All functions/classes here operate on the OTU-token representation shared with the
encoder workstream (WS2):

  token feature dim  F = D_emb + 1   (D_emb = 384 ProkBERT dims + 1 log-abundance)
  community at one timepoint = set of up to N_max tokens [N_max, F] + bool mask [N_max]

The pipeline that produces a token, per the OBS/TOKEN CONTRACT, is:

  abundances (relative, sum-to-1, sparse)
      --clr-->            centered-log-ratio of the abundances
      --z-score-->        z(CLR_log_abundance)                       (1 dim)
  ProkBERT embedding (384) --z-score--> z(ProkBERT_384)              (384 dims)
  token = concat( z(ProkBERT_384), z(CLR_log_abundance) )           (F = 385 dims)

`clr` is applied to abundances BEFORE z-scoring; `PerDimZScore` is fit on the
TRAIN tokens (all F dims) and persisted so val/test/eval reuse the same stats.

Pitfall guarded here (see CLAUDE.md): the VICReg variance term measures per-dim
std, so EVERY feature dim (the 384 ProkBERT dims AND the single abundance dim)
must be z-scored, otherwise the abundance dim dwarfs the rest and dominates both
the encoder and the regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# CLR (centered log-ratio)
# ---------------------------------------------------------------------------
def clr(abund: Tensor, pseudocount: float = 1e-6, dim: int = -1) -> Tensor:
    """Centered log-ratio transform of compositional (relative) abundances.

    CLR(x)_i = log(x_i) - mean_j log(x_j), computed over the OTU axis `dim`.

    Microbiome abundances are relative (sum-to-1) and very sparse, so a positive
    `pseudocount` is added before the log to keep zeros finite. The transform is
    applied per-community (over the OTU axis), which removes the sum-to-1
    constraint and yields features in an unbounded real space suitable for a
    downstream z-score.

    Args:
        abund: non-negative abundances, e.g. [..., N] (relative or counts).
        pseudocount: added before log to handle zeros / sparsity.
        dim: the OTU axis to take the geometric-mean reference over.

    Returns:
        CLR-transformed tensor, same shape as `abund`, float.
    """
    abund = abund.to(torch.float32)
    # Guard against negatives from numerical noise.
    x = torch.clamp(abund, min=0.0) + pseudocount
    log_x = torch.log(x)
    gmean = log_x.mean(dim=dim, keepdim=True)
    return log_x - gmean


# ---------------------------------------------------------------------------
# Per-dimension z-score (fit / transform; serializable)
# ---------------------------------------------------------------------------
@dataclass
class PerDimZScore:
    """Per-feature-dimension standardization over the F-dim token features.

    Fit on the TRAIN tokens; stores `mean`/`std` (each [F]) so the same stats are
    reused at transform time (val/test/eval). `eps` floors the std to avoid
    division by zero on (near-)constant dims.

    Serializable via `state_dict()` / `load_state_dict()` (plain tensors) so the
    fitted stats can be persisted next to a checkpoint.
    """

    mean: Optional[Tensor] = None
    std: Optional[Tensor] = None
    eps: float = 1e-6

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, tokens: Tensor, mask: Optional[Tensor] = None) -> "PerDimZScore":
        """Estimate per-dim mean/std.

        Args:
            tokens: [..., F] feature tensor (any leading shape is flattened).
            mask:   optional boolean tensor broadcastable to tokens' leading dims
                    (True = real token). If given, padding tokens are excluded
                    from the statistics.
        """
        feat = tokens.reshape(-1, tokens.shape[-1]).to(torch.float32)
        if mask is not None:
            keep = mask.reshape(-1).to(torch.bool)
            feat = feat[keep]
        if feat.numel() == 0:
            raise ValueError("PerDimZScore.fit got zero (unmasked) tokens to fit on")
        self.mean = feat.mean(dim=0)
        self.std = feat.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def transform(self, tokens: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("PerDimZScore.transform called before fit()")
        mean = self.mean.to(tokens.device, tokens.dtype)
        std = self.std.to(tokens.device, tokens.dtype)
        return (tokens - mean) / std

    def fit_transform(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.fit(tokens, mask=mask).transform(tokens)

    # -- serialization -------------------------------------------------------
    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, sd: dict) -> "PerDimZScore":
        self.mean = sd["mean"]
        self.std = sd["std"]
        self.eps = sd.get("eps", self.eps)
        return self


# ---------------------------------------------------------------------------
# Two-view SSL augmentation (subsample OTUs + abundance jitter + dropout)
# ---------------------------------------------------------------------------
def augment_community(
    tokens: Tensor,
    mask: Tensor,
    *,
    subsample_frac: float = 0.8,
    jitter_std: float = 0.1,
    dropout_p: float = 0.1,
    abund_dim: int = -1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Produce one stochastic augmented view of a community (for two-view VICReg/BCS).

    Three augmentations, all applied only to REAL tokens (mask=True):
      1. random OTU subsample: keep each real OTU with prob `subsample_frac`;
      2. OTU dropout: additionally drop each (still-real) OTU with prob `dropout_p`;
      3. gaussian jitter on the (single) log-abundance feature dim `abund_dim`.

    Shapes are preserved: dropped/subsampled slots are zeroed in `tokens` and set
    False in `mask` (the encoder ignores them via the padding mask). At least one
    real OTU is always kept (a random survivor is restored if a view empties out),
    so a community never collapses to all-pad.

    Args:
        tokens: [N_max, F] (single community). Higher-rank inputs [..., N_max, F]
                are supported and augmented with the SAME per-sample logic.
        mask:   [N_max] (or [..., N_max]) bool, True = real.
        abund_dim: index (within F) of the log-abundance feature to jitter.
                   Jitter is applied in the (z-scored) feature space.

    Returns:
        (aug_tokens, aug_mask) with the same shapes/dtypes as the inputs.
    """
    if tokens.dim() < 2:
        raise ValueError(f"augment_community expects [..., N_max, F], got {tuple(tokens.shape)}")
    work_tokens = tokens.clone()
    work_mask = mask.clone().to(torch.bool)

    lead = work_mask.shape[:-1]
    n_max = work_mask.shape[-1]
    flat_mask = work_mask.reshape(-1, n_max)
    flat_tokens = work_tokens.reshape(-1, n_max, work_tokens.shape[-1])

    keep_prob = float(subsample_frac) * (1.0 - float(dropout_p))
    keep_prob = max(0.0, min(1.0, keep_prob))

    rand = torch.rand(flat_mask.shape, generator=generator, device=flat_mask.device)
    keep = flat_mask & (rand < keep_prob)

    # Ensure each community keeps >= 1 real OTU (where it had any).
    had_any = flat_mask.any(dim=-1)
    keeps_none = had_any & (~keep.any(dim=-1))
    if keeps_none.any():
        for row in torch.nonzero(keeps_none, as_tuple=False).flatten().tolist():
            real_idx = torch.nonzero(flat_mask[row], as_tuple=False).flatten()
            pick = real_idx[torch.randint(len(real_idx), (1,), generator=generator)]
            keep[row, pick] = True

    # Jitter the log-abundance feature on the surviving real tokens.
    if jitter_std and jitter_std > 0:
        noise = torch.randn(
            flat_tokens.shape[:-1], generator=generator, device=flat_tokens.device
        ) * float(jitter_std)
        ad = abund_dim if abund_dim >= 0 else flat_tokens.shape[-1] + abund_dim
        flat_tokens[..., ad] = flat_tokens[..., ad] + noise * keep.to(flat_tokens.dtype)

    # Zero out dropped slots and update mask.
    flat_tokens = flat_tokens * keep.unsqueeze(-1).to(flat_tokens.dtype)
    out_tokens = flat_tokens.reshape(*lead, n_max, work_tokens.shape[-1])
    out_mask = keep.reshape(*lead, n_max)
    return out_tokens, out_mask


# ---------------------------------------------------------------------------
# I-JEPA style OTU masking
# ---------------------------------------------------------------------------
def mask_otus(
    tokens: Tensor,
    mask: Tensor,
    frac: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split a community's real OTUs into a visible context and masked targets.

    For I-JEPA on a set: hide a fraction `frac` of the REAL OTUs; the predictor
    must predict the representations of the hidden ("target") OTUs from the
    visible ("context") ones.

    Args:
        tokens: [N_max, F] (single community).
        mask:   [N_max] bool, True = real.
        frac:   fraction of real OTUs to hold out as prediction targets.

    Returns:
        visible_tokens: [N_max, F] (target slots zeroed),
        visible_mask:   [N_max] bool (True only on visible context OTUs),
        target_idx:     LongTensor [n_target] indices (into N_max) of masked
                        target OTUs (order is arbitrary). Empty if no targets.
    """
    if tokens.dim() != 2:
        raise ValueError(f"mask_otus expects a single community [N_max, F], got {tuple(tokens.shape)}")
    mask = mask.to(torch.bool)
    real_idx = torch.nonzero(mask, as_tuple=False).flatten()
    n_real = real_idx.numel()
    if n_real == 0:
        return tokens.clone(), mask.clone(), real_idx.new_empty((0,))

    n_target = int(round(float(frac) * n_real))
    n_target = max(0, min(n_real - 1, n_target))  # keep >=1 visible context OTU

    perm = torch.randperm(n_real, generator=generator, device=real_idx.device)
    target_idx = real_idx[perm[:n_target]]

    visible_mask = mask.clone()
    visible_mask[target_idx] = False
    visible_tokens = tokens.clone()
    visible_tokens[target_idx] = 0.0
    return visible_tokens, visible_mask, target_idx
