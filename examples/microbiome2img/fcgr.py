"""
fcgr.py -- turn a DNA sequence (string of A/C/G/T) into an IMAGE.

Frequency Chaos Game Representation (FCGR): instead of feeding an OTU's SSU-rRNA
sequence to a DNA language model (ProkBERT), we render it as a 2^k x 2^k image
whose pixels are k-mer frequencies. Phylogenetically close sequences -> visually
close images, so a plain image encoder (the library's ImpalaEncoder / ResNet, the
mature image-JEPA path) can learn community structure from "DNA pictures".

Chaos Game Representation: start at the centre (0.5, 0.5) of the unit square whose
corners are the four bases; for each base, move halfway toward that base's corner.
After reading k bases the point lands in a unique sub-cell of a 2^k x 2^k grid that
corresponds one-to-one to the last k-mer -- so binning the trajectory at that
resolution yields the exact k-mer frequency spectrum.

Corner convention (only permutes which pixel a k-mer maps to, not validity):
    A = (0, 0)   C = (0, 1)   G = (1, 1)   T = (1, 0)

Public API:
    cgr_points(seq)                -> (x, y) trajectory in [0, 1]^2
    fcgr(seq, k=6, normalize=...)  -> [2^k, 2^k] float32 image
    fcgr_batch(seqs, k=6, ...)     -> [N, 2^k, 2^k] float32 stack
"""

from __future__ import annotations

import numpy as np

# (x, y) corner per base in the unit square
_CORNERS = {
    "A": (0.0, 0.0),
    "C": (0.0, 1.0),
    "G": (1.0, 1.0),
    "T": (1.0, 0.0),
}


def cgr_points(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """Chaos-game trajectory of a sequence. Non-ACGT bases (N, gaps, ...) are
    skipped. Returns two arrays (x, y) of the visited points in [0, 1]^2."""
    seq = seq.upper()
    corners = [_CORNERS[b] for b in seq if b in _CORNERS]
    n = len(corners)
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    px, py = 0.5, 0.5
    for i, (cx, cy) in enumerate(corners):
        px = (px + cx) * 0.5
        py = (py + cy) * 0.5
        x[i] = px
        y[i] = py
    return x, y


def fcgr(seq: str, k: int = 6, normalize: str = "prob") -> np.ndarray:
    """Render a DNA sequence as a [2^k, 2^k] FCGR image.

    k=6 -> 64x64 (matches the 64x64 conv encoders); larger k = finer k-mers.
    normalize: "prob" (sum to 1), "log" (log1p of counts), or "none" (raw counts).
    """
    size = 1 << k  # 2^k
    img = np.zeros((size, size), dtype=np.float64)
    x, y = cgr_points(seq)
    if x.size >= k:
        # each point from index k-1 on encodes the trailing k-mer
        xi = np.minimum((x[k - 1:] * size).astype(np.int64), size - 1)
        yi = np.minimum((y[k - 1:] * size).astype(np.int64), size - 1)
        # row = y (top-down), col = x
        np.add.at(img, (size - 1 - yi, xi), 1.0)
    if normalize == "prob":
        s = img.sum()
        if s > 0:
            img /= s
    elif normalize == "log":
        img = np.log1p(img)
    elif normalize != "none":
        raise ValueError(f"unknown normalize={normalize!r}")
    return img.astype(np.float32)


def fcgr_batch(seqs, k: int = 6, normalize: str = "prob") -> np.ndarray:
    """Stack FCGR images for a list of sequences -> [N, 2^k, 2^k]."""
    return np.stack([fcgr(s, k=k, normalize=normalize) for s in seqs], axis=0)
