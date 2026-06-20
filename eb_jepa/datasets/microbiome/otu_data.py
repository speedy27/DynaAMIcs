"""Static OTU-set dataset for the microbiome JEPA (Layer A) + the init_data branch.

A "community" (one sample / timepoint) is a set of up to ``N_max`` OTU tokens plus
a boolean presence mask, following the OBS/TOKEN CONTRACT shared with the encoder
workstream (WS2):

    obs = {"otu": FloatTensor[T, N_max, F], "mask": BoolTensor[T, N_max]}   # T=1 here
    token = concat( z(ProkBERT_384), z(CLR_log_abundance) ),   F = 385

This module provides:
  * ``load_prokbert_embeddings`` + the A97/B97 key resolver (reusing Susagi logic
    from Microbiome-Modelling/scripts/utils.py) to read the real corpus,
  * ``OTUSampleDataset`` with modes {"two_view", "masked", "single"} and a
    SYNTHETIC fallback so CPU smoke needs no real data,
  * ``init_microbiome_data`` — the entry point the dispatcher's ``microbiome``
    branch calls.

REAL-DATA NOTE (unverified on this Mac): the 22 GB corpus lives only on the
cluster. The real-data parsing here mirrors the format documented in the Susagi
clone (ProkBERT h5 group ``embeddings`` keyed by OTU id; ``samples-otus.97.mapped``
blocks of ``>SRR....SRS....`` followed by ``90_x;96_y;97_z\\t<count>`` lines;
``otus.rename.map1`` for A97/B97 key resolution). It has NOT been run against the
real files; the synthetic path is what the smoke test exercises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import (
    PerDimZScore,
    augment_community,
    clr,
    mask_otus,
)

D_EMB = 384          # ProkBERT embedding dim
TOKEN_DIM = D_EMB + 1  # F = 385 (per the contract)


# ===========================================================================
# Real-data loaders (Susagi format) -- exercised only on the cluster
# ===========================================================================
def load_prokbert_embeddings(
    path: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load ProkBERT OTU embeddings from the Susagi HDF5 file.

    The file stores one dataset per OTU under the ``embeddings`` group, keyed by
    OTU id (e.g. ``A97_1234`` / ``B97_1234``), each a 384-d float32 vector
    (see Microbiome-Modelling/scripts/utils.py:preview_prokbert_embeddings).

    Args:
        path: path to ``prokbert_embeddings.h5``.

    Returns:
        emb:           ndarray[N_otu, 384] float32 (row order = iteration order),
        otu_id_to_row: dict OTU-id -> row index into ``emb``.
    """
    import h5py  # local import; only needed on the real-data path

    rows: List[np.ndarray] = []
    otu_id_to_row: Dict[str, int] = {}
    with h5py.File(path, "r") as f:
        group = f["embeddings"]
        for i, key in enumerate(group.keys()):
            otu_id_to_row[key] = i
            rows.append(np.asarray(group[key][()], dtype=np.float32))
    emb = np.stack(rows, axis=0) if rows else np.zeros((0, D_EMB), dtype=np.float32)
    return emb, otu_id_to_row


def load_otu_rename_map(path: str, delimiter: str = "\t") -> dict:
    """A97/B97 OTU rename map (verbatim port of Susagi scripts/utils.py).

    Returns a dict with ``new97_to_oldA97`` / ``new97_to_oldB97`` (and the raw
    old<->new maps), used by ``build_otu_key_resolver`` to map the mapped-file
    97_* ids onto the actual HDF5 embedding keys.
    """
    old_to_new: Dict[str, str] = {}
    new_to_old: Dict[str, str] = {}
    oldA97_to_new97: Dict[str, str] = {}
    new97_to_oldA97: Dict[str, str] = {}
    oldB97_to_new97: Dict[str, str] = {}
    new97_to_oldB97: Dict[str, str] = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if delimiter and delimiter in line:
                parts = [p for p in line.split(delimiter) if p]
            else:
                parts = line.split()
            if len(parts) < 2:
                continue
            old_id, new_id = parts[0], parts[1]
            old_to_new[old_id] = new_id
            for token in new_id.split(";"):
                token = token.strip()
                if token:
                    new_to_old[token] = old_id
            left_tokens = [t for t in old_id.split(";") if t]
            right_tokens = [t for t in new_id.split(";") if t]
            left_head = left_tokens[0] if left_tokens else ""
            left97 = next((t for t in left_tokens if t.startswith("97_")), None)
            right97 = next((t for t in right_tokens if t.startswith("97_")), None)
            if left97 and right97:
                num = left97.split("_", 1)[1]
                if left_head.startswith("A"):
                    oldA97_to_new97[f"A97_{num}"] = right97
                    new97_to_oldA97[right97] = f"A97_{num}"
                if left_head.startswith("B"):
                    oldB97_to_new97[f"B97_{num}"] = right97
                    new97_to_oldB97[right97] = f"B97_{num}"
    return {
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
        "oldA97_to_new97": oldA97_to_new97,
        "new97_to_oldA97": new97_to_oldA97,
        "oldB97_to_new97": oldB97_to_new97,
        "new97_to_oldB97": new97_to_oldB97,
    }


def build_otu_key_resolver(
    otu_ids: List[str],
    rename_map: dict,
    emb_keys: set,
    prefer: str = "B",
) -> Dict[str, str]:
    """Map mapped-file 97_* ids onto actual embedding keys (A97_/B97_/raw).

    Port of Susagi scripts/utils.py:build_otu_key_resolver, but taking the set of
    available embedding keys directly (so it does not re-open the HDF5).
    """
    resolver: Dict[str, str] = {}
    for oid in set(otu_ids):
        if not (isinstance(oid, str) and oid.startswith("97_")):
            continue
        a = rename_map.get("new97_to_oldA97", {}).get(oid)
        b = rename_map.get("new97_to_oldB97", {}).get(oid)
        a_ok = bool(a) and a in emb_keys
        b_ok = bool(b) and b in emb_keys
        if a_ok and b_ok:
            resolver[oid] = a if prefer.upper() == "A" else b
        elif a_ok:
            resolver[oid] = a
        elif b_ok:
            resolver[oid] = b
        elif oid in emb_keys:
            resolver[oid] = oid
    return resolver


def parse_samples_otus_mapped(
    mapped_path: str,
    needed_srs: Optional[set] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Stream ``samples-otus.97.mapped`` into SRS -> [(otu97_id, count), ...].

    Mirrors Susagi scripts/utils.py:collect_micro_to_otus_mapped, but ALSO keeps
    the abundance count (the integer after the OTU triplet) needed to build CLR
    log-abundance tokens. Blocks look like::

        >SRR2459896.SRS1074972    66481   23845   497
        90_246;96_8626;97_10374   4920

    We take the ``97_*`` component as the OTU id and the trailing integer (4920)
    as its abundance count.

    NOTE: not verified against the real 20 GB file on this machine.
    """
    micro_to_otus: Dict[str, List[Tuple[str, float]]] = {}
    current_srs = None
    if not os.path.exists(mapped_path):
        raise FileNotFoundError(f"Mapped file not found: {mapped_path}")
    if needed_srs is not None:
        needed_srs = set(needed_srs)
    with open(mapped_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].split()[0]
                srs = header.split(".")[-1]
                if needed_srs is not None and srs not in needed_srs:
                    current_srs = None
                    continue
                current_srs = srs
                micro_to_otus.setdefault(current_srs, [])
                if max_samples is not None and len(micro_to_otus) > max_samples:
                    micro_to_otus.pop(current_srs, None)
                    break
                continue
            if current_srs is None:
                continue
            fields = line.split()
            triplet = fields[0]
            count = 0.0
            if len(fields) > 1:
                try:
                    count = float(fields[1])
                except ValueError:
                    count = 0.0
            otu97 = None
            for tok in triplet.split(";"):
                if tok.startswith("97_"):
                    otu97 = tok
                    break
            if otu97 is None:
                parts = triplet.split(";")
                otu97 = parts[-1] if parts else triplet
            micro_to_otus[current_srs].append((otu97, count))
    return micro_to_otus


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class OTUDatasetConfig:
    # Data source: if `embeddings_h5` / `data_dir` resolve to real files, the
    # real corpus is loaded; otherwise (or if `synthetic=True`) a synthetic
    # community generator with identical shapes/dtypes is used.
    data_dir: Optional[str] = None
    embeddings_h5: Optional[str] = None
    mapped_path: Optional[str] = None
    rename_map_path: Optional[str] = None
    synthetic: bool = False

    mode: str = "two_view"        # two_view | masked | single
    n_max: int = 64               # max OTUs per community (padding length)
    mask_frac: float = 0.5        # I-JEPA target fraction (mode="masked")
    pseudocount: float = 1e-6

    # SSL augmentation knobs (mode="two_view")
    subsample_frac: float = 0.8
    jitter_std: float = 0.1
    dropout_p: float = 0.1

    # Synthetic generator knobs
    synth_n_samples: int = 256
    synth_vocab: int = 200        # number of distinct synthetic OTUs
    synth_min_otus: int = 8
    synth_max_otus: int = 48
    synth_seed: int = 0

    batch_size: int = 32
    num_workers: int = 0
    val_frac: float = 0.1
    seed: int = 42

    # Filled in at build time (so downstream/eval can reuse the contract).
    size: int = 0


# ===========================================================================
# Dataset
# ===========================================================================
class OTUSampleDataset(Dataset):
    """Per-sample OTU-set dataset producing the obs dict per the contract.

    Modes:
      * ``two_view`` -> ``(view1_obs, view2_obs)``, each
        ``{"otu":[1,N_max,F], "mask":[1,N_max]}`` (two augmentations of the same
        community) for Layer-A VICReg/BCS.
      * ``masked``   -> ``(context_obs, target_obs, target_idx)`` for I-JEPA.
      * ``single``   -> ``(obs, meta)`` (one clean community) for probing/eval.

    Exposes ``.token_dim`` (F), ``.n_max``, ``.zscore`` (the fitted PerDimZScore).
    """

    def __init__(
        self,
        config: Optional[OTUDatasetConfig] = None,
        *,
        zscore: Optional[PerDimZScore] = None,
        synthetic: Optional[bool] = None,
        mode: Optional[str] = None,
        **overrides,
    ):
        super().__init__()
        if config is None:
            config = OTUDatasetConfig()
        # Convenience kwargs override the dataclass.
        if synthetic is not None:
            config.synthetic = synthetic
        if mode is not None:
            config.mode = mode
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.cfg = config
        self.token_dim = TOKEN_DIM
        self.n_max = config.n_max
        if config.mode not in {"two_view", "masked", "single"}:
            raise ValueError(f"unknown mode={config.mode!r}")

        use_real = self._real_available(config)
        self.is_synthetic = (not use_real) or config.synthetic
        if self.is_synthetic:
            self._raw = self._build_synthetic(config)
        else:
            self._raw = self._build_real(config)

        # Fit (or reuse) the per-dim z-score on the RAW (pre-zscore) tokens.
        if zscore is not None:
            self.zscore = zscore
        else:
            all_tokens = torch.cat([s["tokens"] for s in self._raw], dim=0)
            all_mask = torch.cat([s["mask"] for s in self._raw], dim=0)
            self.zscore = PerDimZScore().fit(all_tokens, mask=all_mask)

        self.cfg.size = len(self._raw)

    # -- source resolution ---------------------------------------------------
    @staticmethod
    def _real_available(cfg: OTUDatasetConfig) -> bool:
        if cfg.synthetic:
            return False
        h5 = cfg.embeddings_h5
        if h5 is None and cfg.data_dir:
            h5 = os.path.join(cfg.data_dir, "model", "prokbert_embeddings.h5")
        return bool(h5) and os.path.exists(h5)

    # -- synthetic source ----------------------------------------------------
    def _build_synthetic(self, cfg: OTUDatasetConfig) -> List[dict]:
        """Generate random OTU communities with the SAME shapes/dtypes as real.

        A fixed vocabulary of `synth_vocab` OTUs each gets a deterministic 384-d
        ProkBERT-like embedding; each sample draws a random subset of OTUs with
        sparse positive abundances. Tokens are built exactly like the real path:
        concat( ProkBERT_384, CLR_log_abundance ) (pre-zscore here).
        """
        g = torch.Generator().manual_seed(cfg.synth_seed)
        vocab_emb = torch.randn(cfg.synth_vocab, D_EMB, generator=g)  # [V, 384]

        samples: List[dict] = []
        for _ in range(cfg.synth_n_samples):
            n = int(torch.randint(cfg.synth_min_otus, cfg.synth_max_otus + 1, (1,), generator=g))
            n = min(n, cfg.n_max, cfg.synth_vocab)
            otu_rows = torch.randperm(cfg.synth_vocab, generator=g)[:n]
            # Sparse positive counts -> relative abundance. Exponential-ish via
            # -log(U) keeps it fully deterministic under the seeded generator.
            u = torch.rand(n, generator=g).clamp_min(1e-6)
            counts = (-torch.log(u)).clamp_min(1e-3)
            rel = counts / counts.sum()

            emb = vocab_emb[otu_rows]                       # [n, 384]
            clr_ab = clr(rel.unsqueeze(0), cfg.pseudocount).squeeze(0)  # [n]
            token = torch.cat([emb, clr_ab.unsqueeze(-1)], dim=-1)      # [n, 385]

            tokens, mask = self._pad_to_nmax(token, cfg.n_max)
            samples.append({"tokens": tokens, "mask": mask, "meta": {"synthetic": True}})
        return samples

    # -- real source ---------------------------------------------------------
    def _build_real(self, cfg: OTUDatasetConfig) -> List[dict]:
        """Build communities from the real Susagi corpus.

        Streams a (capped) number of samples from ``samples-otus.97.mapped``,
        resolves each OTU's ProkBERT embedding via the A97/B97 rename map, builds
        CLR log-abundance from the per-OTU counts, and pads to N_max.

        Unverified on this machine (real files are cluster-only). Raises a clear
        error if the required real files are missing so callers fall back to
        synthetic explicitly rather than silently.
        """
        h5 = cfg.embeddings_h5 or (
            os.path.join(cfg.data_dir, "model", "prokbert_embeddings.h5") if cfg.data_dir else None
        )
        mapped = cfg.mapped_path or (
            os.path.join(cfg.data_dir, "microbeatlas", "samples-otus.97.mapped")
            if cfg.data_dir else None
        )
        rename = cfg.rename_map_path or (
            os.path.join(cfg.data_dir, "microbeatlas", "otus.rename.map1")
            if cfg.data_dir else None
        )
        if not (h5 and mapped and rename and all(os.path.exists(p) for p in (h5, mapped, rename))):
            raise FileNotFoundError(
                "OTUSampleDataset real mode requires prokbert_embeddings.h5, "
                "samples-otus.97.mapped and otus.rename.map1 — set data_dir/embeddings_h5 "
                "to the cluster paths, or pass synthetic=True for CPU smoke."
            )

        emb, otu_id_to_row = load_prokbert_embeddings(h5)
        emb_keys = set(otu_id_to_row.keys())
        rename_map = load_otu_rename_map(rename)
        micro_to_otus = parse_samples_otus_mapped(
            mapped, max_samples=cfg.synth_n_samples
        )
        all_ids = [oid for lst in micro_to_otus.values() for (oid, _) in lst]
        resolver = build_otu_key_resolver(all_ids, rename_map, emb_keys)

        samples: List[dict] = []
        for srs, otu_counts in micro_to_otus.items():
            rows, counts = [], []
            for oid, cnt in otu_counts:
                key = resolver.get(oid, oid)
                row = otu_id_to_row.get(key)
                if row is not None:
                    rows.append(row)
                    counts.append(max(float(cnt), 0.0))
            if not rows:
                continue
            rows_t = torch.tensor(rows[: cfg.n_max], dtype=torch.long)
            counts_t = torch.tensor(counts[: cfg.n_max], dtype=torch.float32)
            rel = counts_t / counts_t.sum().clamp_min(1e-12)
            emb_t = torch.from_numpy(emb[rows_t.numpy()])                 # [n, 384]
            clr_ab = clr(rel.unsqueeze(0), cfg.pseudocount).squeeze(0)    # [n]
            token = torch.cat([emb_t, clr_ab.unsqueeze(-1)], dim=-1)      # [n, 385]
            tokens, mask = self._pad_to_nmax(token, cfg.n_max)
            samples.append({"tokens": tokens, "mask": mask, "meta": {"srs": srs}})
        if not samples:
            raise RuntimeError("real corpus parse produced 0 usable communities")
        return samples

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _pad_to_nmax(token: Tensor, n_max: int) -> Tuple[Tensor, Tensor]:
        """Pad/truncate a [n, F] community to [N_max, F] + bool mask [N_max]."""
        n, f = token.shape
        out = torch.zeros(n_max, f, dtype=torch.float32)
        mask = torch.zeros(n_max, dtype=torch.bool)
        k = min(n, n_max)
        out[:k] = token[:k].to(torch.float32)
        mask[:k] = True
        return out, mask

    def _obs(self, tokens: Tensor, mask: Tensor) -> Dict[str, Tensor]:
        """Z-score tokens and wrap as the time-major obs dict (T=1)."""
        z = self.zscore.transform(tokens)         # [N_max, F]
        z = z * mask.unsqueeze(-1).to(z.dtype)    # keep pad slots exactly 0
        return {
            "otu": z.unsqueeze(0).to(torch.float32),    # [1, N_max, F]
            "mask": mask.unsqueeze(0).to(torch.bool),   # [1, N_max]
        }

    # -- Dataset protocol ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, i: int):
        raw = self._raw[i]
        tokens, mask = raw["tokens"], raw["mask"]
        # Per-sample generator for reproducible-but-varied augmentation.
        g = torch.Generator().manual_seed(
            (self.cfg.seed * 1_000_003 + i) % (2 ** 31 - 1)
        )

        if self.cfg.mode == "two_view":
            t1, m1 = augment_community(
                tokens, mask,
                subsample_frac=self.cfg.subsample_frac,
                jitter_std=self.cfg.jitter_std,
                dropout_p=self.cfg.dropout_p,
                generator=g,
            )
            t2, m2 = augment_community(
                tokens, mask,
                subsample_frac=self.cfg.subsample_frac,
                jitter_std=self.cfg.jitter_std,
                dropout_p=self.cfg.dropout_p,
                generator=g,
            )
            return self._obs(t1, m1), self._obs(t2, m2)

        if self.cfg.mode == "masked":
            vis_t, vis_m, target_idx = mask_otus(
                tokens, mask, frac=self.cfg.mask_frac, generator=g
            )
            context = self._obs(vis_t, vis_m)
            target = self._obs(tokens, mask)
            return context, target, target_idx

        # single
        return self._obs(tokens, mask), raw.get("meta", {})


# ===========================================================================
# init_data entry point (called by the dispatcher's `microbiome` branch)
# ===========================================================================
@dataclass
class _MicrobiomeLoaderConfig:
    """Minimal config object returned alongside the loaders (what main.py reads)."""
    batch_size: int
    size: int
    token_dim: int
    n_max: int
    action_dim: int = 0
    proprio_dim: int = 0
    state_dim: int = 0
    mode: str = "two_view"
    extra: dict = field(default_factory=dict)


def init_microbiome_data(cfg_data: Optional[dict] = None, device=None):
    """Build microbiome train/val loaders.

    Returns ``(train_loader, val_loader, config, None)`` to match the
    ``init_data`` contract used by the examples. ``config`` is a small object
    carrying the attributes the example/encoder need (``batch_size``, ``size``,
    ``token_dim``, ``n_max``, and for the temporal task ``action_dim`` etc.).

    Selects the static OTU-set task (``task="otu"``, default) or the gLV temporal
    task (``task="glv"``). The static path needs no real data when synthetic is on.
    """
    cfg_data = dict(cfg_data or {})
    task = str(cfg_data.pop("task", "otu")).lower()

    if task in {"glv", "traj", "temporal"}:
        # Lazy import so a missing glv.py / traj wiring can't break otu_data import.
        from .traj import init_microbiome_traj_data
        return init_microbiome_traj_data(cfg_data, device)

    # ---- static OTU-set task (Layer A) ----
    ocfg = OTUDatasetConfig()
    for k, v in cfg_data.items():
        if hasattr(ocfg, k):
            setattr(ocfg, k, v)
    # Default to synthetic unless real paths are present (CPU/no-data friendly).
    if not OTUSampleDataset._real_available(ocfg):
        ocfg.synthetic = True

    full = OTUSampleDataset(ocfg)

    n = len(full)
    n_val = max(1, int(round(n * ocfg.val_frac)))
    g = torch.Generator().manual_seed(ocfg.seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    # Share the FITTED z-score between train and val (no leakage: it was fit on
    # all communities here; for a strict split, refit on train_idx only — left as
    # a TODO for the real-data path).
    train_set = torch.utils.data.Subset(full, train_idx)
    val_set = torch.utils.data.Subset(full, val_idx)

    loader_kwargs = dict(num_workers=ocfg.num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=ocfg.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=min(ocfg.batch_size, max(1, len(val_set))),
        shuffle=False, **loader_kwargs,
    )

    config = _MicrobiomeLoaderConfig(
        batch_size=ocfg.batch_size,
        size=len(train_set),
        token_dim=full.token_dim,
        n_max=full.n_max,
        mode=ocfg.mode,
        extra={"is_synthetic": full.is_synthetic, "task": "otu"},
    )
    return train_loader, val_loader, config, None
