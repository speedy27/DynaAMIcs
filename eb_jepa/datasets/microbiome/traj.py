"""Temporal (action-conditioned) microbiome trajectories for the world model (Layer B).

``GLVTrajDataset`` wraps WS5's generalized-Lotka-Volterra simulator
(``GLVSimulator.generate_trajectories``) and exposes each trajectory in the
EB-JEPA ``TrajDataset`` 5-tuple convention so the existing temporal machinery
(``TrajSlicerDataset`` -> RNN predictor -> planning) works unchanged:

    __getitem__(i) -> (obs_dict, act[T, K], state[T, S], reward[T], extra)

with ``obs`` matching the OBS/TOKEN CONTRACT (time-major, sliceable along T):

    obs = {"otu": FloatTensor[T, N_max, F], "mask": BoolTensor[T, N_max]},  F = 385

For synthetic gLV species there is no DNA, so each of the ``S`` species gets a
FIXED seeded 384-d "species embedding" (the analog of a ProkBERT vector); the
per-timepoint token is ``concat( species_embedding[s], z(log_abundance_s) )`` and
the mask marks species present (abundance > eps). This lets the SAME
set-transformer encoder consume gLV communities.

The simulator is imported LAZILY (``from .glv import ...`` inside ``__init__``) so
a missing ``glv.py`` cannot break importing this module; if it is absent, a tiny
inline stub produces arrays of the identical shapes for smoke testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..traj_dset import TrajDataset
from .transforms import PerDimZScore

D_EMB = 384
TOKEN_DIM = D_EMB + 1  # F = 385


# ---------------------------------------------------------------------------
# Inline stub used ONLY if eb_jepa/datasets/microbiome/glv.py is absent.
# Mirrors the WS5 API shape contract (see tasks/05-glv-simulator.md) so the
# trajectory dataset + smoke test run before WS5 lands. NOT a real simulator.
# ---------------------------------------------------------------------------
@dataclass
class _StubGLVConfig:
    n_species: int = 16
    n_candidate: int = 4
    dt: float = 0.05
    steps_per_action: int = 1
    noise_std: float = 0.0
    seed: int = 0


class _StubGLVSimulator:
    """Shape-faithful placeholder; produces smooth random non-negative states."""

    def __init__(self, config: _StubGLVConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    @property
    def n_species(self) -> int:
        return self.config.n_species

    @property
    def action_dim(self) -> int:
        return self.config.n_candidate

    @property
    def candidate_index(self) -> np.ndarray:
        return np.arange(self.config.n_candidate)

    @property
    def attractors(self) -> np.ndarray:
        return np.abs(self._rng.standard_normal((2, self.config.n_species))).astype(np.float32)

    def generate_trajectories(
        self, n: int, T: int, action_policy: str = "random", seed: int = 0
    ) -> dict:
        rng = np.random.default_rng(seed)
        S, K = self.config.n_species, self.config.n_candidate
        actions = rng.standard_normal((n, T, K)).astype(np.float32) * 0.1
        # Smooth-ish non-negative states via a cumulative random walk then relu.
        steps = rng.standard_normal((n, T + 1, S)).astype(np.float32) * 0.1
        states = np.cumsum(steps, axis=1)
        states = np.maximum(states - states.min() + 0.01, 0.0).astype(np.float32)
        return {"states": states, "actions": actions}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GLVTrajConfig:
    n_traj: int = 64          # number of trajectories
    T: int = 32               # env steps per trajectory (=> T+1 states)
    n_species: int = 16       # S
    n_candidate: int = 4      # K (action panel size == action_dim)
    dt: float = 0.05
    steps_per_action: int = 1
    noise_std: float = 0.0
    action_policy: str = "random"
    eps_present: float = 1e-4  # abundance below this -> species absent (mask=False)
    pseudocount: float = 1e-6
    emb_seed: int = 0          # seed for the fixed species embeddings
    sim_seed: int = 0          # seed for trajectory generation

    # ---- gLV STRUCTURAL knobs (plumbed for the generalization experiment; default == GLVConfig
    # defaults, so any existing config is bit-identical). These (NOT the seed) define the interaction
    # matrix A + the attractors: the simulator builds A deterministically from them. Varying them
    # produces genuinely different gLV INSTANCES (different A, different target attractors). ----
    n_guilds: int = 3
    self_lim: float = -1.0
    within_frac: float = 0.4
    comp_strong: float = -2.5
    comp_weak: float = -0.4
    growth: float = 1.0
    immigration: float = 1e-3

    batch_size: int = 16
    num_workers: int = 0
    val_frac: float = 0.1
    seed: int = 42
    size: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GLVTrajDataset(TrajDataset):
    """gLV trajectories as EB-JEPA (obs, act, state, reward, extra) sequences.

    N_max equals the number of gLV species S (every species is a token slot; the
    mask gates which are present at each timepoint). Tokens are z-scored with a
    PerDimZScore fit across all timepoints of all trajectories.

    Attributes:
        proprio_dim: 0 (no separate proprioception channel here).
        action_dim:  K (== n_candidate).
        state_dim:   S (== n_species); the raw gLV abundance state, for probes.
        token_dim:   F (== 385).
        n_max:       S.
    """

    def __init__(self, config: Optional[GLVTrajConfig] = None, *, zscore: Optional[PerDimZScore] = None, **overrides):
        super().__init__()
        if config is None:
            config = GLVTrajConfig()
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.cfg = config

        sim, used_stub = self._make_simulator(config)
        self.used_stub = used_stub
        self.action_dim = int(sim.action_dim)
        self.state_dim = int(sim.n_species)
        self.proprio_dim = 0
        self.n_max = int(sim.n_species)
        self.token_dim = TOKEN_DIM

        data = sim.generate_trajectories(
            n=config.n_traj, T=config.T, action_policy=config.action_policy, seed=config.sim_seed
        )
        states = np.asarray(data["states"], dtype=np.float32)   # [n, T+1, S]
        actions = np.asarray(data["actions"], dtype=np.float32)  # [n, T, K]
        # Align state length to action length (T) so obs/act share the time axis.
        states = states[:, : config.T, :]
        self._states = torch.from_numpy(states)                  # [n, T, S]
        self._actions = torch.from_numpy(actions)                # [n, T, K]

        # Fixed seeded species embeddings (analog of ProkBERT vectors).
        g = torch.Generator().manual_seed(config.emb_seed)
        self._species_emb = torch.randn(self.n_max, D_EMB, generator=g)  # [S, 384]

        # Build raw (pre-zscore) tokens for every trajectory/timepoint, then fit
        # the per-dim z-score across all real tokens.
        self._raw_tokens, self._raw_masks = self._build_tokens(self._states, config)
        if zscore is not None:
            self.zscore = zscore
        else:
            flat_tok = self._raw_tokens.reshape(-1, TOKEN_DIM)
            flat_mask = self._raw_masks.reshape(-1)
            self.zscore = PerDimZScore().fit(flat_tok, mask=flat_mask)

        self.cfg.size = self._states.shape[0]

    # -- simulator construction ---------------------------------------------
    @staticmethod
    def _make_simulator(cfg: GLVTrajConfig):
        """Lazily import the real gLV simulator; fall back to the inline stub."""
        try:
            from .glv import GLVSimulator, GLVConfig  # noqa: WPS433 (lazy import by design)
            gcfg = GLVConfig(
                n_species=cfg.n_species,
                n_candidate=cfg.n_candidate,
                dt=cfg.dt,
                steps_per_action=cfg.steps_per_action,
                noise_std=cfg.noise_std,
                seed=cfg.sim_seed,
                n_guilds=cfg.n_guilds,
                self_lim=cfg.self_lim,
                within_frac=cfg.within_frac,
                comp_strong=cfg.comp_strong,
                comp_weak=cfg.comp_weak,
                growth=cfg.growth,
                immigration=cfg.immigration,
            )
            return GLVSimulator(gcfg), False
        except Exception:
            scfg = _StubGLVConfig(
                n_species=cfg.n_species,
                n_candidate=cfg.n_candidate,
                dt=cfg.dt,
                steps_per_action=cfg.steps_per_action,
                noise_std=cfg.noise_std,
                seed=cfg.sim_seed,
            )
            return _StubGLVSimulator(scfg), True

    # -- token construction --------------------------------------------------
    def _build_tokens(self, states: Tensor, cfg: GLVTrajConfig) -> Tuple[Tensor, Tensor]:
        """[n, T, S] abundances -> tokens [n, T, S, F] + mask [n, T, S].

        Per timepoint: relative-abundance -> CLR over present species (others get
        the abundance feature from CLR too, but are masked out), then token =
        concat(species_emb, clr_log_abundance). The species embedding is the same
        for a given species across all timepoints (it identifies the taxon).
        """
        from .transforms import clr

        n, T, S = states.shape
        # Relative abundance per timepoint.
        denom = states.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rel = states / denom                                   # [n, T, S]
        clr_ab = clr(rel, cfg.pseudocount, dim=-1)             # [n, T, S]
        mask = states > cfg.eps_present                        # [n, T, S] bool

        emb = self._species_emb.view(1, 1, S, D_EMB).expand(n, T, S, D_EMB)
        tokens = torch.cat([emb, clr_ab.unsqueeze(-1)], dim=-1)  # [n, T, S, 385]
        # Zero out absent species' features (they are masked anyway).
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens.contiguous(), mask.contiguous()

    # -- TrajDataset protocol ------------------------------------------------
    def get_seq_length(self, idx: int) -> int:
        return int(self._states.shape[1])  # T (same for all)

    def __len__(self) -> int:
        return int(self._states.shape[0])

    def __getitem__(self, i: int):
        raw_tok = self._raw_tokens[i]      # [T, S, F]
        mask = self._raw_masks[i]          # [T, S]
        z = self.zscore.transform(raw_tok)
        z = z * mask.unsqueeze(-1).to(z.dtype)  # keep absent slots exactly 0

        obs = {
            "otu": z.to(torch.float32),         # [T, N_max(=S), F]
            "mask": mask.to(torch.bool),        # [T, N_max(=S)]
        }
        act = self._actions[i].to(torch.float32)     # [T, K]
        state = self._states[i].to(torch.float32)    # [T, S]
        reward = torch.zeros(act.shape[0], dtype=torch.float32)  # [T]
        extra: dict = {"traj_idx": i}
        return obs, act, state, reward, extra


# ---------------------------------------------------------------------------
# init entry point for the temporal task (called from init_microbiome_data)
# ---------------------------------------------------------------------------
@dataclass
class _GLVLoaderConfig:
    batch_size: int
    size: int
    token_dim: int
    n_max: int
    action_dim: int
    proprio_dim: int
    state_dim: int
    num_frames: int
    extra: dict = field(default_factory=dict)


def init_microbiome_traj_data(cfg_data: Optional[dict] = None, device=None):
    """Build gLV trajectory train/val loaders sliced into fixed-length windows.

    Returns ``(train_loader, val_loader, config, None)``. Slicing uses
    ``TrajSlicerDataset`` (the same component as two_rooms), so each batch is a
    window of ``num_frames`` consecutive timepoints with collated obs dicts.
    """
    from ..traj_dset import (
        TrajSlicerDataset,  # noqa: F401  (kept for explicitness / parity with two_rooms)
        get_train_val_sliced,
    )

    cfg_data = dict(cfg_data or {})
    num_frames = int(cfg_data.pop("num_frames", 4))
    frameskip = int(cfg_data.pop("frameskip", 1))
    train_fraction = float(cfg_data.pop("train_fraction", 1.0 - 0.1))

    gcfg = GLVTrajConfig()
    for k, v in cfg_data.items():
        if hasattr(gcfg, k):
            setattr(gcfg, k, v)

    base = GLVTrajDataset(gcfg)

    _, _, train_slices, val_slices = get_train_val_sliced(
        base,
        train_fraction=train_fraction,
        random_seed=gcfg.seed,
        num_frames=num_frames,
        frameskip=frameskip,
    )

    loader_kwargs = dict(num_workers=gcfg.num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_slices, batch_size=gcfg.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_slices, batch_size=min(gcfg.batch_size, max(1, len(val_slices))),
        shuffle=False, **loader_kwargs,
    )

    config = _GLVLoaderConfig(
        batch_size=gcfg.batch_size,
        size=len(train_slices),
        token_dim=base.token_dim,
        n_max=base.n_max,
        action_dim=train_slices.action_dim,
        proprio_dim=train_slices.proprio_dim,
        state_dim=train_slices.state_dim,
        num_frames=num_frames,
        extra={"used_stub_glv": base.used_stub, "task": "glv"},
    )
    return train_loader, val_loader, config, None
