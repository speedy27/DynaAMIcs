"""WS3 — Intervention planning on the gLV simulator (the Layer-B *application* result).

Given a TRAINED gLV world model (SetTransformerEncoder f_theta + GRU RNNPredictor g_phi from
``train_worldmodel.run``), plan a sequence of continuous K-dim interventions (delta-abundance on the
candidate panel) that drive the community from a start attractor to a TARGET attractor, by minimizing
distance to the target *representation* in the model's LATENT space. We then compare to baselines and
report success rate (final true gLV state within tolerance of the target attractor).

WHY A LEAN LATENT-SPACE MPPI (and NOT GCAgent):
    ``eb_jepa/planning.py`` MPPIPlanner/GCAgent assume TENSOR observations (``GCAgent.unroll`` calls
    ``obs_init.repeat(...)``). Our observation is a DICT ({"otu", "mask"}), so that path breaks. We
    therefore implement a small MPPI directly in latent space: encode the start once, then roll the
    GRU predictor forward purely in latent space (no re-encode inside the optimizer), score candidate
    action sequences by latent distance to the target representation, and update the sampling mean by
    exp-weighted elites — mirroring the MPPI math in planning.py:1299-1338, but in latent space.

THE WORLD-MODEL CONTRACTS WE RELY ON (all verified against the code, not guessed):
    * encoder: ``SetTransformerEncoder(obs_dict) -> [B, D, T, 1, 1]``      (architectures.py:601)
    * predictor (RNNPredictor): ``forward(state[B,D,1,1,1], action[B,K,1]) -> [B,D,1,1,1]``; the GRU
      uses ``state`` as its hidden ([1,B,D]) and ``action`` as its input ([1,B,K]).  (architectures.py:436)
      => rolling forward H steps from z0 in latent space is just H calls to ``predictor``.
    * a single gLV state ``x[S]`` is encoded into the obs dict by REUSING the EXACT token construction
      of ``GLVTrajDataset`` (``_build_tokens`` + the fixed seeded ``_species_emb`` + the fitted
      ``zscore``), so the encoder sees tokens built identically to training. (traj.py:205, 235)
    * checkpoint: ``train_worldmodel.run`` saves ``save_checkpoint(model=jepa,
      encoder_state_dict=..., idm_coeff=...)`` => ``model_state_dict`` is the FULL JEPA (trained
      encoder + trained GRU predictor). We rebuild the JEPA exactly as ``train_worldmodel`` builds it,
      then load ``model_state_dict`` so we plan with the trained predictor too. (training_utils.py:146)

INTEGRITY: success rates printed/saved here come ONLY from the actual MPC rollouts executed in this
process (the planned action is applied to the real ``GLVSimulator``; success = the TRUE final state is
within tolerance of the target attractor). The smoke (``_smoke_plan.py``) uses a RANDOM, untrained
model with no checkpoint — it proves the HARNESS runs end to end; a random model is NOT expected to
succeed, and we say so.

Fire entry (with a trained checkpoint):
    .venv-cpu/bin/python -m examples.microbiome_jepa.plan_glv \
        --checkpoint <exp_dir>/latest.pth.tar \
        --fname examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml \
        --n_episodes 20 --seeds 0,1,2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from eb_jepa.architectures import (
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    SetTransformerEncoder,
)
from eb_jepa.datasets.microbiome.glv import GLVConfig, GLVSimulator
from eb_jepa.datasets.microbiome.traj import GLVTrajConfig, GLVTrajDataset
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

logger = get_logger(__name__)


# ==================================================================================================
# gLV state  ->  encoder obs dict   (REUSES GLVTrajDataset token construction; see traj.py:205,235)
# ==================================================================================================
class StateEncoder:
    """Turn a single raw gLV abundance state ``x[S]`` into the encoder's obs dict and encode it.

    The token construction MUST match training exactly, or the trained encoder sees out-of-distribution
    inputs. We therefore reuse a real ``GLVTrajDataset`` (built with the SAME gLV params + ``emb_seed``
    + ``sim_seed`` as the training data config) and call its OWN ``_build_tokens`` (CLR + species
    embedding) and its OWN fitted ``zscore`` — i.e. the identical pipeline as ``GLVTrajDataset.__getitem__``
    (traj.py:235-244). ``_build_tokens`` and ``zscore`` are deterministic given the config, so this
    reconstructs the training-time normalization as faithfully as possible without the original loader
    object. (See "ASSUMPTIONS" in the module-level summary returned by ``run``.)
    """

    def __init__(self, glv_dataset: GLVTrajDataset, device):
        self.ds = glv_dataset
        self.device = device
        self.S = int(glv_dataset.n_max)

    @torch.no_grad()
    def obs(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        """``x[S]`` (numpy abundance) -> obs ``{"otu":[1,1,S,F], "mask":[1,1,S]}`` on device."""
        x = np.asarray(x, dtype=np.float32).reshape(1, 1, self.S)  # [n=1, T=1, S]
        states = torch.from_numpy(x)
        raw_tok, mask = self.ds._build_tokens(states, self.ds.cfg)  # [1,1,S,F], [1,1,S]
        z = self.ds.zscore.transform(raw_tok)                       # per-dim z-score (training stats)
        z = z * mask.unsqueeze(-1).to(z.dtype)                      # keep absent slots exactly 0
        return {
            "otu": z.to(torch.float32).to(self.device),   # [1,1,S,F]
            "mask": mask.to(torch.bool).to(self.device),  # [1,1,S]
        }

    @torch.no_grad()
    def encode(self, jepa: JEPA, x: np.ndarray) -> torch.Tensor:
        """``x[S]`` -> latent state ``z [1, D, 1, 1, 1]`` (the RNNPredictor's state convention)."""
        state = jepa.encode(self.obs(x))   # [1, D, T=1, 1, 1]
        return state[:, :, 0:1]            # [1, D, 1, 1, 1]


# ==================================================================================================
# Latent rollout with the GRU predictor (no re-encode) + the MPPI core
# ==================================================================================================
@torch.no_grad()
def rollout_latent(
    predictor: RNNPredictor, z0: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Roll the latent forward H steps for a BATCH of action sequences, purely in latent space.

    Args:
        predictor: the trained RNNPredictor (GRU). forward(state[B,D,1,1,1], action[B,K,1]) -> [B,D,1,1,1].
        z0:        [1, D, 1, 1, 1] start latent (shared across all N candidate sequences).
        actions:   [N, K, H] candidate action sequences (N sequences, K action dims, H horizon).
    Returns:
        zs: [N, H, D] the latent state AFTER each of the H steps (zs[:, t] = z_{t+1}).
    """
    N, K, H = actions.shape
    D = z0.shape[1]
    z = z0.expand(N, D, 1, 1, 1).contiguous()  # broadcast start latent to all N sequences
    outs = []
    for t in range(H):
        a_t = actions[:, :, t].unsqueeze(-1)   # [N, K, 1]  (RNNPredictor action convention)
        z = predictor(z, a_t)                  # [N, D, 1, 1, 1]
        outs.append(z.flatten(1))              # [N, D]
    return torch.stack(outs, dim=1)            # [N, H, D]


@dataclass
class MPPIConfig:
    horizon: int = 8          # planning horizon H (steps looked ahead)
    n_samples: int = 256      # candidate action sequences per iteration
    n_elites: int = 32        # elites used to refit the sampling distribution
    n_iters: int = 4          # MPPI refinement iterations
    temperature: float = 1.0  # exp-weighting temperature (higher => sharper toward best elite)
    init_std: float = 0.25    # initial per-dim action std for sampling
    min_std: float = 0.02     # floor on the sampling std (avoid premature collapse)
    cumulative: bool = True   # True => cost = sum over steps; False => final-step cost only (ablation)


@torch.no_grad()
def mppi_plan(
    predictor: RNNPredictor,
    z0: torch.Tensor,
    z_tgt: torch.Tensor,
    action_dim: int,
    action_max: float,
    cfg: MPPIConfig,
    mean_init: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One MPPI plan in LATENT space. Mirrors planning.py:1299-1338 but cost = latent distance to z_tgt.

    Args:
        predictor: trained GRU predictor (rolls the latent forward).
        z0:        [1, D, 1, 1, 1] start latent.
        z_tgt:     [1, D]          target latent (flattened community vector of the target attractor).
        action_dim K, action_max: action box is [-action_max, action_max]^K (gLV clips to this anyway).
        cfg:       MPPIConfig.
        mean_init: [H, K] optional warm-start mean (e.g. previous plan shifted); else zeros.
        generator: optional torch.Generator for reproducible sampling.
    Returns:
        (best_first_action [K], full_mean_plan [H, K]).  best_first_action is what MPC executes.
    """
    device = z0.device
    H, K = cfg.horizon, action_dim
    z_tgt = z_tgt.reshape(1, 1, -1)  # [1, 1, D] for broadcast over [N, H, D]

    mean = torch.zeros(H, K, device=device) if mean_init is None else mean_init.clone().to(device)
    std = torch.full((H, K), cfg.init_std, device=device)

    score = None
    actions = None
    for _ in range(cfg.n_iters):
        # Sample N action sequences ~ N(mean, std), clamp to the env action box.  [N, H, K]
        noise = torch.randn(cfg.n_samples, H, K, device=device, generator=generator)
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        actions = actions.clamp(-action_max, action_max)

        # Roll latent forward and score: distance to target latent per step.  [N, H]
        a_thk = actions.permute(0, 2, 1).contiguous()      # [N, K, H] for rollout_latent
        zs = rollout_latent(predictor, z0, a_thk)          # [N, H, D]
        step_dist = torch.linalg.norm(zs - z_tgt, dim=-1)  # [N, H]  L2 latent distance each step
        if cfg.cumulative:
            cost = step_dist.sum(dim=1)                     # [N]  (CLAUDE.md: cumulative helps)
        else:
            cost = step_dist[:, -1]                         # [N]  final-state cost only (ablation)

        # Elites = lowest cost.  Exp-weight by (min_cost - cost) so the best elite gets weight ~1.
        n_el = min(cfg.n_elites, cfg.n_samples)
        elite_cost, elite_idx = torch.topk(-cost, n_el)    # -cost so topk picks the cheapest
        elite_cost = -elite_cost                            # [n_el] actual costs (ascending)
        elite_actions = actions[elite_idx]                  # [n_el, H, K]

        min_cost = cost.min()
        w = torch.exp(cfg.temperature * (min_cost - elite_cost))  # [n_el], best elite -> 1
        w = w / (w.sum() + 1e-9)
        mean = (w.view(n_el, 1, 1) * elite_actions).sum(dim=0)    # [H, K]
        var = (w.view(n_el, 1, 1) * (elite_actions - mean.unsqueeze(0)) ** 2).sum(dim=0)
        std = var.sqrt().clamp_min(cfg.min_std)                   # [H, K]
        score = w

    # Return the MEAN plan's first action (eval-mode: deterministic best estimate).
    best_first = mean[0].clamp(-action_max, action_max)
    return best_first, mean


# ==================================================================================================
# Baselines (each returns the action [K] to apply at the current state)
# ==================================================================================================
def _greedy_action(
    sim: GLVSimulator, x: np.ndarray, target_state: np.ndarray, action_max: float,
    n_levels: int = 2,
) -> np.ndarray:
    """1-step ORACLE-ish greedy in TRUE state space: the single-candidate dose (or no-op) that most
    reduces Euclidean distance to the target attractor *after one env step*. This is the baseline the
    gLV's NON-MONOTONICITY is designed to defeat (it gets stuck needing a temporary move-away).
    Uses ``sim._step_from`` (a pure functional step) so it does NOT mutate the live env state."""
    K = sim.action_dim
    best_a = np.zeros(K, dtype=np.float32)
    best_d = float(np.linalg.norm(sim._step_from(x, best_a) - target_state))  # no-op baseline
    levels = [action_max * (i + 1) / n_levels for i in range(n_levels)]
    for k in range(K):
        for amt in levels:
            a = np.zeros(K, dtype=np.float32)
            a[k] = amt
            d = float(np.linalg.norm(sim._step_from(x, a) - target_state))
            if d < best_d - 1e-12:
                best_d, best_a = d, a
    return best_a


# ==================================================================================================
# One MPC episode for a given method
# ==================================================================================================
@dataclass
class EpisodeResult:
    method: str
    success: bool
    final_dist: float          # TRUE-state Euclidean distance to target attractor at the end
    start_dist: float          # ... at the start (for context)
    best_dist: float           # closest TRUE-state distance reached at any step
    tol: float
    src: int
    tgt: int
    n_steps: int


@torch.no_grad()
def run_episode(
    method: str,
    sim: GLVSimulator,
    jepa: JEPA,
    state_enc: StateEncoder,
    src: int,
    tgt: int,
    tol: float,
    mpc_steps: int,
    mppi_cfg: MPPIConfig,
    rng: np.random.Generator,
    torch_gen: Optional[torch.Generator] = None,
) -> EpisodeResult:
    """Run one MPC episode driving the community from attractor ``src`` toward attractor ``tgt``.

    The MPC loop (for ``mppi`` / ``final_only``): encode the CURRENT true state -> latent; plan with
    MPPI in latent space; EXECUTE the planned first action in the real ``GLVSimulator``; re-encode the
    new state; replan; until ``mpc_steps`` or success. ``random`` applies random doses; ``greedy``
    applies the 1-step state-space oracle action. Success = TRUE final state within ``tol`` of the
    target attractor (relative L2 in abundance space; see ``run`` for how ``tol`` is set)."""
    action_max = float(sim.config.action_max)
    K = int(sim.action_dim)
    target_state = sim.attractors[tgt]

    x = sim.reset(attractor=src).astype(np.float32)
    start_dist = float(np.linalg.norm(x - target_state))
    best_dist = start_dist

    # Encode the (fixed) target community ONCE into a target latent.
    z_tgt = state_enc.encode(jepa, target_state).flatten(1)  # [1, D]

    warm_mean = None  # MPPI warm-start (shifted previous plan)
    for _ in range(mpc_steps):
        if method == "random":
            a = rng.uniform(0.0, action_max, size=K).astype(np.float32)
        elif method == "greedy":
            a = _greedy_action(sim, x, target_state, action_max)
        elif method in ("mppi", "final_only"):
            z0 = state_enc.encode(jepa, x)  # [1, D, 1, 1, 1]
            cfg = mppi_cfg
            if method == "final_only":
                cfg = MPPIConfig(**{**mppi_cfg.__dict__, "cumulative": False})
            a, mean_plan = mppi_plan(
                jepa.predictor, z0, z_tgt, K, action_max, cfg,
                mean_init=warm_mean, generator=torch_gen,
            )
            # Warm-start next step with the plan shifted one step (standard MPC receding horizon).
            warm_mean = torch.zeros_like(mean_plan)
            warm_mean[:-1] = mean_plan[1:]
            a = a.detach().cpu().numpy().astype(np.float32)
        else:
            raise ValueError(f"unknown method {method!r}")

        if not np.all(np.isfinite(a)):
            raise FloatingPointError(f"method {method!r} produced non-finite action: {a}")

        x = sim.step(a).astype(np.float32)
        d = float(np.linalg.norm(x - target_state))
        best_dist = min(best_dist, d)
        if d < tol:
            break

    final_dist = float(np.linalg.norm(x - target_state))
    return EpisodeResult(
        method=method, success=bool(final_dist < tol), final_dist=final_dist,
        start_dist=start_dist, best_dist=best_dist, tol=tol, src=src, tgt=tgt,
        n_steps=mpc_steps,
    )


# ==================================================================================================
# Build the trained world model from cfg + checkpoint (mirrors train_worldmodel.run exactly)
# ==================================================================================================
def build_world_model(
    fname: str,
    checkpoint: Optional[str],
    device,
    overrides: Optional[dict] = None,
):
    """Rebuild encoder + GRU predictor (+ idm/regularizer, for state-dict compatibility) EXACTLY as
    ``train_worldmodel.run`` does, then load ``model_state_dict`` (the full trained JEPA) from the
    checkpoint. Returns (jepa, cfg, K). If ``checkpoint`` is None -> random init (smoke only)."""
    from eb_jepa.training_utils import load_config

    cfg = load_config(fname, overrides or None, quiet=True)
    K = int(cfg.data.n_candidate)
    token_dim = int(cfg.model.token_dim)

    encoder = SetTransformerEncoder(
        token_dim=token_dim,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        pool=cfg.model.pool,
    )
    D = getattr(encoder, "mlp_output_dim", cfg.model.d_model)
    predictor = RNNPredictor(hidden_size=D, action_dim=K, final_ln=encoder.final_ln)
    aencoder = nn.Identity()

    # Rebuild the regularizer + IDM with the SAME structure as training so model_state_dict loads
    # cleanly (we do not use them for planning, but they are part of the saved JEPA state).
    rcfg = cfg.model.regularizer
    projector = Projector(f"{D}-{D * 4}-{D * 4}") if rcfg.get("use_proj", True) else None
    idm_state_dim = (
        projector.out_dim
        if (projector is not None and rcfg.get("idm_after_proj", False))
        else D
    )
    idm = InverseDynamicsModel(
        state_dim=idm_state_dim, hidden_dim=int(rcfg.get("idm_hidden", 256)), action_dim=K
    )
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=rcfg.cov_coeff,
        std_coeff=rcfg.std_coeff,
        sim_coeff_t=rcfg.sim_coeff_t,
        idm_coeff=float(rcfg.idm_coeff),
        idm=idm,
        first_t_only=rcfg.get("first_t_only", False),
        projector=projector,
        spatial_as_samples=rcfg.get("spatial_as_samples", False),
        idm_after_proj=rcfg.get("idm_after_proj", False),
        sim_t_after_proj=rcfg.get("sim_t_after_proj", False),
    )
    jepa = JEPA(encoder, aencoder, predictor, regularizer, SquareLossSeq()).to(device)

    if checkpoint is None:
        logger.warning("[plan] no --checkpoint -> RANDOM-INIT world model (sanity/harness only, "
                       "NOT a trained result; do not expect planning success).")
        jepa.eval()
        return jepa, cfg, K

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict")
    if sd is None:
        raise KeyError(f"checkpoint {checkpoint} has no 'model_state_dict' (expected from "
                       f"train_worldmodel.save_checkpoint).")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}  # de-compile if needed
    info = jepa.load_state_dict(sd, strict=False)
    if getattr(info, "missing_keys", None):
        logger.info(f"[plan] load_state_dict missing keys (first 5): {info.missing_keys[:5]}")
    if getattr(info, "unexpected_keys", None):
        logger.info(f"[plan] load_state_dict unexpected keys (first 5): {info.unexpected_keys[:5]}")
    logger.info(f"[plan] loaded trained world model (model_state_dict) from {checkpoint}")
    jepa.eval()
    return jepa, cfg, K


def build_glv_and_encoder(cfg, device) -> Tuple[GLVSimulator, StateEncoder]:
    """Build the gLV env + the StateEncoder, BOTH using the SAME params as the training data config so
    the env dynamics, the species embeddings, and the fitted z-score all match training."""
    d = cfg.data
    # STRUCTURAL knobs (n_guilds / comp strengths / growth / ...) define the interaction matrix A and
    # the attractors; default to GLVConfig defaults so a config that omits them is identical to before.
    _gd = GLVConfig()
    glv_cfg = GLVConfig(
        n_species=int(d.n_species),
        n_candidate=int(d.n_candidate),
        dt=float(d.dt),
        steps_per_action=int(d.get("steps_per_action", 1)),
        noise_std=float(d.get("noise_std", 0.0)),
        seed=int(d.get("sim_seed", d.get("seed", 0))),
        n_guilds=int(d.get("n_guilds", _gd.n_guilds)),
        self_lim=float(d.get("self_lim", _gd.self_lim)),
        within_frac=float(d.get("within_frac", _gd.within_frac)),
        comp_strong=float(d.get("comp_strong", _gd.comp_strong)),
        comp_weak=float(d.get("comp_weak", _gd.comp_weak)),
        growth=float(d.get("growth", _gd.growth)),
        immigration=float(d.get("immigration", _gd.immigration)),
    )
    sim = GLVSimulator(glv_cfg)

    # A GLVTrajDataset built with the SAME data config -> identical _species_emb + fitted zscore as the
    # training loader (init_microbiome_traj_data builds GLVTrajDataset from the same cfg.data).
    traj_cfg = GLVTrajConfig()
    for k in ("n_traj", "T", "n_species", "n_candidate", "dt", "steps_per_action", "noise_std",
              "action_policy", "emb_seed", "sim_seed", "eps_present", "pseudocount",
              "n_guilds", "self_lim", "within_frac", "comp_strong", "comp_weak", "growth", "immigration"):
        if hasattr(d, k) and d.get(k) is not None and hasattr(traj_cfg, k):
            setattr(traj_cfg, k, d.get(k))
    glv_ds = GLVTrajDataset(traj_cfg)
    state_enc = StateEncoder(glv_ds, device)
    return sim, state_enc


# ==================================================================================================
# Fire entry: plan with all methods over many (src,tgt) episodes x seeds; JSON + figure
# ==================================================================================================
METHODS = ["random", "greedy", "final_only", "mppi"]


def _agg(values: List[float]) -> Tuple[float, float]:
    """mean and standard error over a list of per-seed success rates."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    checkpoint: Optional[str] = None,
    seeds: str = "0,1,2",
    n_episodes: int = 20,
    mpc_steps: int = 40,
    horizon: int = 8,
    n_samples: int = 256,
    n_elites: int = 32,
    n_iters: int = 4,
    temperature: float = 1.0,
    init_std: float = 0.25,
    tol_frac: float = 0.15,
    methods: Optional[str] = None,
    device: str = "cpu",
    out: str = "checkpoints/microbiome_jepa/planning",
    overrides: Optional[dict] = None,
) -> dict:
    """Plan interventions on gLV with a trained world model; report per-method success rates.

    Protocol: for each seed, sample ``n_episodes`` (src!=tgt) attractor pairs and run a full MPC
    episode per method. Success = TRUE final gLV state within ``tol`` of the target attractor, where
    ``tol = tol_frac * mean inter-attractor distance`` (so the threshold is tied to the attractor
    separation scale, as the brief asks). Aggregate mean +/- s.e. of success rate across seeds; save
    JSON + a success-rate bar figure.

    INTEGRITY: every reported number is measured from the actual MPC rollouts in this run. With a
    random model (no --checkpoint) success will be ~0 and that only validates the harness.
    """
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    # fire turns "a,b,c" into a tuple ('a','b','c') and a single token into a str — handle both.
    if methods is None:
        method_list = METHODS
    elif isinstance(methods, (list, tuple)):
        method_list = [str(m).strip() for m in methods if str(m).strip()]
    else:
        method_list = [m.strip() for m in str(methods).split(",") if m.strip()]
    if isinstance(seeds, (list, tuple)):
        seed_list = [int(s) for s in seeds]
    elif isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip() != ""]

    jepa, cfg, K = build_world_model(fname, checkpoint, dev, overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    n_attr = int(sim.attractors.shape[0])
    if n_attr < 2:
        raise RuntimeError("need >= 2 attractors to define a planning task; check gLV config.")

    # Tolerance tied to inter-attractor distance scale (brief: "threshold tied to inter-attractor dist").
    attr = sim.attractors
    inter = [float(np.linalg.norm(attr[i] - attr[j]))
             for i in range(n_attr) for j in range(n_attr) if i != j]
    attr_scale = float(np.mean(inter)) if inter else 1.0
    tol = tol_frac * attr_scale

    mppi_cfg = MPPIConfig(
        horizon=horizon, n_samples=n_samples, n_elites=n_elites, n_iters=n_iters,
        temperature=temperature, init_std=init_std, cumulative=True,
    )

    logger.info(f"[plan] world-model: trained={checkpoint is not None} D={jepa.predictor.rnn.hidden_size} "
                f"K={K} | gLV: S={sim.n_species} n_attr={n_attr} action_max={sim.config.action_max} "
                f"stub_glv={state_enc.ds.used_stub}")
    logger.info(f"[plan] tol={tol:.4f} (={tol_frac}*attr_scale={attr_scale:.4f}); "
                f"methods={method_list}; episodes/seed={n_episodes}; mpc_steps={mpc_steps}")

    records: List[dict] = []  # one per (seed, episode, method)
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator(device=dev).manual_seed(seed)
        # Sample episode (src, tgt) pairs for this seed (src != tgt).
        pairs = []
        for _ in range(n_episodes):
            s = int(rng.integers(n_attr))
            t = int(rng.integers(n_attr - 1))
            if t >= s:
                t += 1
            pairs.append((s, t))

        for (src, tgt) in pairs:
            for m in method_list:
                res = run_episode(
                    m, sim, jepa, state_enc, src, tgt, tol, mpc_steps, mppi_cfg, rng,
                    torch_gen=torch_gen,
                )
                records.append({"seed": seed, **res.__dict__})

    # ---- aggregate: success rate per method per seed, then mean +/- s.e. across seeds ----
    summary: Dict[str, Dict[str, float]] = {}
    for m in method_list:
        per_seed_rate = []
        for seed in seed_list:
            ep = [r for r in records if r["method"] == m and r["seed"] == seed]
            per_seed_rate.append(np.mean([r["success"] for r in ep]) if ep else float("nan"))
        mean, se = _agg(per_seed_rate)
        all_ep = [r for r in records if r["method"] == m]
        summary[m] = {
            "success_rate_mean": mean,
            "success_rate_se": se,
            "per_seed_success_rate": [float(x) for x in per_seed_rate],
            "mean_final_dist": float(np.mean([r["final_dist"] for r in all_ep])) if all_ep else float("nan"),
            "mean_best_dist": float(np.mean([r["best_dist"] for r in all_ep])) if all_ep else float("nan"),
            "mean_start_dist": float(np.mean([r["start_dist"] for r in all_ep])) if all_ep else float("nan"),
            "n_episodes_total": len(all_ep),
        }

    # ---- print table ----
    print("\n================ gLV INTERVENTION PLANNING (success rate) ================")
    print(f"trained_model={checkpoint is not None} seeds={seed_list} episodes/seed={n_episodes} "
          f"mpc_steps={mpc_steps} horizon={horizon} tol={tol:.4f} stub_glv={state_enc.ds.used_stub}")
    print("method".ljust(14) + "success_rate".ljust(22) + "mean_final_dist".ljust(18) + "mean_best_dist")
    for m in method_list:
        s = summary[m]
        print(m.ljust(14)
              + f"{s['success_rate_mean']:.3f} ± {s['success_rate_se']:.3f}".ljust(22)
              + f"{s['mean_final_dist']:.3f}".ljust(18)
              + f"{s['mean_best_dist']:.3f}")
    if checkpoint is None:
        print("NOTE: random/untrained model -> success ~0 expected; this only validates the harness.")

    # ---- save JSON ----
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "planning_results.json"
    payload = {
        "fname": fname,
        "checkpoint": checkpoint,
        "trained_model": checkpoint is not None,
        "seeds": seed_list,
        "n_episodes": n_episodes,
        "mpc_steps": mpc_steps,
        "tol": tol,
        "tol_frac": tol_frac,
        "attractor_scale": attr_scale,
        "n_attractors": n_attr,
        "action_dim": K,
        "used_stub_glv": bool(state_enc.ds.used_stub),
        "mppi_cfg": mppi_cfg.__dict__,
        "summary": summary,
        "records": records,
    }
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nsaved results -> {res_path}")

    # ---- figure: success rate per method (mean +/- s.e.) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        means = [summary[m]["success_rate_mean"] for m in method_list]
        ses = [summary[m]["success_rate_se"] for m in method_list]
        colors = {"random": "#999", "greedy": "#c84", "final_only": "#48c", "mppi": "#2a7"}
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(range(len(method_list)), means, yerr=ses, capsize=4,
               color=[colors.get(m, "#777") for m in method_list])
        ax.set_xticks(range(len(method_list)))
        ax.set_xticklabels(method_list)
        ax.set_ylabel("planning success rate")
        ax.set_ylim(0, 1.0)
        title = "gLV intervention planning — success rate by method"
        if checkpoint is None:
            title += "\n(RANDOM model: harness check only, success ~0 expected)"
        else:
            title += f"\n(mean ± s.e., {len(seed_list)} seed(s), {n_episodes} episodes/seed)"
        ax.set_title(title)
        fig.tight_layout()
        fig_path = out_dir / "planning_success_rate.png"
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        print(f"saved figure  -> {fig_path}")
        payload["figure"] = str(fig_path)
    except Exception as e:  # never let plotting crash the report
        print(f"[plan] figure skipped: {type(e).__name__}: {e}")
        payload["figure_error"] = repr(e)

    return {"summary": summary, "results_json": str(res_path),
            "figure": payload.get("figure"), "tol": tol, "trained_model": checkpoint is not None}


if __name__ == "__main__":
    import fire

    fire.Fire(run)
