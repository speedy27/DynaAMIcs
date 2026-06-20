"""
Layer B — action-conditioned temporal JEPA world model on the gLV simulator.

THE HEADLINE EXPERIMENT (IDM ablation, Sobal-et-al. analog):
  Train the same world model with the inverse-dynamics term ON (idm_coeff>0) vs OFF (idm_coeff=0).
  Hypothesis: WITHOUT IDM a VICReg-style temporal JEPA collapses onto SLOW features (predicting the
  next latent becomes trivial -> pred_loss -> 0 while the variance hinge stays active), so the encoder
  ignores real community dynamics. WITH IDM (the encoder must let the inverse model recover the action
  from consecutive latents) the encoder is forced to represent the fast community dynamics.
  We log Lpred (pred_loss) and Lvar (std_loss) every step to watch this directly.

Wiring (all contracts verified in CLAUDE.md "Repo reality check"):
  encoder  = SetTransformerEncoder(obs dict -> [B, D, T, 1, 1])                 # WS2
  predictor= RNNPredictor(hidden_size=D, action_dim=K, final_ln=encoder.final_ln)  # GRU: action=input, state=hidden
  aencoder = Identity (raw K-dim action delta on the candidate panel)
  idm      = InverseDynamicsModel(state_dim=D, action_dim=K)
  reg      = VC_IDM_Sim_Regularizer(std, cov, sim_t, idm_coeff, idm, projector, ...)  # forward(state,[B,K,T])
  jepa     = JEPA(encoder, aencoder, predictor, reg, SquareLossSeq())
  data     = init_data("microbiome", task=glv) -> (obs_dict, act[B,T,K], state[B,T,S], reward); we
             transpose act -> [B,K,T] for unroll/IDM.

Smoke (CPU):
  .venv-cpu/bin/python -m examples.microbiome_jepa.train_worldmodel \
      --fname examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml \
      --optim.epochs 2 --data.n_traj 32 --data.batch_size 16
"""

import time
from pathlib import Path

import fire
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    SetTransformerEncoder,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.losses import SIGReg_IDM_Sim_Regularizer, SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)

logger = get_logger(__name__)


def obs_to(obs, device):
    return {k: v.to(device, non_blocking=True) for k, v in obs.items()}


@torch.no_grad()
def feature_collapse_std(jepa, obs):
    """Mean per-dim std of the FIRST-timepoint latent across the batch; -> 0 = collapse."""
    state = jepa.encode(obs)  # [B, D, T, 1, 1]
    z0 = state[:, :, 0].flatten(1).float()  # [B, D]
    return z0.std(dim=0).mean().item()


def isometry_metric_loss(z, x, log_scale, n_pairs, gen):
    """HYBRID metric-preserving (isometry) auxiliary — NOT pure JEPA (it uses TRUE-state supervision).

    Penalize the gap between the LATENT distance ||z_a - z_b|| and the TRUE gLV state distance
    ||x_a - x_b|| (Euclidean in raw S-dim abundance space, the exact metric planning success is
    measured in), up to a single learned global scale exp(log_scale). Minimizing this bakes the true
    metric into the latent so RAW latent distance becomes an informative planning cost (the M3 wall the
    diagnosis pinned: weak-reg latent-vs-true distance corr ~0, decode R^2 ~0.89 — metric, not dynamics).

    Sampled pairs (seeded) keep it O(n_pairs) and bounded; squared-distance clamp keeps the sqrt grad
    finite even if two sampled states coincide. The (still-active) VICReg std/cov terms prevent the
    trivial collapse-to-a-point solution that would also drive this loss to zero.
    """
    M = z.shape[0]
    i = torch.randint(0, M, (n_pairs,), generator=gen)
    j = torch.randint(0, M, (n_pairs,), generator=gen)
    j = torch.where(i == j, (j + 1) % M, j)            # never pair a sample with itself
    i, j = i.to(z.device), j.to(z.device)
    dz = ((z[i] - z[j]) ** 2).sum(-1).clamp_min(1e-12).sqrt()   # latent distances
    dx = ((x[i] - x[j]) ** 2).sum(-1).clamp_min(1e-12).sqrt()   # true-state distances
    return ((dz - log_scale.exp() * dx) ** 2).mean()


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    cfg=None,
    folder=None,
    return_model: bool = False,
    **overrides,
):
    """Train the action-conditioned gLV world model (IDM ablation via model.regularizer.idm_coeff).

    Returns ``metrics`` (dict), or ``(metrics, jepa)`` when ``return_model=True`` (used by run_ablation).
    """
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    idm_coeff = float(cfg.model.regularizer.idm_coeff)
    tag = "idm_on" if idm_coeff > 0 else "idm_off"

    if folder is None:
        exp_name = f"{get_exp_name('microbiome_wm', cfg)}_{tag}"
        exp_dir = get_unified_experiment_dir(
            example_name="microbiome_jepa",
            sweep_name=get_default_dev_name(),
            exp_name=exp_name,
            seed=cfg.meta.seed,
        )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_name = exp_dir.name.rsplit("_seed", 1)[0]

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "microbiome_wm", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["microbiome_jepa", "layerB", tag, f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
    )

    # -- DATA (WS1 gLV trajectory loader)
    loader, val_loader, data_config, _ = init_data(
        env_name="microbiome",
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
        device=device,
    )
    K = int(data_config.action_dim)
    token_dim = int(data_config.token_dim)
    logger.info(f"gLV loader: action_dim K={K} token_dim={token_dim} "
                f"n_max={data_config.n_max} steps/epoch={len(loader)} stub={data_config.extra}")

    # -- MODEL
    rcfg = cfg.model.regularizer
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
    reg_type = str(rcfg.get("type", "vicreg")).lower()

    if reg_type == "sigreg":
        # SIGReg (LeJEPA) anti-collapse on the ENCODER output (no projector), so it shapes the exact
        # latent space the planner measures distance in (the M3 geometry hypothesis).
        idm = InverseDynamicsModel(
            state_dim=D, hidden_dim=int(rcfg.get("idm_hidden", 256)), action_dim=K
        ).to(device)
        regularizer = SIGReg_IDM_Sim_Regularizer(
            sigreg_coeff=float(rcfg.get("sigreg_coeff", 1.0)),
            sim_coeff_t=rcfg.sim_coeff_t,
            idm_coeff=idm_coeff,
            idm=idm,
            num_slices=int(rcfg.get("num_slices", 256)),
            first_t_only=rcfg.get("first_t_only", False),
        )
    else:
        projector = Projector(f"{D}-{D * 4}-{D * 4}") if rcfg.get("use_proj", True) else None
        idm_state_dim = (projector.out_dim if (projector is not None and rcfg.get("idm_after_proj", False)) else D)
        idm = InverseDynamicsModel(
            state_dim=idm_state_dim, hidden_dim=int(rcfg.get("idm_hidden", 256)), action_dim=K
        ).to(device)
        regularizer = VC_IDM_Sim_Regularizer(
            cov_coeff=rcfg.cov_coeff,
            std_coeff=rcfg.std_coeff,
            sim_coeff_t=rcfg.sim_coeff_t,
            idm_coeff=idm_coeff,
            idm=idm,
            first_t_only=rcfg.get("first_t_only", False),
            projector=projector,
            spatial_as_samples=rcfg.get("spatial_as_samples", False),
            idm_after_proj=rcfg.get("idm_after_proj", False),
            sim_t_after_proj=rcfg.get("sim_t_after_proj", False),
        )
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    # --- HYBRID metric-preserving auxiliary (config-gated; default OFF => behavior unchanged) ---
    # metric_coeff>0 adds an isometry term (latent dist -> TRUE gLV state dist) on top of the pure-JEPA
    # objective. This uses ground-truth state supervision, so it is explicitly a HYBRID ablation, NOT
    # pure JEPA. exp(metric_scale) is a single learned global scale matching latent<->state units.
    metric_coeff = float(rcfg.get("metric_coeff", 0.0))
    metric_pairs = int(rcfg.get("metric_pairs", 4096))
    metric_scale = torch.nn.Parameter(torch.zeros(1, device=device)) if metric_coeff > 0 else None
    metric_gen = torch.Generator(device="cpu").manual_seed(int(cfg.meta.seed)) if metric_coeff > 0 else None
    if metric_coeff > 0:
        logger.info(f"=== HYBRID metric loss ON: metric_coeff={metric_coeff} pairs={metric_pairs} "
                    f"(isometry latent->TRUE-state dist; uses true-state supervision => NOT pure JEPA) ===")

    log_model_info(
        jepa,
        {
            "encoder": sum(p.numel() for p in encoder.parameters()),
            "predictor": sum(p.numel() for p in predictor.parameters()),
            "idm": sum(p.numel() for p in idm.parameters()),
        },
    )
    log_config(cfg)
    logger.info(f"=== ABLATION: idm_coeff={idm_coeff} ({tag}) ===  D={D} K={K}")

    steps_per_epoch = max(1, len(loader))
    total_steps = cfg.optim.epochs * steps_per_epoch
    opt_params = list(jepa.parameters()) + ([metric_scale] if metric_scale is not None else [])
    optimizer = AdamW(opt_params, lr=cfg.optim.lr,
                      weight_decay=cfg.optim.get("weight_decay", 1e-5))
    scheduler = CosineWithWarmup(optimizer, total_steps, warmup_ratio=cfg.optim.get("warmup_ratio", 0.1))

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg.training.get("dtype", "bfloat16").lower(), torch.bfloat16
    )
    use_amp = cfg.training.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    grad_clip = cfg.optim.get("grad_clip", 0.0)

    metrics = {}
    for epoch in range(cfg.optim.epochs):
        jepa.train()
        t0 = time.time()
        agg = {}
        last_obs = None
        pbar = tqdm(loader, desc=f"[{tag}] Epoch {epoch}/{cfg.optim.epochs - 1}",
                    disable=cfg.logging.get("tqdm_silent", False))
        for batch in pbar:
            obs, act = batch[0], batch[1]
            obs = obs_to(obs, device)
            act = act.transpose(1, 2).to(device, non_blocking=True)  # [B,T,K] -> [B,K,T]

            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                _, (loss, regl, regl_unw, regldict, pl) = jepa.unroll(
                    obs, act,
                    nsteps=cfg.model.nsteps,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=True,
                    return_all_steps=False,
                )
                m_loss = None
                if metric_coeff > 0:
                    # second (grad-enabled) encode of the SAME obs; batch[2] is the TRUE [B,Tw,S] state.
                    state_true = batch[2].to(device, non_blocking=True).float()      # [B, Tw, S]
                    z5 = jepa.encoder(obs)                                            # [B, D, Tw, 1, 1]
                    Bc, Dc, Tc = z5.shape[0], z5.shape[1], z5.shape[2]
                    zc = z5[..., 0, 0].permute(0, 2, 1).reshape(Bc * Tc, Dc)          # [B*Tw, D]
                    xc = state_true.reshape(Bc * Tc, -1)                              # [B*Tw, S]
                    m_loss = isometry_metric_loss(zc, xc, metric_scale, metric_pairs, metric_gen)
                    loss = loss + metric_coeff * m_loss
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(jepa.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            row = {"total": loss.item(), "pred": pl.item() if torch.is_tensor(pl) else float(pl),
                   "reg": regl.item(), **regldict}
            if m_loss is not None:
                row["metric_loss"] = float(m_loss.detach())
                row["metric_scale"] = float(metric_scale.detach().exp())
            for k, v in row.items():
                agg[k] = agg.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
            last_obs = obs  # for the once-per-epoch collapse diagnostic (avoid a per-step extra encode)
            pbar.set_postfix({"tot": f"{loss.item():.3f}", "pred": f"{row['pred']:.4f}",
                              "std_l": f"{regldict.get('std_loss', 0):.3f}"})

        nb = max(1, len(loader))
        metrics = {k: v / nb for k, v in agg.items()}
        metrics["feat_std"] = feature_collapse_std(jepa, last_obs) if last_obs is not None else 0.0
        logger.info(f"[{tag}] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if wandb_run and epoch % cfg.logging.get("log_every", 1) == 0:
            wandb.log({"epoch": epoch, "epoch_time": time.time() - t0, "idm_coeff": idm_coeff,
                       **{f"train/{k}": v for k, v in metrics.items()}})

        save_checkpoint(exp_dir / "latest.pth.tar", model=jepa, optimizer=optimizer, epoch=epoch,
                        encoder_state_dict=encoder.state_dict(), idm_coeff=idm_coeff)

    if wandb_run:
        wandb.finish()
    logger.info(f"[{tag}] done. final metrics: {metrics}")
    return (metrics, jepa) if return_model else metrics


if __name__ == "__main__":
    fire.Fire(run)
