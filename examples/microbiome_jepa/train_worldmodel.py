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
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
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


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """Train the action-conditioned gLV world model (IDM ablation via model.regularizer.idm_coeff)."""
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
    optimizer = AdamW(jepa.parameters(), lr=cfg.optim.lr,
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
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(jepa.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            fstd = feature_collapse_std(jepa, obs)
            row = {"total": loss.item(), "pred": pl.item() if torch.is_tensor(pl) else float(pl),
                   "reg": regl.item(), "feat_std": fstd, **regldict}
            for k, v in row.items():
                agg[k] = agg.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
            pbar.set_postfix({"tot": f"{loss.item():.3f}", "pred": f"{row['pred']:.4f}",
                              "std_l": f"{regldict.get('std_loss', 0):.3f}", "fstd": f"{fstd:.3f}"})

        nb = max(1, len(loader))
        metrics = {k: v / nb for k, v in agg.items()}
        logger.info(f"[{tag}] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if wandb_run and epoch % cfg.logging.get("log_every", 1) == 0:
            wandb.log({"epoch": epoch, "epoch_time": time.time() - t0, "idm_coeff": idm_coeff,
                       **{f"train/{k}": v for k, v in metrics.items()}})

        save_checkpoint(exp_dir / "latest.pth.tar", model=jepa, optimizer=optimizer, epoch=epoch,
                        encoder_state_dict=encoder.state_dict(), idm_coeff=idm_coeff)

    if wandb_run:
        wandb.finish()
    logger.info(f"[{tag}] done. final metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    fire.Fire(run)
