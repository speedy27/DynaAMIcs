"""
Layer A — static community set-JEPA (two-view VICReg/BCS) for the microbiome modality.

This is the INTEGRATION file (owned by the orchestrator). It wires together the parts delivered by
the parallel workstreams against the pinned contracts:
  - WS1 data:    init_data("microbiome", ...) yields two augmented views of each community, each an
                 obs dict {"otu": [B,1,N_max,F], "mask": [B,1,N_max]} (F = emb_dim + 1).
  - WS2 encoder: SetTransformerEncoder(obs) -> state [B, D, T, 1, 1]; here T=1, flattened to [B, D].
  - eb_jepa:     VICRegLoss / BCS (two-view, return dict with "loss"), Projector, CosineWithWarmup.

Collapse watch (the rubric's headline concern): we log VICReg's var/cov/invariance sub-losses AND a
direct feature-std diagnostic (mean per-dim std of the embedding; -> 0 means the encoder collapsed).

Smoke (CPU, no data needed) — note fire override syntax is `--key value` (with `--`):
  .venv-cpu/bin/python -m examples.microbiome_jepa.main \
      --fname examples/microbiome_jepa/cfgs/layerA_vicreg.yaml \
      --optim.epochs 2 --data.size 256 --data.batch_size 32
  (log_wandb/use_amp default to false in the yaml; bare `key=value` without `--` binds to the
  positional `cfg` arg and breaks — always use `--key value`.)
"""

import copy
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

from eb_jepa.architectures import Projector, SetTransformerEncoder
from eb_jepa.datasets.utils import init_data
from eb_jepa.logging import get_logger
from eb_jepa.losses import BCS, VICRegLoss
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
    """Move an obs dict {"otu":..., "mask":...} to device."""
    return {k: v.to(device, non_blocking=True) for k, v in obs.items()}


def _unpack_views(batch):
    """two_view loader yields (view1_obs, view2_obs) or (view1_obs, view2_obs, label)."""
    if len(batch) == 2:
        return batch[0], batch[1], None
    return batch[0], batch[1], batch[2]


@torch.no_grad()
def feature_collapse_std(feat):
    """Mean per-dimension std of the embedding across the batch; -> 0 signals collapse."""
    return feat.float().std(dim=0).mean().item()


class CommunitySSL(nn.Module):
    """Set-transformer encoder + projector for two-view community SSL."""

    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def embed(self, obs):
        state = self.encoder(obs)  # [B, D, T, 1, 1], T=1 for static communities
        return state.flatten(1)  # [B, D]

    def forward(self, obs):
        feat = self.embed(obs)
        z = self.projector(feat)
        return feat, z


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerA_vicreg.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """Train a static community set-JEPA with two-view VICReg/BCS."""
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    if folder is None:
        exp_name = get_exp_name("microbiome_jepa", cfg)
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
        config={"example": "microbiome_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["microbiome_jepa", "layerA", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
    )

    # -- DATA (WS1): two-view community loader
    loader, val_loader, data_config, _ = init_data(
        env_name="microbiome",
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
        device=device,
    )

    # -- MODEL (WS2 encoder + eb_jepa projector)
    encoder = SetTransformerEncoder(
        token_dim=cfg.model.token_dim,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        pool=cfg.model.pool,
    )
    D = getattr(encoder, "mlp_output_dim", cfg.model.d_model)
    if cfg.model.use_projector:
        projector = Projector(f"{D}-{D * 4}-{D * 4}")
    else:
        projector = nn.Identity()
    model = CommunitySSL(encoder, projector).to(device)

    # EXP2 (GeneJepa): optional EMA TEACHER — view2's target comes from an EMA copy of the model
    # (stop-grad), a la BYOL/DINO/I-JEPA. A single attributable change on top of the SIGReg/VICReg setup.
    use_ema = bool(cfg.model.get("use_ema", False))
    ema_decay = float(cfg.model.get("ema_decay", 0.996))
    target_model = None
    if use_ema:
        target_model = copy.deepcopy(model)
        for p in target_model.parameters():
            p.requires_grad_(False)
        logger.info(f"EMA teacher ON (decay={ema_decay}): view2 target = EMA(model), stop-grad")

    log_model_info(
        model,
        {
            "encoder": sum(p.numel() for p in encoder.parameters()),
            "projector": sum(p.numel() for p in projector.parameters()),
        },
    )
    log_config(cfg)

    if cfg.loss.type == "vicreg":
        loss_fn = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff)
    elif cfg.loss.type == "bcs":
        loss_fn = BCS(lmbd=cfg.loss.lmbd)
    else:
        raise ValueError(f"Unknown loss.type={cfg.loss.type!r}; expected 'vicreg' or 'bcs'")

    steps_per_epoch = max(1, len(loader))
    total_steps = cfg.optim.epochs * steps_per_epoch
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-5),
    )
    scheduler = CosineWithWarmup(
        optimizer, total_steps, warmup_ratio=cfg.optim.get("warmup_ratio", 0.1)
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg.training.get("dtype", "bfloat16").lower(), torch.bfloat16
    )
    use_amp = cfg.training.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"device={device} amp={use_amp} dtype={dtype} D={D} steps/epoch={steps_per_epoch}")

    metrics = {}
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0 = time.time()
        agg = {}
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
        )
        for batch in pbar:
            v1, v2, _ = _unpack_views(batch)
            v1, v2 = obs_to(v1, device), obs_to(v2, device)

            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                f1, z1 = model(v1)
                if target_model is not None:
                    with torch.no_grad():
                        _, z2 = target_model(v2)   # EMA-teacher target (stop-grad)
                else:
                    _, z2 = model(v2)
                loss_dict = loss_fn(z1, z2)
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if target_model is not None:   # EMA update of the teacher
                with torch.no_grad():
                    for pt, ps in zip(target_model.parameters(), model.parameters()):
                        pt.data.mul_(ema_decay).add_(ps.data, alpha=1.0 - ema_decay)
                    for bt, bs in zip(target_model.buffers(), model.buffers()):
                        bt.data.copy_(bs.data)

            fstd = feature_collapse_std(f1)
            for k, v in loss_dict.items():
                agg[k] = agg.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
            agg["feat_std"] = agg.get("feat_std", 0.0) + fstd
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "feat_std": f"{fstd:.3f}"})

        nb = max(1, len(loader))
        metrics = {k: v / nb for k, v in agg.items()}
        logger.info(
            f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        if wandb_run and epoch % cfg.logging.get("log_every", 1) == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "epoch_time": time.time() - t0,
                    **{f"train/{k}": v for k, v in metrics.items()},
                }
            )

        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            encoder_state_dict=encoder.state_dict(),
        )

    if wandb_run:
        wandb.finish()
    logger.info(f"done. final metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    fire.Fire(run)
