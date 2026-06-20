"""
Microbiome2img-JEPA: the FCGR (DNA-as-image) encoder plugged into the *library*
EB-JEPA world model -- the integration of the microbiome2img probe with eb_jepa.

Same recipe as examples/microbiome, with ONE component swapped:

  encoder    = FCGRSetEncoder    (DNA-as-image: per-OTU FCGR image -> abundance pool)
  predictor  = RNNPredictor      (action-conditioned latent dynamics)            [library]
  regularizer= VC_IDM_Sim_Regularizer (var+cov anti-collapse + temporal + IDM)   [library]
  predcost   = SquareLossSeq     (prediction energy in latent space)             [library]
  + AlphaDiversityLoss + PhyloDispersionLoss + TemporalVarianceLoss (same aux terms)

This is the controlled "image-CGR vs ProkBERT-embedding" ablation: the JEPA, the
predictor, the regularizer, the losses and the metrics are IDENTICAL to the main
microbiome example -- only the per-OTU token representation changes. It trains today
on a synthetic FCGR cohort (examples/microbiome2img/synth.py) since the real cache
holds ProkBERT vectors, not raw sequences.

Run (smoke test):
  python -m examples.microbiome2img.main --fname examples/microbiome2img/cfgs/train.yaml \
      optim.epochs=2 logging.log_wandb=false
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import (
    FCGRSetEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import (
    AlphaDiversityLoss,
    PhyloDispersionLoss,
    SquareLossSeq,
    TemporalVarianceLoss,
    VC_IDM_Sim_Regularizer,
    effective_rank,
)
from eb_jepa.schedulers import CosineWithWarmup
from examples.microbiome2img.synth import SynthFCGRConfig, make_loaders


def build_jepa(cfg, action_dim, device):
    D = cfg.model.dstc
    encoder = FCGRSetEncoder(k=cfg.data.k, h_d=cfg.model.henc, out_d=D)
    predictor = RNNPredictor(hidden_size=D, action_dim=action_dim, final_ln=nn.LayerNorm(D))
    action_encoder = nn.Identity()
    idm = InverseDynamicsModel(state_dim=D, hidden_dim=cfg.model.hpre, action_dim=action_dim)
    projector = Projector(f"{D}-{4 * D}-{4 * D}")
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.loss.cov_coeff,
        std_coeff=cfg.loss.std_coeff,
        sim_coeff_t=cfg.loss.sim_coeff_t,
        idm_coeff=cfg.loss.idm_coeff,
        idm=idm,
        projector=projector,
        first_t_only=False,
    )
    predcost = SquareLossSeq()
    return JEPA(encoder, action_encoder, predictor, regularizer, predcost).to(device)


@torch.no_grad()
def _gather(jepa, alpha_loss, phylo_loss, loader, cfg, device):
    tot = {"pred": 0.0, "ident": 0.0, "div": 0.0, "phylo": 0.0, "tvar": 0.0, "n": 0}
    feats, labels, ages = [], [], []
    for batch in loader:
        obs = batch["observations"].to(device)
        act = batch["actions"].to(device)
        state = jepa.encoder(obs)  # [B, D, T, 1, 1]
        preds, _ = jepa.unroll(obs, act, nsteps=cfg.model.nsteps,
                               unroll_mode="autoregressive", compute_loss=False,
                               return_all_steps=False)
        Tn = min(state.shape[2], preds.shape[2])
        b = obs.shape[0]
        tot["pred"] += torch.mean((preds[:, :, 1:Tn] - state[:, :, 1:Tn]) ** 2).item() * b
        tot["ident"] += torch.mean((state[:, :, : Tn - 1] - state[:, :, 1:Tn]) ** 2).item() * b
        tot["tvar"] += state[..., 0, 0].var(dim=2).mean().item() * b
        tot["div"] += alpha_loss(state, batch["diversity"].to(device)).item() * b
        tot["phylo"] += phylo_loss(state, batch["phylo"].to(device)).item() * b
        tot["n"] += b
        feats.append(state.mean(dim=2)[..., 0, 0].cpu().numpy())  # [B, D] subject-pooled
        labels.append(batch["label"].numpy())
        ages.append(batch["age"].numpy())
    return (tot, np.concatenate(feats), np.concatenate(labels).astype(int),
            np.concatenate(ages))


def evaluate(jepa, alpha_loss, phylo_loss, val_loader, cfg, device, fit_loader=None):
    """Val metrics. Probes are fit on `fit_loader` (train subjects) and scored on
    `val_loader` -- subject-disjoint, no leakage."""
    jepa.eval()
    tot, Xv, yv, av = _gather(jepa, alpha_loss, phylo_loss, val_loader, cfg, device)
    n = max(1, tot["n"])
    metrics = {k: v / n for k, v in tot.items() if k != "n"}
    metrics["skill_vs_identity"] = metrics["ident"] / max(1e-9, metrics["pred"])
    metrics["effrank"] = effective_rank(Xv)
    if fit_loader is not None:
        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.metrics import r2_score, roc_auc_score
            _, Xt, yt, at = _gather(jepa, alpha_loss, phylo_loss, fit_loader, cfg, device)
            reg = Ridge(alpha=1.0).fit(Xt, at)
            metrics["age_r2"] = float(r2_score(av, reg.predict(Xv)))
            if len(np.unique(yt)) == 2 and len(np.unique(yv)) == 2:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt, yt)
                metrics["pheno_auroc"] = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
        except Exception:
            pass
    return metrics


def run(fname, overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)
    np.random.seed(cfg.meta.seed)

    dcfg = SynthFCGRConfig(
        k=cfg.data.k, n_clades=cfg.data.n_clades, otus_per_clade=cfg.data.otus_per_clade,
        seq_len=cfg.data.seq_len, between=cfg.data.between, divergence=cfg.data.divergence,
        n_subjects=cfg.data.n_subjects, n_window=cfg.data.n_window, n_max=cfg.data.n_max,
        decay=cfg.data.decay, diet_strength=cfg.data.diet_strength, noise=cfg.data.noise,
        val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed,
    )
    train_ds, val_ds, train_loader, val_loader = make_loaders(
        dcfg, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )
    A = train_ds.action_dim
    print(f"== microbiome2img-jepa (FCGR) | device={device} | action_dim={A} | "
          f"S={train_ds.S}x{train_ds.S} K={train_ds.K} | "
          f"train={len(train_ds)} val={len(val_ds)} ==")

    jepa = build_jepa(cfg, A, device)
    alpha_loss = AlphaDiversityLoss(state_dim=cfg.model.dstc).to(device)
    phylo_loss = PhyloDispersionLoss().to(device)
    tvar_loss = TemporalVarianceLoss(margin=cfg.loss.get("tvar_margin", 1.0))

    n_params = sum(p.numel() for p in jepa.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in jepa.encoder.parameters())
    print(f"== params: total={n_params / 1e6:.2f}M (FCGR encoder={n_enc / 1e6:.2f}M) ==")

    params = list(jepa.parameters()) + list(alpha_loss.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = CosineWithWarmup(opt, total_steps=max(1, len(train_loader) * cfg.optim.epochs),
                             warmup_ratio=0.1, min_lr=cfg.optim.lr * 0.01)

    ld, lp, lt = cfg.loss.div_coeff, cfg.loss.phylo_coeff, cfg.loss.get("tvar_coeff", 0.0)
    probe_every = int(cfg.optim.get("probe_every", 5))
    last_val = {}
    for ep in range(1, cfg.optim.epochs + 1):
        jepa.train()
        agg = {"loss": 0.0, "ploss": 0.0, "div": 0.0, "phylo": 0.0, "tvarl": 0.0, "n": 0}
        for batch in train_loader:
            obs = batch["observations"].to(device)
            act = batch["actions"].to(device)
            preds, (loss, rloss, runw, rdict, ploss) = jepa.unroll(
                obs, act, nsteps=cfg.model.nsteps, unroll_mode="autoregressive",
                compute_loss=True,
            )
            state = jepa.encoder(obs)
            l_div = alpha_loss(state, batch["diversity"].to(device))
            l_phylo = phylo_loss(state, batch["phylo"].to(device))
            l_tvar = tvar_loss(state)
            total = loss + ld * l_div + lp * l_phylo + lt * l_tvar

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            b = obs.shape[0]
            ploss_val = ploss.item() if torch.is_tensor(ploss) else float(ploss)
            agg["loss"] += total.item() * b
            agg["ploss"] += ploss_val * b
            agg["div"] += l_div.item() * b
            agg["phylo"] += l_phylo.item() * b
            agg["tvarl"] += l_tvar.item() * b
            agg["n"] += b
        n = max(1, agg["n"])
        do_probe = (ep % probe_every == 0) or (ep == cfg.optim.epochs)
        val = evaluate(jepa, alpha_loss, phylo_loss, val_loader, cfg, device,
                       fit_loader=train_loader if do_probe else None)
        last_val = val
        print(f"[ep {ep:03d}] train loss={agg['loss']/n:.3f} pred={agg['ploss']/n:.4f} "
              f"div={agg['div']/n:.4f} phylo={agg['phylo']/n:.4f} tvarL={agg['tvarl']/n:.4f} "
              f"|| val skill={val['skill_vs_identity']:.3f}x tvar={val['tvar']:.4f} "
              f"effrank={val['effrank']:.1f} age_r2={val.get('age_r2', float('nan')):.3f} "
              f"pheno_auroc={val.get('pheno_auroc', float('nan')):.3f}")

    ckpt_dir = os.environ.get("EBJEPA_CKPTS", "checkpoints/microbiome2img")
    os.makedirs(ckpt_dir, exist_ok=True)
    out = os.path.join(ckpt_dir, "microbiome2img_jepa.pt")
    torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg)}, out)
    print(f"saved -> {out}")

    summary = {
        "encoder": "fcgr",
        "seed": int(cfg.meta.seed), "epochs": int(cfg.optim.epochs),
        "params_M": n_params / 1e6,
        "metrics": {k: float(v) for k, v in last_val.items()},
    }
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"metrics -> {os.path.join(ckpt_dir, 'metrics.json')}  "
          f"skill={last_val.get('skill_vs_identity', float('nan')):.3f} "
          f"effrank={last_val.get('effrank', float('nan')):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/microbiome2img/cfgs/train.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
