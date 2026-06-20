"""
Microbiome-JEPA: an energy-based JEPA world model for gut bacterial communities.

This is the headline EB-JEPA challenge -- carrying the recipe to a new, noisy,
high-dimensional biological modality -- with TWO microbiome-specific losses:

  * AlphaDiversityLoss  (diversity preservation): the latent must keep ecological
    alpha-diversity decodable, so imagined futures don't collapse the community's
    diversity structure.
  * PhyloDispersionLoss (soft-UniFrac): latent geometry must respect microbial
    phylogeny, using abundance-weighted mean ProkBERT embeddings as a tree-free
    phylogenetic descriptor.

Architecture (pure JEPA, no imposter / no reconstruction):
  encoder   = SetEncoder           (permutation-invariant, abundance-weighted DeepSets)
  predictor = RNNPredictor         (action-conditioned latent dynamics; diet/feeding)
  regularizer = VC_IDM_Sim_Regularizer (var+cov anti-collapse + temporal + inverse-dynamics)
  predcost  = SquareLossSeq        (prediction energy in latent space)

Run (smoke test):
  python -m examples.microbiome.main --fname examples/microbiome/cfgs/train.yaml \
      optim.epochs=2 logging.log_wandb=false
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import (
    FCGRSetEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    SetEncoder,
)
from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders
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


def _condition_label(coeffs):
    """Compact ablation label from the active microbiome-specific terms."""
    active = [name for name in ("div", "phylo", "tvar") if coeffs.get(name, 0) > 0]
    return "+".join(active) if active else "baseline"


class ResidualRNNPredictor(RNNPredictor):
    """RNN predictor that outputs the CHANGE: z_{t+1} = z_t + g(z_t, a).

    A zero-initialized delta head makes it start at EXACT identity (skill=1 at init),
    so it matches persistence / linear-AR by construction and only learns the small
    correction -- the fix for "a linear model beats the world-model predictor" on the
    slow microbiome trajectories (where the optimal map is ~identity + a tiny term).
    """

    def __init__(self, hidden_size, action_dim, num_layers=1, **_):
        super().__init__(hidden_size=hidden_size, action_dim=action_dim,
                         num_layers=num_layers, final_ln=None)
        self.delta_head = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, state, action):
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous()  # [1, B, D]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()   # [1, B, A]
        out, _ = self.rnn(rnn_input, rnn_state)                    # [1, B, D]
        delta = self.delta_head(out[0])                            # [B, D] (== 0 at init)
        delta = delta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    # [B, D, 1, 1, 1]
        return state + delta


class NormSetEncoder(SetEncoder):
    """SetEncoder with per-feature z-score on the token-MLP input. The log-abundance
    channel otherwise dwarfs the ProkBERT dims and dominates the VICReg variance term
    -- per-feature normalization is required. The RAW abundance is kept for the pooling
    weight + padding mask; only the MLP *features* are standardized. Stats are set once
    from the training set via set_feature_stats() and stored as buffers (checkpointed).
    """

    def __init__(self, emb_dim=384, h_d=256, out_d=128, abundance_weighted=True):
        super().__init__(emb_dim=emb_dim, h_d=h_d, out_d=out_d,
                         abundance_weighted=abundance_weighted)
        self.register_buffer("feat_mean", torch.zeros(emb_dim + 1))
        self.register_buffer("feat_std", torch.ones(emb_dim + 1))

    def set_feature_stats(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=self.feat_mean.dtype))
        self.feat_std.copy_(torch.as_tensor(std, dtype=self.feat_std.dtype).clamp_min(1e-6))

    def _forward(self, x):
        logab = x[:, self.emb_dim : self.emb_dim + 1]            # RAW -> pooling weight + mask
        m = self.feat_mean.view(1, -1, 1, 1)
        s = self.feat_std.view(1, -1, 1, 1)
        h = self.token_mlp((x - m) / s)                          # standardized features
        if self.abundance_weighted:
            w = torch.relu(logab)                                # padded slots (ab=0) -> 0
        else:
            w = (logab > 0).float()
        w = w / (w.sum(dim=2, keepdim=True) + 1e-6)
        z = (h * w).sum(dim=2, keepdim=True)
        z = self.post(z)
        return z


@torch.no_grad()
def _feature_stats(loader, emb_dim, device):
    """Per-channel mean/std over PRESENT OTU tokens (log-abundance > 0) across the
    training set, for the encoder's per-feature z-score (padded slots excluded)."""
    C = emb_dim + 1
    csum = torch.zeros(C, device=device)
    csqsum = torch.zeros(C, device=device)
    count = 0
    for b in loader:
        x = b["observations"].to(device)[..., 0]   # [B, C, T, N]
        present = x[:, emb_dim] > 0                 # [B, T, N]
        xp = x.permute(0, 2, 3, 1)[present]         # [P, C]
        csum += xp.sum(0)
        csqsum += (xp * xp).sum(0)
        count += xp.shape[0]
    mean = csum / max(1, count)
    std = (csqsum / max(1, count) - mean * mean).clamp_min(1e-8).sqrt()
    return mean.cpu(), std.cpu()


def build_jepa(cfg, action_dim, device):
    D = cfg.model.dstc
    enc_kind = cfg.model.get("encoder", "set")
    if enc_kind == "fcgr":
        # image-JEPA: each OTU token = an FCGR image of its DNA, read by a CNN.
        encoder = FCGRSetEncoder(k=cfg.model.get("fcgr_k", 6),
                                 h_d=cfg.model.get("fcgr_hd", 32), out_d=D)
    elif cfg.model.get("normalize_features", False):
        encoder = NormSetEncoder(emb_dim=cfg.model.emb_dim, h_d=cfg.model.henc, out_d=D)
    else:
        encoder = SetEncoder(emb_dim=cfg.model.emb_dim, h_d=cfg.model.henc, out_d=D)
    if cfg.model.get("residual_predictor", False):
        predictor = ResidualRNNPredictor(hidden_size=D, action_dim=action_dim)
    else:
        predictor = RNNPredictor(hidden_size=D, action_dim=action_dim,
                                 final_ln=nn.LayerNorm(D))
    action_encoder = nn.Identity()
    idm = InverseDynamicsModel(state_dim=D, hidden_dim=cfg.model.hpre, action_dim=action_dim)
    projector = Projector(f"{D}-{4*D}-{4*D}")
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
    """Pooled latents + labels + prediction/identity/div/phylo accumulators."""
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
        # temporal variance: how much a trajectory moves in latent over time (collapse monitor)
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
    """Val metrics. The T1D probe is fit on `fit_loader` (train) and scored on
    `val_loader` -- subject-disjoint, no leakage."""
    jepa.eval()
    tot, Xv, yv, av = _gather(jepa, alpha_loss, phylo_loss, val_loader, cfg, device)
    n = max(1, tot["n"])
    metrics = {k: v / n for k, v in tot.items() if k != "n"}
    metrics["skill_vs_identity"] = metrics["ident"] / max(1e-9, metrics["pred"])
    metrics["effrank"] = effective_rank(Xv)  # collapse monitor: dims actually used
    if fit_loader is not None:
        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.metrics import r2_score, roc_auc_score
            _, Xt, yt, at = _gather(jepa, alpha_loss, phylo_loss, fit_loader, cfg, device)
            # microbiome aging clock: predict host age from the community latent
            reg = Ridge(alpha=1.0).fit(Xt, at)
            metrics["age_r2"] = float(r2_score(av, reg.predict(Xv)))
            # secondary, harder probe: host T1D phenotype
            if len(np.unique(yt)) == 2 and len(np.unique(yv)) == 2:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt, yt)
                metrics["t1d_auroc"] = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
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

    dcfg = MicrobiomeConfig(
        cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
        tp_stride=cfg.data.get("tp_stride", 1),
        n_max=cfg.data.n_max, emb_dim=cfg.model.emb_dim,
        val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed,
        fcgr_path=cfg.data.get("fcgr_path", None),
    )
    train_ds, val_ds, train_loader, val_loader = make_loaders(
        dcfg, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )
    A = train_ds.action_dim
    print(f"== microbiome-jepa | device={device} | action_dim={A} | "
          f"train_windows={len(train_ds)} val_windows={len(val_ds)} ==")

    jepa = build_jepa(cfg, A, device)
    if cfg.model.get("normalize_features", False) and hasattr(jepa.encoder, "set_feature_stats"):
        fmean, fstd = _feature_stats(train_loader, cfg.model.emb_dim, device)
        jepa.encoder.set_feature_stats(fmean, fstd)
        print(f"== feature-norm ON: per-dim z-score | abundance ch mean={fmean[-1]:.2f} std={fstd[-1]:.2f} ==")
    alpha_loss = AlphaDiversityLoss(state_dim=cfg.model.dstc).to(device)
    phylo_loss = PhyloDispersionLoss().to(device)
    tvar_loss = TemporalVarianceLoss(margin=cfg.loss.get("tvar_margin", 1.0))

    n_params = sum(p.numel() for p in jepa.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in jepa.encoder.parameters())
    print(f"== params: total={n_params / 1e6:.2f}M (encoder={n_enc / 1e6:.2f}M) ==")

    params = list(jepa.parameters()) + list(alpha_loss.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = CosineWithWarmup(opt, total_steps=max(1, len(train_loader) * cfg.optim.epochs),
                             warmup_ratio=0.1, min_lr=cfg.optim.lr * 0.01)

    ld, lp, lt = cfg.loss.div_coeff, cfg.loss.phylo_coeff, cfg.loss.get("tvar_coeff", 0.0)
    pc = cfg.loss.get("pred_coeff", 1.0)
    probe_every = int(cfg.optim.get("probe_every", 5))
    last_val = {}
    for ep in range(1, cfg.optim.epochs + 1):
        jepa.train()
        agg = {"loss": 0.0, "rloss": 0.0, "ploss": 0.0, "div": 0.0, "phylo": 0.0, "tvarl": 0.0, "n": 0}
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
            total = rloss + pc * ploss + ld * l_div + lp * l_phylo + lt * l_tvar

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            b = obs.shape[0]
            ploss_val = ploss.item() if torch.is_tensor(ploss) else float(ploss)
            agg["loss"] += total.item() * b; agg["rloss"] += rloss.item() * b
            agg["ploss"] += ploss_val * b; agg["div"] += l_div.item() * b
            agg["phylo"] += l_phylo.item() * b; agg["tvarl"] += l_tvar.item() * b
            agg["n"] += b
        n = max(1, agg["n"])
        do_probe = (ep % probe_every == 0) or (ep == cfg.optim.epochs)
        val = evaluate(jepa, alpha_loss, phylo_loss, val_loader, cfg, device,
                       fit_loader=train_loader if do_probe else None)
        last_val = val
        print(f"[ep {ep:03d}] train loss={agg['loss']/n:.3f} pred={agg['ploss']/n:.4f} "
              f"div={agg['div']/n:.4f} phylo={agg['phylo']/n:.4f} tvarL={agg['tvarl']/n:.4f} "
              f"reg={agg['rloss']/n:.3f} || val skill={val['skill_vs_identity']:.3f}x "
              f"tvar={val['tvar']:.4f} effrank={val['effrank']:.1f} "
              f"age_r2={val.get('age_r2', float('nan')):.3f} "
              f"t1d_auroc={val.get('t1d_auroc', float('nan')):.3f}")

    ckpt_dir = os.environ.get("EBJEPA_CKPTS", "checkpoints/microbiome")
    os.makedirs(ckpt_dir, exist_ok=True)
    out = os.path.join(ckpt_dir, "microbiome_jepa.pt")
    torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg),
                "milk_vocab": train_ds.milk_vocab}, out)
    print(f"saved -> {out}")

    coeffs = {
        "div": float(ld), "phylo": float(lp), "tvar": float(lt),
        "std": float(cfg.loss.std_coeff), "cov": float(cfg.loss.cov_coeff),
        "idm": float(cfg.loss.idm_coeff), "sim_t": float(cfg.loss.sim_coeff_t),
    }
    summary = {
        "seed": int(cfg.meta.seed), "epochs": int(cfg.optim.epochs),
        "params_M": n_params / 1e6, "coeffs": coeffs,
        "condition": _condition_label(coeffs),
        "metrics": {k: float(v) for k, v in last_val.items()},
    }
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"metrics -> {os.path.join(ckpt_dir, 'metrics.json')}  "
          f"[{summary['condition']}] "
          f"skill={last_val.get('skill_vs_identity', float('nan')):.3f} "
          f"effrank={last_val.get('effrank', float('nan')):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/microbiome/cfgs/train.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
