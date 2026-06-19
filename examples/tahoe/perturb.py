"""
Tahoe perturbation WORLD MODEL — a true EB-JEPA with a frozen pretrained encoder.

  encoder f_θ   = FROZEN MosaicFM-3B embedding (identity over precomputed vectors)
  predictor g_φ = RNNPredictor, ACTION-conditioned on the drug (Morgan fingerprint)
  energy        = || g_φ(z_control, drug) − z_perturbed ||²   via eb_jepa.JEPA.unroll
  biology losses: PerturbationSignatureLoss (drug-signature consistency)
                  + PathwayCoherenceLoss   (gene-program / embedding-module structure)

Only the predictor is trained (the encoder is frozen) — exactly the "frozen
pretrained encoder + learned predictor" world-model setting. The drug induces a
large, controlled state change, so the no-change baseline is beatable (skill > 1),
unlike the slow microbiome trajectories.

Training curves are saved to:
  <ckpt>/tahoe_perturb_curves.json   (incremental, one entry per epoch)
  <ckpt>/tahoe_perturb_curves.png    (figure generated at end of training)

  python -m examples.tahoe.perturb --fname examples/tahoe/cfgs/perturb.yaml
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import RNNPredictor
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, PathwayCoherenceLoss, PerturbationSignatureLoss
from eb_jepa.datasets.tahoe.pert_dataset import PertConfig, make_loaders


class FrozenIdentityEncoder(nn.Module):
    """The state IS the frozen MosaicFM embedding -> identity, no trainable params."""
    def forward(self, x):
        return x


class NoReg(nn.Module):
    """No anti-collapse term needed: the encoder is frozen, so it cannot collapse."""
    def forward(self, state, actions=None):
        z = (state.sum() * 0.0)
        return z, z, {}


def _states(obs):
    z_ctrl = obs[:, :, 0, 0, 0]
    z_pert = obs[:, :, 1, 0, 0]
    return z_ctrl, z_pert


@torch.no_grad()
def evaluate(jepa, loader, device, train_shift=None):
    jepa.eval()
    tot = {"pred": 0.0, "ident": 0.0, "mshift": 0.0, "n": 0}
    for b in loader:
        obs = b["observations"].to(device); act = b["actions"].to(device)
        z_ctrl, z_pert = _states(obs)
        preds, _ = jepa.unroll(obs, act, nsteps=1, unroll_mode="autoregressive", compute_loss=False)
        pred = preds[:, :, -1, 0, 0]
        n = obs.shape[0]
        tot["pred"] += torch.mean((pred - z_pert) ** 2).item() * n
        tot["ident"] += torch.mean((z_ctrl - z_pert) ** 2).item() * n
        if train_shift is not None:
            ms = z_ctrl + train_shift[b["drug"].to(device)]
            tot["mshift"] += torch.mean((ms - z_pert) ** 2).item() * n
        tot["n"] += n
    n = max(1, tot["n"])
    m = {k: v / n for k, v in tot.items() if k != "n"}
    m["skill_vs_identity"] = m["ident"] / max(1e-9, m["pred"])
    m["skill_vs_meanshift"] = m["mshift"] / max(1e-9, m["pred"]) if train_shift is not None else float("nan")
    return m


def compute_mean_shift(loader, n_drugs, D, device):
    """Average (z_pert - z_ctrl) per drug over the training set (mean-shift baseline)."""
    s = torch.zeros(n_drugs, D, device=device); c = torch.zeros(n_drugs, device=device)
    for b in loader:
        obs = b["observations"].to(device); z_ctrl, z_pert = _states(obs)
        d = b["drug"].to(device)
        s.index_add_(0, d, (z_pert - z_ctrl)); c.index_add_(0, d, torch.ones_like(d, dtype=s.dtype))
    return s / c.clamp_min(1).unsqueeze(1)


def save_curves_plot(curves, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PRIM = "#0f2d50"; SEC = "#009688"
        epochs = [c["epoch"] for c in curves]

        val_epochs = [c["epoch"] for c in curves if "val_pred" in c]
        has_val = len(val_epochs) > 0

        fig, axes = plt.subplots(1, 3 if has_val else 2, figsize=(15 if has_val else 10, 4.5))

        # training losses (linear)
        ax = axes[0]
        ax.plot(epochs, [c["train_pred"] for c in curves], color=PRIM, lw=1.5, label="pred MSE")
        ax.plot(epochs, [c["train_sig"] for c in curves], color=SEC, lw=1.5, label="signature")
        ax.plot(epochs, [c["train_path"] for c in curves], color="#7e57c2", lw=1.5, label="pathway")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title("Training losses", color=PRIM)
        ax.legend(); ax.grid(True, alpha=0.3)

        # training losses (log scale)
        ax = axes[1]
        ax.semilogy(epochs, [c["train_pred"] for c in curves], color=PRIM, lw=1.5, label="pred MSE")
        ax.semilogy(epochs, [c["train_sig"] for c in curves], color=SEC, lw=1.5, label="signature")
        ax.semilogy(epochs, [c["train_path"] for c in curves], color="#7e57c2", lw=1.5, label="pathway")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)")
        ax.set_title("Training losses (log scale)", color=PRIM)
        ax.legend(); ax.grid(True, alpha=0.3)

        # val skill curves
        if has_val:
            ax = axes[2]
            skill_id = [c["skill_vs_identity"] for c in curves if "val_pred" in c]
            skill_ms = [c.get("skill_vs_meanshift", float("nan")) for c in curves if "val_pred" in c]
            ax.plot(val_epochs[:len(skill_id)], skill_id, color=SEC, lw=1.5, marker="o",
                    markersize=3, label="skill vs no-effect")
            finite_ms = [(e, s) for e, s in zip(val_epochs, skill_ms)
                         if not (isinstance(s, float) and s != s)]
            if finite_ms:
                ve, vs = zip(*finite_ms)
                ax.plot(list(ve), list(vs), color=PRIM, lw=1.5, marker="s",
                        markersize=3, label="skill vs mean-shift")
            ax.axhline(1.0, ls="--", color="gray", lw=1, label="baseline = 1×")
            ax.set_xlabel("epoch"); ax.set_ylabel("skill (ratio)")
            ax.set_title("Val skill (higher = better)", color=PRIM)
            ax.legend(); ax.grid(True, alpha=0.3)

        fig.suptitle("Tahoe perturbation world model — training curves", color=PRIM, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"curves plot -> {out_path}")
    except Exception as e:
        print(f"[warn] could not save curves plot: {e}")


def run(fname, overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed); np.random.seed(cfg.meta.seed)

    dcfg = PertConfig(cache_path=cfg.data.cache_path, val_fraction=cfg.data.val_fraction,
                      seed=cfg.meta.seed)
    tr, va, train_loader, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size,
                                                    num_workers=cfg.data.num_workers)
    D, A = tr.D, tr.action_dim
    print(f"== tahoe-perturb EB-JEPA | device={device} | D={D} action_dim={A} | "
          f"train={len(tr)} val={len(va)} | epochs={cfg.optim.epochs} "
          f"| control={'DMSO' if tr.has_ctrl else 'centroid'} ==")

    encoder = FrozenIdentityEncoder()
    n_layers = int(cfg.model.get("layers", 1)) if "model" in cfg else 1
    predictor = RNNPredictor(hidden_size=D, action_dim=A, num_layers=n_layers, final_ln=nn.LayerNorm(D))
    jepa = JEPA(encoder, nn.Identity(), predictor, NoReg(), SquareLossSeq()).to(device)
    sig = PerturbationSignatureLoss().to(device)
    path = PathwayCoherenceLoss().to(device)
    ls, lp = cfg.loss.sig_coeff, cfg.loss.path_coeff

    opt = torch.optim.AdamW(predictor.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.optim.epochs, eta_min=cfg.optim.lr * 0.01)

    train_shift = compute_mean_shift(train_loader, len(tr.drug_names), D, device)

    ckpt_dir = os.environ.get("EBJEPA_CKPTS", "checkpoints/tahoe")
    os.makedirs(ckpt_dir, exist_ok=True)
    curves_json = os.path.join(ckpt_dir, "tahoe_perturb_curves.json")
    curves_png = os.path.join(ckpt_dir, "tahoe_perturb_curves.png")
    curves = []
    best = {"skill_vs_meanshift": -1.0}

    for ep in range(1, cfg.optim.epochs + 1):
        jepa.train()
        agg = {"pred": 0.0, "sig": 0.0, "path": 0.0, "n": 0}
        for b in train_loader:
            obs = b["observations"].to(device); act = b["actions"].to(device)
            z_ctrl, _ = _states(obs)
            preds, (loss, rloss, _, _, ploss) = jepa.unroll(
                obs, act, nsteps=1, unroll_mode="autoregressive", compute_loss=True)
            pred = preds[:, :, -1, 0, 0]
            l_sig = sig(pred - z_ctrl, b["drug"].to(device))
            l_path = path(pred, b["pathway"].to(device))
            total = loss + ls * l_sig + lp * l_path
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0); opt.step()
            n = obs.shape[0]
            agg["pred"] += (ploss.item() if torch.is_tensor(ploss) else float(ploss)) * n
            agg["sig"] += l_sig.item() * n; agg["path"] += l_path.item() * n; agg["n"] += n
        sched.step()

        n = max(1, agg["n"])
        row = {
            "epoch": ep,
            "train_pred": agg["pred"] / n,
            "train_sig": agg["sig"] / n,
            "train_path": agg["path"] / n,
            "lr": sched.get_last_lr()[0],
        }
        msg = (f"[ep {ep:03d}/{cfg.optim.epochs}] "
               f"pred={row['train_pred']:.4f} sig={row['train_sig']:.4f} path={row['train_path']:.4f}")

        if ep % cfg.optim.probe_every == 0 or ep == cfg.optim.epochs:
            m = evaluate(jepa, val_loader, device, train_shift)
            row.update({"val_pred": m["pred"], "skill_vs_identity": m["skill_vs_identity"],
                        "skill_vs_meanshift": m["skill_vs_meanshift"]})
            star = ""
            if m["skill_vs_meanshift"] > best["skill_vs_meanshift"]:
                best = {**m, "epoch": ep}; star = " *BEST*"
                torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg), "epoch": ep},
                           os.path.join(ckpt_dir, "tahoe_perturb_best.pt"))
            msg += (f" || val pred={m['pred']:.4f} "
                    f"skill_vs_noeffect={m['skill_vs_identity']:.3f}x "
                    f"skill_vs_meanshift={m['skill_vs_meanshift']:.3f}x{star}")

        print(msg, flush=True)
        curves.append(row)
        with open(curves_json, "w") as f:
            json.dump(curves, f, indent=2)

    torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg),
                "epoch": cfg.optim.epochs},
               os.path.join(ckpt_dir, "tahoe_perturb.pt"))
    print(f"saved final -> {os.path.join(ckpt_dir, 'tahoe_perturb.pt')}")

    save_curves_plot(curves, curves_png)

    print(f"\n== BEST (ep {best.get('epoch', '?')}): "
          f"skill_vs_noeffect={best.get('skill_vs_identity', float('nan')):.3f}x "
          f"skill_vs_meanshift={best['skill_vs_meanshift']:.3f}x ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/perturb.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
