"""
Tahoe perturbation WORLD MODEL — a true EB-JEPA with a frozen pretrained encoder.

  encoder f_θ   = FROZEN MosaicFM-3B embedding (identity over precomputed vectors)
  predictor g_φ = RNNPredictor, ACTION-conditioned on the drug (Morgan fingerprint)
  energy        = || g_φ(z_control, drug) − z_perturbed ||²   via eb_jepa.JEPA.unroll
  biology losses: PerturbationSignatureLoss (drug-signature consistency)
                  + PathwayCoherenceLoss   (gene-program / embedding-module structure)

Only the predictor is trained (the encoder is frozen) — exactly the "frozen
pretrained encoder + learned predictor" world-model setting.

Split: val_fraction=0.2 -> train 80%; then val 50/50 -> val 10%, test 10%.
Pred MSE + sig + pathway losses logged every epoch on train, val, test.
Skill metrics (vs no-effect, vs mean-shift) logged every probe_every epochs on val and test.
Early stopping on skill_vs_meanshift (patience_probes consecutive probe evaluations).

Training curves saved to:
  <ckpt>/tahoe_perturb_curves.json   (incremental)
  <ckpt>/tahoe_perturb_curves.png    (train/val/test losses + skill curves)

  python -m examples.tahoe.perturb --fname examples/tahoe/cfgs/perturb.yaml
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
    """No anti-collapse: encoder is frozen."""
    def forward(self, state, actions=None):
        return state.sum() * 0.0, state.sum() * 0.0, {}


def _states(obs):
    return obs[:, :, 0, 0, 0], obs[:, :, 1, 0, 0]


@torch.no_grad()
def compute_losses(jepa, sig_fn, path_fn, ls, lp, loader, device):
    """Pred MSE + signature + pathway losses on a dataloader — no gradient."""
    jepa.eval()
    agg = {"pred": 0.0, "sig": 0.0, "path": 0.0, "n": 0}
    for b in loader:
        obs = b["observations"].to(device); act = b["actions"].to(device)
        z_ctrl, _ = _states(obs)
        preds, (_, _, _, _, ploss) = jepa.unroll(
            obs, act, nsteps=1, unroll_mode="autoregressive", compute_loss=True)
        pred = preds[:, :, -1, 0, 0]
        l_sig = sig_fn(pred - z_ctrl, b["drug"].to(device))
        l_path = path_fn(pred, b["pathway"].to(device))
        n = obs.shape[0]
        agg["pred"] += (ploss.item() if torch.is_tensor(ploss) else float(ploss)) * n
        agg["sig"] += l_sig.item() * n; agg["path"] += l_path.item() * n; agg["n"] += n
    n = max(1, agg["n"])
    return {k: agg[k] / n for k in ("pred", "sig", "path")}


@torch.no_grad()
def evaluate(jepa, loader, device, train_shift=None):
    """Prediction MSE + skill metrics on a dataloader."""
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

        PRIM = "#0f2d50"; VAL_C = "#009688"; TEST_C = "#e67e22"
        epochs = [c["epoch"] for c in curves]
        val_epochs = [c["epoch"] for c in curves if "val_skill_vs_identity" in c]
        has_val = len(val_epochs) > 0

        fig, axes = plt.subplots(1, 3 if has_val else 2, figsize=(15 if has_val else 10, 4.5))

        # panel 1: pred MSE on train / val / test
        ax = axes[0]
        ax.semilogy(epochs, [c["train_pred"] for c in curves], color=PRIM, lw=1.5, label="train")
        if any("val_pred" in c for c in curves):
            ax.semilogy(epochs, [c.get("val_pred", float("nan")) for c in curves],
                        color=VAL_C, lw=1.5, ls="--", label="val")
        if any("test_pred" in c for c in curves):
            ax.semilogy(epochs, [c.get("test_pred", float("nan")) for c in curves],
                        color=TEST_C, lw=1.5, ls=":", label="test")
        ax.set_xlabel("epoch"); ax.set_ylabel("pred MSE (log)")
        ax.set_title("Prediction MSE train/val/test", color=PRIM)
        ax.legend(); ax.grid(True, alpha=0.3)

        # panel 2: sig + pathway losses on train / val / test
        ax = axes[1]
        ax.semilogy(epochs, [c["train_sig"] for c in curves], color=PRIM, lw=1.5,
                    ls="-", label="sig (train)")
        ax.semilogy(epochs, [c["train_path"] for c in curves], color=PRIM, lw=1.5,
                    ls="--", label="path (train)")
        if any("val_sig" in c for c in curves):
            ax.semilogy(epochs, [c.get("val_sig", float("nan")) for c in curves],
                        color=VAL_C, lw=1.2, ls="-", label="sig (val)")
            ax.semilogy(epochs, [c.get("val_path", float("nan")) for c in curves],
                        color=VAL_C, lw=1.2, ls="--", label="path (val)")
        if any("test_sig" in c for c in curves):
            ax.semilogy(epochs, [c.get("test_sig", float("nan")) for c in curves],
                        color=TEST_C, lw=1.2, ls="-", label="sig (test)")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
        ax.set_title("Sig + pathway losses", color=PRIM)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # panel 3: skill curves on val and test
        if has_val:
            ax = axes[2]
            for (label, key, color, ls) in [
                ("val skill vs no-effect", "val_skill_vs_identity", VAL_C, "-"),
                ("val skill vs mean-shift", "val_skill_vs_meanshift", VAL_C, "--"),
                ("test skill vs no-effect", "test_skill_vs_identity", TEST_C, "-"),
                ("test skill vs mean-shift", "test_skill_vs_meanshift", TEST_C, "--"),
            ]:
                vals = [c[key] for c in curves if key in c and not (isinstance(c[key], float) and c[key] != c[key])]
                ep_sub = [c["epoch"] for c in curves if key in c and not (isinstance(c[key], float) and c[key] != c[key])]
                if vals:
                    ax.plot(ep_sub, vals, color=color, ls=ls, lw=1.5, marker="o",
                            markersize=3, label=label)
            ax.axhline(1.0, ls=":", color="gray", lw=1, label="baseline = 1×")
            stopped = next((c for c in reversed(curves) if c.get("early_stop")), None)
            if stopped:
                ax.axvline(stopped["epoch"], ls="--", color="crimson", lw=1.2,
                           label=f"early stop (ep {stopped['epoch']})")
            ax.set_xlabel("epoch"); ax.set_ylabel("skill (ratio)")
            ax.set_title("Val & test skill (higher = better)", color=PRIM)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

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
    tr, va, train_loader, _ = make_loaders(dcfg, batch_size=cfg.data.batch_size,
                                           num_workers=cfg.data.num_workers)
    D, A = tr.D, tr.action_dim
    patience_probes = int(cfg.optim.get("patience_probes", 5))

    # split val 50/50 -> val 10% and test 10% of total
    n_half = len(va) // 2
    val_loader = DataLoader(Subset(va, range(n_half)), batch_size=cfg.data.batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(Subset(va, range(n_half, len(va))), batch_size=cfg.data.batch_size,
                             shuffle=False, drop_last=False)

    print(f"== tahoe-perturb | device={device} | D={D} action_dim={A} | "
          f"train={len(tr)} val={n_half} test={len(va)-n_half} | "
          f"epochs={cfg.optim.epochs} patience_probes={patience_probes} "
          f"| control={'DMSO' if tr.has_ctrl else 'centroid'} ==")

    encoder = FrozenIdentityEncoder()
    n_layers = int(cfg.model.get("layers", 1)) if "model" in cfg else 1
    predictor = RNNPredictor(hidden_size=D, action_dim=A, num_layers=n_layers, final_ln=nn.LayerNorm(D))
    jepa = JEPA(encoder, nn.Identity(), predictor, NoReg(), SquareLossSeq()).to(device)
    sig_fn = PerturbationSignatureLoss().to(device)
    path_fn = PathwayCoherenceLoss().to(device)
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
    probes_since_best = 0
    t0 = time.time()

    for ep in range(1, cfg.optim.epochs + 1):
        jepa.train()
        ep_t0 = time.time()
        agg = {"pred": 0.0, "sig": 0.0, "path": 0.0, "n": 0}
        for b in train_loader:
            obs = b["observations"].to(device); act = b["actions"].to(device)
            z_ctrl, _ = _states(obs)
            preds, (loss, _, _, _, ploss) = jepa.unroll(
                obs, act, nsteps=1, unroll_mode="autoregressive", compute_loss=True)
            pred = preds[:, :, -1, 0, 0]
            l_sig = sig_fn(pred - z_ctrl, b["drug"].to(device))
            l_path = path_fn(pred, b["pathway"].to(device))
            total = loss + ls * l_sig + lp * l_path
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0); opt.step()
            n = obs.shape[0]
            agg["pred"] += (ploss.item() if torch.is_tensor(ploss) else float(ploss)) * n
            agg["sig"] += l_sig.item() * n; agg["path"] += l_path.item() * n; agg["n"] += n
        sched.step()

        # val and test losses every epoch (forward-only, fast)
        val_m = compute_losses(jepa, sig_fn, path_fn, ls, lp, val_loader, device)
        test_m = compute_losses(jepa, sig_fn, path_fn, ls, lp, test_loader, device)
        jepa.train()

        n = max(1, agg["n"])
        ep_time = time.time() - ep_t0
        elapsed = time.time() - t0
        eta_s = (elapsed / ep) * (cfg.optim.epochs - ep)
        row = {
            "epoch": ep,
            "train_pred": agg["pred"] / n, "train_sig": agg["sig"] / n, "train_path": agg["path"] / n,
            "val_pred": val_m["pred"],     "val_sig": val_m["sig"],     "val_path": val_m["path"],
            "test_pred": test_m["pred"],   "test_sig": test_m["sig"],   "test_path": test_m["path"],
            "lr": sched.get_last_lr()[0],
            "epoch_time_s": round(ep_time, 2),
        }
        msg = (f"[ep {ep:03d}/{cfg.optim.epochs}] "
               f"pred tr={row['train_pred']:.4f} va={row['val_pred']:.4f} te={row['test_pred']:.4f} | "
               f"sig tr={row['train_sig']:.4f} va={row['val_sig']:.4f} | "
               f"t={ep_time:.1f}s eta={eta_s/60:.0f}m")

        do_probe = (ep % cfg.optim.probe_every == 0 or ep == cfg.optim.epochs)
        if do_probe:
            # skill metrics on val and test
            vm = evaluate(jepa, val_loader, device, train_shift)
            tm = evaluate(jepa, test_loader, device, train_shift)
            row.update({
                "val_skill_vs_identity": vm["skill_vs_identity"],
                "val_skill_vs_meanshift": vm["skill_vs_meanshift"],
                "test_skill_vs_identity": tm["skill_vs_identity"],
                "test_skill_vs_meanshift": tm["skill_vs_meanshift"],
            })
            msg += (f" || val skill_id={vm['skill_vs_identity']:.3f}x "
                    f"skill_ms={vm['skill_vs_meanshift']:.3f}x | "
                    f"test skill_id={tm['skill_vs_identity']:.3f}x "
                    f"skill_ms={tm['skill_vs_meanshift']:.3f}x")

            if vm["skill_vs_meanshift"] > best["skill_vs_meanshift"]:
                best = {**vm, "epoch": ep}; probes_since_best = 0; msg += " *BEST*"
                torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg), "epoch": ep},
                           os.path.join(ckpt_dir, "tahoe_perturb_best.pt"))
            else:
                probes_since_best += 1

        print(msg, flush=True)
        curves.append(row)
        with open(curves_json, "w") as f:
            json.dump(curves, f, indent=2)

        if do_probe and probes_since_best >= patience_probes:
            print(f"[early stop] val skill_vs_meanshift flat for {probes_since_best} probe evals "
                  f"({probes_since_best * cfg.optim.probe_every} epochs) — stopping at ep {ep}.",
                  flush=True)
            curves[-1]["early_stop"] = True
            with open(curves_json, "w") as f:
                json.dump(curves, f, indent=2)
            break

    torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg),
                "epoch": curves[-1]["epoch"]},
               os.path.join(ckpt_dir, "tahoe_perturb.pt"))
    total_time = time.time() - t0
    print(f"saved -> {os.path.join(ckpt_dir, 'tahoe_perturb.pt')} | total: {total_time/60:.1f} min")
    save_curves_plot(curves, curves_png)

    print(f"\n== BEST (ep {best.get('epoch', '?')}): "
          f"skill_vs_noeffect={best.get('skill_vs_identity', float('nan')):.3f}x "
          f"skill_vs_meanshift={best['skill_vs_meanshift']:.3f}x ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/perturb.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
