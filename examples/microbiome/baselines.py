"""
Baselines for Microbiome-JEPA -- the controlled comparison every representation
paper needs. We score a *ladder* of community representations on the SAME
subject-disjoint linear probes (host-age regression + T1D classification), so the
only thing that changes is the representation:

  REPRESENTATION ladder (increasing sophistication):
  1. diversity       Shannon alpha-diversity summary stats over the window
                     (mean/std/min/max) -> the simplest ecological descriptor.
  2. rank-abundance  window-averaged sorted log-abundance curve (community
                     evenness/dominance), no sequence identity, NO learning.
  3. raw             abundance-weighted mean ProkBERT descriptor (NO learning)
                     -> "does the JEPA beat trivial sequence+abundance features?"
  4. random-encoder  an UNTRAINED SetEncoder, same architecture, random weights
                     (averaged over a few seeds) -> "is it the training that
                     helps, or just the architecture / set-pooling inductive bias?"
  5. mlp-supervised  a small MLP fit END-TO-END on raw mean-ProkBERT (the classic
                     'MLP baseline to beat'); a nonlinear supervised reference.
  6. jepa            the trained encoder(s); pass several checkpoints to get a
                     mean +/- std across seeds (no more single-run claims).

Every row goes through one identical pipeline: subject-pool the representation
over the window, standardize on TRAIN, fit Ridge (age) + balanced
LogisticRegression (T1D) on TRAIN, score on VAL. Subjects in train/val are
disjoint (dataset split), so there is no leakage.

With one or more --ckpt, we ALSO print a WORLD-MODEL dynamics table: is the
trained predictor better than trivial persistence / a global mean-shift / a
linear AR model (all in latent space)?  -> answers "does the predictor capture
real community dynamics, not just the unbeatable no-change baseline?".

  # non-learned + random + supervised baselines only (no trained model needed):
  python -m examples.microbiome.baselines

  # add the trained JEPA column + the dynamics table from a checkpoint:
  python -m examples.microbiome.baselines --ckpt checkpoints/microbiome/microbiome_jepa.pt

  # error bars over seeds: pass several checkpoints:
  python -m examples.microbiome.baselines --ckpt run1/microbiome_jepa.pt run2/... run3/...
"""

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.architectures import SetEncoder
from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders


def _gather_raw(loader):
    """Subject-pooled RAW descriptor: the abundance-weighted mean ProkBERT
    embedding of the community, averaged over the window. No model involved."""
    X, age, lab = [], [], []
    for b in loader:
        X.append(b["phylo"].mean(dim=1).numpy())  # [B, T, E] -> [B, E]
        age.append(b["age"].numpy())
        lab.append(b["label"].numpy())
    return np.concatenate(X), np.concatenate(age), np.concatenate(lab).astype(int)


def _gather_diversity(loader):
    """Classic ecology baseline #1: Shannon alpha-diversity summary stats over the
    window (mean / std / min / max). The very first descriptor a microbiologist
    would try -- no sequences, no learning."""
    X, age, lab = [], [], []
    for b in loader:
        d = b["diversity"]  # [B, T]
        feat = torch.stack([d.mean(1), d.std(1), d.amin(1), d.amax(1)], dim=1)  # [B, 4]
        X.append(feat.numpy())
        age.append(b["age"].numpy())
        lab.append(b["label"].numpy())
    return np.concatenate(X), np.concatenate(age), np.concatenate(lab).astype(int)


def _gather_rank_abundance(loader, emb_dim):
    """Classic ecology baseline #2: the window-averaged rank-abundance curve --
    the sorted log-abundance profile over the OTU slots. Captures community
    evenness / dominance without sequence identity or learning."""
    X, age, lab = [], [], []
    for b in loader:
        logab = b["observations"][:, emb_dim, :, :, 0]               # [B, T, N] log-abundance
        ranked = torch.sort(logab, dim=-1, descending=True).values   # rank-abundance curve
        X.append(ranked.mean(1).numpy())                             # [B, N] mean over window
        age.append(b["age"].numpy())
        lab.append(b["label"].numpy())
    return np.concatenate(X), np.concatenate(age), np.concatenate(lab).astype(int)


@torch.no_grad()
def _gather_encoder(encoder, loader, device):
    """Subject-pooled latent from an encoder (trained or random), pooled over the
    window exactly like main.py's probe (state.mean over T)."""
    X, age, lab = [], [], []
    for b in loader:
        state = encoder(b["observations"].to(device))  # [B, D, T, 1, 1]
        X.append(state.mean(dim=2)[..., 0, 0].cpu().numpy())  # [B, D]
        age.append(b["age"].numpy())
        lab.append(b["label"].numpy())
    return np.concatenate(X), np.concatenate(age), np.concatenate(lab).astype(int)


def _probe(Xt, at, yt, Xv, av, yv):
    """One standardized linear probe. Fit on train, score on val."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xt)
    Xt, Xv = sc.transform(Xt), sc.transform(Xv)
    out = {}
    out["age_r2"] = float(r2_score(av, Ridge(alpha=1.0).fit(Xt, at).predict(Xv)))
    if len(np.unique(yt)) == 2 and len(np.unique(yv)) == 2:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt, yt)
        out["t1d_auroc"] = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
    else:
        out["t1d_auroc"] = float("nan")
    return out


def _probe_mlp(Xt, at, yt, Xv, av, yv, seed=0):
    """Supervised nonlinear reference -- the classic 'MLP baseline to beat'. Same
    standardized train/val split as `_probe`, but a small MLP fit end-to-end."""
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xt)
    Xt, Xv = sc.transform(Xt), sc.transform(Xv)
    out = {}
    reg = MLPRegressor(hidden_layer_sizes=(128,), max_iter=500, random_state=seed)
    out["age_r2"] = float(r2_score(av, reg.fit(Xt, at).predict(Xv)))
    if len(np.unique(yt)) == 2 and len(np.unique(yv)) == 2:
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=seed)
        out["t1d_auroc"] = float(roc_auc_score(yv, clf.fit(Xt, yt).predict_proba(Xv)[:, 1]))
    else:
        out["t1d_auroc"] = float("nan")
    return out


@torch.no_grad()
def _gather_transitions(jepa, loader, cfg, device):
    """Collect latent transitions (z_t, z_{t+1}, a_t) + the JEPA predictor next-state
    MSE in TWO modes, so the comparison with the 1-step baselines is honest:
      - rollout : K-step autoregressive (feeds its OWN predictions; planning-relevant,
                  what main.py's skill uses) -> error accumulates over the horizon.
      - 1-step  : teacher-forced single step from the TRUE z_t -> apples-to-apples with
                  identity / mean-shift / linear-AR (all 1-step from the true state).
    """
    Zt, Ztp1, At = [], [], []
    sse_ar, sse_tf, n_elt = 0.0, 0.0, 0
    for b in loader:
        obs = b["observations"].to(device)
        act = b["actions"].to(device)
        state = jepa.encoder(obs)  # [B, D, T, 1, 1]
        preds, _ = jepa.unroll(obs, act, nsteps=cfg.model.nsteps,
                               unroll_mode="autoregressive", compute_loss=False,
                               return_all_steps=False)
        Tn = min(state.shape[2], preds.shape[2])
        s = state[..., 0, 0]   # [B, D, T]
        p = preds[..., 0, 0]   # [B, D, T]
        D = s.shape[1]
        zt = s[:, :, : Tn - 1].permute(0, 2, 1).reshape(-1, D)     # z_t
        ztp1 = s[:, :, 1:Tn].permute(0, 2, 1).reshape(-1, D)       # z_{t+1} (target)
        at = act[:, :, : Tn - 1].permute(0, 2, 1).reshape(-1, act.shape[1])  # a_t
        pjep = p[:, :, 1:Tn].permute(0, 2, 1).reshape(-1, D)       # rollout prediction
        # teacher-forced 1-step: the RNN predictor is single-step (state = GRU hidden,
        # action = GRU input), so fold time into the batch and predict z_{t+1} from the
        # TRUE z_t (apples-to-apples with the 1-step baselines).
        Tm1 = Tn - 1
        st = state[:, :, :Tm1]                                     # [B, D, Tm1, 1, 1]
        ac = act[:, :, :Tm1]                                       # [B, A, Tm1]
        Bsz = st.shape[0]
        st_f = st.permute(0, 2, 1, 3, 4).reshape(Bsz * Tm1, D, 1, 1, 1)
        ac_f = ac.permute(0, 2, 1).reshape(Bsz * Tm1, ac.shape[1], 1)
        ptf = jepa.predictor(st_f, jepa.action_encoder(ac_f))[..., 0, 0, 0]  # [B*Tm1, D]
        Zt.append(zt.cpu().numpy()); Ztp1.append(ztp1.cpu().numpy()); At.append(at.cpu().numpy())
        sse_ar += float(((pjep - ztp1) ** 2).sum().cpu())
        sse_tf += float(((ptf - ztp1) ** 2).sum().cpu())
        n_elt += ztp1.numel()
    return (np.concatenate(Zt), np.concatenate(Ztp1), np.concatenate(At),
            sse_ar / max(1, n_elt), sse_tf / max(1, n_elt))


def _dynamics_table(jepa, train_loader, val_loader, cfg, device):
    """World-model baselines in latent space: is the trained predictor better than
    trivial persistence / a global mean-shift / a linear AR model?  Microbiome
    trajectories are slow, so the no-change baseline is strong -- beating mean-shift
    and linear AR is what actually proves the predictor learned dynamics."""
    from sklearn.linear_model import Ridge

    Zt_tr, Ztp1_tr, At_tr, _, _ = _gather_transitions(jepa, train_loader, cfg, device)
    Zt, Ztp1, At, jepa_ar, jepa_tf = _gather_transitions(jepa, val_loader, cfg, device)

    ident_mse = float(np.mean((Zt - Ztp1) ** 2))                  # no-change (persistence)
    delta = (Ztp1_tr - Zt_tr).mean(0, keepdims=True)              # global mean-shift (train)
    mshift_mse = float(np.mean((Zt + delta - Ztp1) ** 2))
    lin = Ridge(alpha=1.0).fit(np.concatenate([Zt_tr, At_tr], 1), Ztp1_tr)  # linear AR W[z,a]
    lin_mse = float(np.mean((lin.predict(np.concatenate([Zt, At], 1)) - Ztp1) ** 2))

    nk = int(cfg.model.nsteps)
    rows = [
        ("identity (no-change)", ident_mse, "1-step"),
        ("global mean-shift", mshift_mse, "1-step"),
        ("linear AR  W[z,a]", lin_mse, "1-step"),
        ("JEPA 1-step (TF)", jepa_tf, "1-step"),
        ("JEPA rollout", jepa_ar, f"{nk}-step"),
    ]
    print("\n== Microbiome world-model dynamics baselines (latent next-state MSE) ==")
    print(f"{'dynamics model':<24}{'horizon':>9}{'latent_mse':>13}{'skill_vs_ident':>16}")
    print("-" * 62)
    for name, mse, hz in rows:
        print(f"{name:<24}{hz:>9}{mse:>13.3e}{ident_mse / max(1e-9, mse):>16.3f}")
    print("\nRead: skill = identity_MSE / model_MSE (>1 beats no-change). Compare the 1-step "
          "rows apples-to-apples (vs linear AR); the rollout row shows multi-step error growth.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="trained JEPA checkpoint(s); >1 -> mean+/-std JEPA row + "
                         "dynamics table built from the first checkpoint")
    ap.add_argument("--cfg", default="examples/microbiome/cfgs/train.yaml",
                    help="config used when no --ckpt is given")
    ap.add_argument("--rand-seeds", type=int, nargs="+", default=[1, 1000, 10000],
                    help="seeds for the untrained random-encoder baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = args.ckpt or []
    if ckpts:
        blob0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(blob0["cfg"])
    else:
        cfg = OmegaConf.load(args.cfg)

    dcfg = MicrobiomeConfig(
        cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
        tp_stride=cfg.data.get("tp_stride", 1),
        n_max=cfg.data.n_max, emb_dim=cfg.model.emb_dim,
        val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed,
    )
    _, _, train_loader, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)

    rows = {}  # name -> {age_r2, t1d_auroc} (scalars) or (mean, std) tuples

    # ---- 1. classic ecology: Shannon diversity stats (no learning) --------
    Xt, at, yt = _gather_diversity(train_loader)
    Xv, av, yv = _gather_diversity(val_loader)
    rows["diversity (Shannon)"] = _probe(Xt, at, yt, Xv, av, yv)

    # ---- 2. classic ecology: rank-abundance curve (no learning) -----------
    Xt, at, yt = _gather_rank_abundance(train_loader, cfg.model.emb_dim)
    Xv, av, yv = _gather_rank_abundance(val_loader, cfg.model.emb_dim)
    rows["rank-abundance"] = _probe(Xt, at, yt, Xv, av, yv)

    # ---- 3. raw mean-ProkBERT (no learning) -------------------------------
    Xt, at, yt = _gather_raw(train_loader)
    Xv, av, yv = _gather_raw(val_loader)
    rows["raw (mean ProkBERT)"] = _probe(Xt, at, yt, Xv, av, yv)

    # ---- 4. supervised MLP on raw mean-ProkBERT (the 'MLP to beat') -------
    rows["MLP on raw (supervised)"] = _probe_mlp(Xt, at, yt, Xv, av, yv)

    # ---- 5. random untrained encoder (averaged over seeds) ----------------
    rand = {"age_r2": [], "t1d_auroc": []}
    for s in args.rand_seeds:
        torch.manual_seed(s)
        enc = SetEncoder(emb_dim=cfg.model.emb_dim, h_d=cfg.model.henc,
                         out_d=cfg.model.dstc).to(device).eval()
        Xt, at, yt = _gather_encoder(enc, train_loader, device)
        Xv, av, yv = _gather_encoder(enc, val_loader, device)
        r = _probe(Xt, at, yt, Xv, av, yv)
        rand["age_r2"].append(r["age_r2"])
        rand["t1d_auroc"].append(r["t1d_auroc"])
    rows[f"random encoder (n={len(args.rand_seeds)})"] = {
        "age_r2": (float(np.mean(rand["age_r2"])), float(np.std(rand["age_r2"]))),
        "t1d_auroc": (float(np.nanmean(rand["t1d_auroc"])), float(np.nanstd(rand["t1d_auroc"]))),
    }

    # ---- 6. trained JEPA encoder(s) (optional; >1 -> mean+/-std) ----------
    first_jepa = None
    if ckpts:
        from examples.microbiome.main import build_jepa
        jepa_scores = {"age_r2": [], "t1d_auroc": []}
        for cpath in ckpts:
            blob = torch.load(cpath, map_location="cpu", weights_only=False)
            ccfg = OmegaConf.create(blob["cfg"])
            jepa = build_jepa(ccfg, train_loader.dataset.action_dim, device)
            jepa.load_state_dict(blob["jepa"])
            jepa.eval()
            if first_jepa is None:
                first_jepa = jepa
            Xt, at, yt = _gather_encoder(jepa.encoder, train_loader, device)
            Xv, av, yv = _gather_encoder(jepa.encoder, val_loader, device)
            r = _probe(Xt, at, yt, Xv, av, yv)
            jepa_scores["age_r2"].append(r["age_r2"])
            jepa_scores["t1d_auroc"].append(r["t1d_auroc"])
        if len(ckpts) > 1:
            rows[f"JEPA trained (n={len(ckpts)})"] = {
                "age_r2": (float(np.mean(jepa_scores["age_r2"])), float(np.std(jepa_scores["age_r2"]))),
                "t1d_auroc": (float(np.nanmean(jepa_scores["t1d_auroc"])), float(np.nanstd(jepa_scores["t1d_auroc"]))),
            }
        else:
            rows["JEPA (trained)"] = {"age_r2": jepa_scores["age_r2"][0],
                                      "t1d_auroc": jepa_scores["t1d_auroc"][0]}

    # ---- print the representation comparison table ------------------------
    def _fmt(v):
        if isinstance(v, tuple):
            return f"{v[0]:.3f}+/-{v[1]:.3f}"
        return f"{v:.3f}"

    print("\n== Microbiome representation baselines "
          "(subject-disjoint standardized linear probe) ==")
    print(f"{'representation':<26}{'age_r2':>16}{'t1d_auroc':>16}")
    print("-" * 58)
    for name, m in rows.items():
        print(f"{name:<26}{_fmt(m['age_r2']):>16}{_fmt(m['t1d_auroc']):>16}")
    if not ckpts:
        print("\n(no --ckpt given: 'JEPA (trained)' row + dynamics table omitted. "
              "Pass --ckpt <path> [<path> ...] to add them.)")
    print("\nRead: age_r2 = host-age R^2 (higher=better, 0=no signal); "
          "t1d_auroc = T1D AUROC (0.5=chance).")

    # ---- world-model dynamics baselines (needs a trained predictor) -------
    if first_jepa is not None:
        _dynamics_table(first_jepa, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
