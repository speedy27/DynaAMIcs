"""
Representation-collapse ablation (the jury's favorite figure): train the SAME
two-view cell JEPA on PBMC3k with vs without the SIGReg anti-collapse term, and
track the embedding's per-dimension std + effective rank over training, plus the
final linear-probe accuracy. Without anti-collapse the encoder collapses (std->0,
rank->1, probe->chance); SIGReg keeps it healthy.

  python examples/tahoe/slides/collapse_demo.py --cache artifacts/pbmc/cache.pt --out examples/tahoe/slides
"""
import argparse, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from eb_jepa.architectures import Projector
from eb_jepa.losses import BCS

PRIM = "#0f2d50"; SEC = "#009688"


def eff_rank(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / (s.sum() + 1e-9)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def run_one(X, y, device, mode, epochs=40, d=128, bs=256, noise=0.3, drop=0.3):
    K = X.shape[1]
    enc = nn.Sequential(nn.Linear(K, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Linear(512, d)).to(device)
    proj = Projector(f"{d}-{4*d}-{4*d}").to(device)
    reg = BCS(num_slices=256, lmbd=10.0).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, device=device)
    stds, ranks = [], []
    for ep in range(epochs):
        enc.train(); perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i+bs]]
            if xb.shape[0] < 8: continue
            m1 = (torch.rand_like(xb) > drop).float(); m2 = (torch.rand_like(xb) > drop).float()
            v1 = xb * m1 * (1 + noise*torch.randn_like(xb)); v2 = xb * m2 * (1 + noise*torch.randn_like(xb))
            z1, z2 = enc(v1), enc(v2)
            if mode == "sigreg":
                loss = reg(proj(z1), proj(z2))["loss"]              # invariance + SIGReg anti-collapse
            else:
                loss = F.mse_loss(z1, z2)                            # invariance ONLY -> collapses
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval()
        with torch.no_grad():
            Z = enc(Xt).cpu().numpy()
        stds.append(float(Z.std(0).mean())); ranks.append(eff_rank(Z))
    # final probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); nv = len(X)//5
    va, tr = idx[:nv], idx[nv:]
    Z = enc(Xt).detach().cpu().numpy()
    acc = accuracy_score(y[va], LogisticRegression(max_iter=300).fit(Z[tr], y[tr]).predict(Z[va]))
    return stds, ranks, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/pbmc/cache.pt")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = torch.load(args.cache, weights_only=False)
    X = b["X"].float().numpy(); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = b["cell_line"].numpy()

    res = {}
    for mode in ["sigreg", "none"]:
        print("running", mode)
        res[mode] = run_one(X, y, device, mode)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for mode, c, lab in [("sigreg", SEC, "with SIGReg"), ("none", "crimson", "no anti-collapse")]:
        ax[0].plot(res[mode][0], c=c, label=lab); ax[1].plot(res[mode][1], c=c, label=lab)
    ax[0].set_title("embedding per-dim std", color=PRIM); ax[0].set_xlabel("epoch"); ax[0].axhline(0, c="gray", lw=.5); ax[0].legend()
    ax[1].set_title("effective rank of embedding", color=PRIM); ax[1].set_xlabel("epoch"); ax[1].legend()
    accs = [res["sigreg"][2], res["none"][2]]
    ax[2].bar(["with\nSIGReg", "no anti-\ncollapse"], accs, color=[SEC, "crimson"])
    ax[2].axhline(1/len(set(y)), ls="--", c="gray", label="chance"); ax[2].set_ylim(0, 1)
    ax[2].set_title("PBMC3k cell-type probe accuracy", color=PRIM); ax[2].bar_label(ax[2].containers[0], fmt="%.2f")
    fig.suptitle("Representation collapse: SIGReg keeps the encoder alive", color=PRIM, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "collapse.png"), dpi=150)
    print("saved collapse.png | std:", round(res["sigreg"][0][-1],3), "vs", round(res["none"][0][-1],3),
          "| acc:", round(accs[0],3), "vs", round(accs[1],3))


if __name__ == "__main__":
    main()
