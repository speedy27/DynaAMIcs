"""
encoder_ablation.py - controlled MLP vs Set-Transformer encoder comparison for the
Tahoe cell-state JEPA, on the SAME seeds and the SAME linear-probe protocol.

Why this is a separate driver (and NOT inside experiments.py): the encoder choice
only matters on the REPRESENTATION side. The per-gene tokens a Set-Transformer needs
live in the gene-panel cache used by main.py. The perturbation world model in
experiments.py runs on a pooled MosaicFM embedding [N, 2560] and has no per-gene
encoder to ablate. So this driver wraps main.run() and compares:

  encoder = mlp             CellEncoder, a 2-layer MLP over the gene panel
  encoder = settransformer  per-gene tokens + Perceiver pooling (multi-source ready)

for every seed, reporting linear-probe macro-F1 per task (mean +/- std). It reuses
main.py's exact training + probe so the only thing that changes is the encoder.

  python -m examples.tahoe.encoder_ablation \
      --fname examples/tahoe/cfgs/train.yaml --epochs 30 --seeds 1 1000 10000 \
      data.cache_path=$WORK/tahoe/cache.pt

  # smoke (no real cache needed for the aggregation/figure code path):
  python -m examples.tahoe.encoder_ablation --selftest
"""
import argparse
import json
import os

import numpy as np


def aggregate(runs, feat="JEPA(ours)"):
    """runs: list of dicts from main.run(). Group macro-F1 by (encoder, task)."""
    by = {}
    for r in runs:
        enc = r["encoder"]
        for task, feats in r["metrics"].items():
            if feat in feats:
                by.setdefault(enc, {}).setdefault(task, []).append(feats[feat]["macro_f1"])
    out = {}
    for enc, tasks in by.items():
        out[enc] = {t: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v), "seeds": v}
                    for t, v in tasks.items()}
    return out


def make_figure(agg, path):
    """Grouped bar chart (tasks x encoders), mean +/- std, deck colour scheme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PRIM, SEC, GREY = "#0f2d50", "#009688", "#b0b8c4"
    encs = [e for e in ("mlp", "settransformer") if e in agg] or list(agg)
    if not encs:
        return
    tasks = list(agg[encs[0]].keys())
    x = np.arange(len(tasks)); w = 0.8 / max(1, len(encs))
    colors = {"mlp": GREY, "settransformer": SEC}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, enc in enumerate(encs):
        means = [agg[enc].get(t, {}).get("mean", 0.0) for t in tasks]
        stds = [agg[enc].get(t, {}).get("std", 0.0) for t in tasks]
        ax.bar(x + (i - (len(encs) - 1) / 2) * w, means, w, yerr=stds, capsize=3,
               label=enc, color=colors.get(enc, PRIM), edgecolor=PRIM, linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("linear-probe macro-F1  (mean +/- std)")
    ax.set_title("Encoder ablation: MLP vs Set-Transformer (frozen-feature probe)", color=PRIM)
    ax.legend()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160)
    print("figure ->", path)


def _print_table(agg):
    print("\n== encoder ablation (JEPA macro-F1, mean +/- std) ==")
    for enc, tasks in agg.items():
        for t, s in tasks.items():
            print(f"  {enc:15s} {t:10s} F1={s['mean']:.3f} +/- {s['std']:.3f}  (n={s['n']})")


def _selftest(out, fig):
    """Exercise aggregate + figure with synthetic run dicts (no cache/GPU needed)."""
    rng = np.random.default_rng(0)
    runs = []
    base = {"mlp": dict(cell_line=0.90, drug=0.14, moa=0.30),
            "settransformer": dict(cell_line=0.93, drug=0.20, moa=0.34)}
    for enc, b in base.items():
        for seed in (1, 1000, 10000):
            metrics = {t: {"JEPA(ours)": {"macro_f1": float(v + rng.normal(0, 0.01)), "acc": 0.0},
                           "raw": {"macro_f1": 0.0, "acc": 0.0}}
                       for t, v in b.items()}
            runs.append({"encoder": enc, "seed": seed, "genes": 2000, "reg": "sigreg",
                         "metrics": metrics})
    agg = aggregate(runs)
    os.makedirs(out, exist_ok=True)
    json.dump({"aggregate": agg, "runs": runs}, open(os.path.join(out, "results.json"), "w"), indent=2)
    _print_table(agg)
    make_figure(agg, fig)
    assert set(agg) == {"mlp", "settransformer"}
    assert agg["settransformer"]["cell_line"]["n"] == 3
    print("selftest OK ->", os.path.join(out, "results.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/train.yaml")
    ap.add_argument("--out", default="artifacts/tahoe/encoder_ablation")
    ap.add_argument("--fig", default=os.path.join(os.path.dirname(__file__), "slides", "encoder_ablation.png"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 1000, 10000])
    ap.add_argument("--encoders", nargs="+", default=["mlp", "settransformer"])
    ap.add_argument("--selftest", action="store_true",
                    help="run aggregation+figure on synthetic data (no cache/GPU)")
    args, overrides = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    if args.selftest:
        _selftest(args.out, args.fig)
        return

    from examples.tahoe.main import run  # imported lazily so --selftest needs no torch

    runs = []
    for enc in args.encoders:
        for seed in args.seeds:
            ov = list(overrides) + [
                f"model.encoder={enc}", f"meta.seed={seed}",
                f"optim.epochs={args.epochs}", "logging.log_wandb=false",
            ]
            print(f"\n========== encoder={enc} seed={seed} ==========")
            runs.append(run(args.fname, ov))
            json.dump(runs, open(os.path.join(args.out, "runs.json"), "w"), indent=2)

    agg = aggregate(runs)
    json.dump({"aggregate": agg, "runs": runs},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)
    _print_table(agg)
    make_figure(agg, args.fig)
    print("saved ->", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
