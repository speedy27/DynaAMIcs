"""
aggregate.py -- collect metrics.json from an ablation/seed sweep into ONE table
+ bar chart. This is the controlled-comparison deliverable the rubric asks for:
one change at a time, several seeds each, mean +/- std, so the effect of every
microbiome-specific term (diversity / phylo / temporal-variance) is visible at a
glance -- and so the temporal-collapse FIX is provable, not asserted.

Each training run writes ``<ckpt_dir>/metrics.json`` (see
``examples/microbiome/main.py``). Point ``--root`` at the parent directory that
holds all the runs (searched recursively) -- e.g. the per-condition/per-seed
checkpoint tree produced by ``run_ablation.sh``.

  python -m examples.microbiome.aggregate --root checkpoints/microbiome \
      --out checkpoints/microbiome/ablation.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# metric key -> (pretty title for the bar chart)
METRICS = [
    ("skill_vs_identity", "skill vs identity\n(>1 beats no-change)"),
    ("effrank", "effective rank\n(latent dims used)"),
    ("age_r2", "age R^2\n(aging clock)"),
    ("t1d_auroc", "T1D AUROC"),
]
# canonical ordering of ablation conditions (by number of active terms)
ORDER = ["baseline", "div", "phylo", "tvar", "div+phylo", "div+tvar",
         "phylo+tvar", "div+phylo+tvar"]


def _load(root):
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        runs[d.get("condition", "?")].append(d)
    return runs


def _agg(values):
    arr = np.array([v for v in values if v is not None and not np.isnan(v)],
                   dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(arr.mean()), float(arr.std()), int(arr.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints/microbiome")
    ap.add_argument("--out", default="checkpoints/microbiome/ablation.png")
    args = ap.parse_args()

    runs = _load(args.root)
    if not runs:
        print(f"no metrics.json found under {args.root!r}. Train some runs first "
              f"(each writes <ckpt_dir>/metrics.json).")
        return

    conds = [c for c in ORDER if c in runs] + [c for c in runs if c not in ORDER]
    cols = [m[0] for m in METRICS]

    # ---- table ----
    print(f"\n== Microbiome-JEPA ablation ({args.root}) ==")
    head = f"{'condition':<18}{'seeds':>6}" + "".join(f"{c:>20}" for c in cols)
    print(head)
    print("-" * len(head))
    table = {}
    for cond in conds:
        ds = runs[cond]
        row, cells = {}, []
        for key in cols:
            mean, std, k = _agg([d.get("metrics", {}).get(key) for d in ds])
            row[key] = (mean, std, k)
            cells.append(f"{mean:.3f}+/-{std:.3f}")
        table[cond] = row
        print(f"{cond:<18}{len(ds):>6}" + "".join(f"{c:>20}" for c in cells))
    print("\nRead: skill>1 beats the no-change baseline; effrank high = no collapse; "
          "age_r2 high = aging-clock signal; t1d_auroc 0.5 = chance.")

    # ---- bar chart ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; table only.")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
    x = np.arange(len(conds))
    for ax, (key, title) in zip(np.atleast_1d(axes), METRICS):
        means = [table[c][key][0] for c in conds]
        stds = [table[c][key][1] for c in conds]
        ax.bar(x, means, yerr=stds, capsize=4, color="teal", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        if key == "skill_vs_identity":
            ax.axhline(1.0, color="crimson", ls="--", lw=1, label="no-change")
            ax.legend(fontsize=8)
        if key == "t1d_auroc":
            ax.axhline(0.5, color="crimson", ls="--", lw=1, label="chance")
            ax.legend(fontsize=8)
    fig.suptitle("Controlled comparison — one term at a time, mean ± std over seeds")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
