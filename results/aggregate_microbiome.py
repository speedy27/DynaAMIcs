"""Aggregate microbiome metrics across seeds per condition."""
import json, glob, os, statistics as S
from collections import defaultdict

agg = defaultdict(lambda: defaultdict(list))
files = glob.glob(os.path.join("microbiome", "*", "seed*", "metrics.json"))
for f in sorted(files):
    parts = f.replace(os.sep, "/").split("/")
    cond = parts[1]
    with open(f) as fh:
        m = json.load(fh)
    for k, v in m["metrics"].items():
        if isinstance(v, (int, float)):
            agg[cond][k].append(v)

KEYS = ["skill_vs_identity", "t1d_auroc", "age_r2", "effrank", "div", "phylo", "tvar"]
print(f"{'condition':12s} | " + " | ".join(f"{k:>20s}" for k in KEYS))
print("-" * (14 + 23 * len(KEYS)))
for cond in sorted(agg):
    row = []
    for k in KEYS:
        vs = agg[cond].get(k, [])
        if not vs:
            row.append("        n/a        ")
        else:
            mu = sum(vs) / len(vs)
            sd = S.pstdev(vs) if len(vs) > 1 else 0.0
            row.append(f"{mu:7.3f} +/- {sd:5.3f} n{len(vs)}")
    print(f"{cond:12s} | " + " | ".join(row))
