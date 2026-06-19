"""Generate the figures for the jury slide deck (saved next to main.tex)."""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
PRIM = "#0f2d50"; SEC = "#009688"; GREY = "#b0b8c4"

# ---- Fig 1: Tahoe perturbation world-model beats baselines (skill) ----
fig, ax = plt.subplots(figsize=(6.2, 4.0))
names = ["no-effect\nbaseline", "mean-shift\nbaseline", "EB-JEPA\n(ours)"]
vals = [1.0, 1.0, 1.199]
cols = [GREY, GREY, SEC]
b = ax.bar(names, vals, color=cols, edgecolor=PRIM, linewidth=1.5)
ax.axhline(1.0, ls="--", c=PRIM, lw=1)
ax.set_ylim(0.9, 1.28)
ax.set_ylabel("skill (baseline MSE / our MSE)  — higher is better")
ax.set_title("Drug-perturbation world model: control + drug → perturbed state", color=PRIM)
ax.bar_label(b, fmt="%.2fx", padding=3, color=PRIM, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_skill.png", dpi=160)

# ---- Fig 2: from-scratch representation learns identity, not drug response ----
fig, ax = plt.subplots(figsize=(6.2, 4.0))
tasks = ["cell line\n(50-way)", "drug\n(66-way)", "MoA"]
raw = [0.908, 0.138, 0.299]      # raw-expression linear probe
jepa = [0.926, 0.018, 0.058]     # from-scratch two-view JEPA (gene panel)
x = np.arange(len(tasks)); w = 0.38
ax.bar(x - w/2, raw, w, label="raw expression", color=GREY, edgecolor=PRIM)
ax.bar(x + w/2, jepa, w, label="from-scratch JEPA", color=PRIM)
ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylabel("linear-probe macro-F1")
ax.set_title("Two-view SSL captures cell identity, NOT drug response", color=PRIM)
ax.legend()
fig.tight_layout(); fig.savefig(f"{HERE}/fig_probe.png", dpi=160)

# ---- Fig 3: microbiome — clock works, dynamics collapse ----
log = os.path.join(HERE, "..", "..", "..", "artifacts", "train_log.txt")
eps, age, tvar = [], [], []
if os.path.exists(log):
    for ln in open(log):
        m = re.search(r"\[ep\s+(\d+)\]", ln)
        if not m: continue
        a = re.search(r"age_r2=([0-9.]+)", ln); t = re.search(r"tvar=([0-9.]+)", ln)
        eps.append(int(m.group(1))); age.append(float(a.group(1)) if a else np.nan)
        tvar.append(float(t.group(1)) if t else np.nan)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
if eps:
    ax.plot(eps, age, "-o", c=SEC, label="age R² (microbiome clock)")
    ax.plot(eps, tvar, "-o", c="crimson", label="temporal variance (tvar)")
ax.axhline(0, c="gray", lw=.5)
ax.set_xlabel("epoch"); ax.set_ylabel("score")
ax.set_title("Microbiome: representation works, world model collapses", color=PRIM)
ax.legend()
fig.tight_layout(); fig.savefig(f"{HERE}/fig_microbiome.png", dpi=160)
print("figures written to", HERE)
