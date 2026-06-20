"""Generate JEPA benchmark figures (CLR-RMSE, skill, gain, summary)."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
results = {
    "Persistence": {
        "clr_rmse": {
            "h1":  {"mean": 0.4578, "std": 0.0159},
            "h3":  {"mean": 0.7484, "std": 0.0197},
            "h5":  {"mean": 0.9207, "std": 0.0217},
            "h10": {"mean": 1.2444, "std": 0.0271},
        },
        "skill": {"h1": 1.000, "h3": 1.000, "h5": 1.000, "h10": 1.000},
    },
    "gLV-L2 (Ridge)": {
        "clr_rmse": {
            "h1":  {"mean": 0.3994, "std": 0.0179},
            "h3":  {"mean": 0.5293, "std": 0.0286},
            "h5":  {"mean": 0.5643, "std": 0.0349},
            "h10": {"mean": 0.6172, "std": 0.0391},
        },
        "skill": {"h1": 1.146, "h3": 1.414, "h5": 1.632, "h10": 2.016},
    },
    "gLV-net (MLP)": {
        "clr_rmse": {
            "h1":  {"mean": 0.2960, "std": 0.0124},
            "h3":  {"mean": 0.4137, "std": 0.0182},
            "h5":  {"mean": 0.4679, "std": 0.0207},
            "h10": {"mean": 0.5650, "std": 0.0230},
        },
        "skill": {"h1": 1.546, "h3": 1.809, "h5": 1.968, "h10": 2.203},
    },
    "JEPA (ours)": {
        "clr_rmse": {
            "h1":  {"mean": 0.221, "std": 0.018},
            "h3":  {"mean": 0.308, "std": 0.025},
            "h5":  {"mean": 0.362, "std": 0.031},
            "h10": {"mean": 0.468, "std": 0.041},
        },
        "skill": {"h1": 2.071, "h3": 2.430, "h5": 2.543, "h10": 2.660},
    },
}

HORIZONS = [1, 3, 5, 10]
HKEYS    = ["h1", "h3", "h5", "h10"]

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "Persistence":      "#9e9e9e",
    "gLV-L2 (Ridge)":  "#f4a261",
    "gLV-net (MLP)":   "#457b9d",
    "JEPA (ours)":     "#e63946",
}
MARKERS = {
    "Persistence":      "s",
    "gLV-L2 (Ridge)":  "^",
    "gLV-net (MLP)":   "D",
    "JEPA (ours)":     "o",
}
LINESTYLES = {
    "Persistence":      (0, (4, 2)),
    "gLV-L2 (Ridge)":  (0, (3, 1, 1, 1)),
    "gLV-net (MLP)":   "--",
    "JEPA (ours)":     "-",
}
LINEWIDTHS = {
    "Persistence":      1.6,
    "gLV-L2 (Ridge)":  1.8,
    "gLV-net (MLP)":   1.8,
    "JEPA (ours)":     2.6,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
})


def _rmse_arrays(model):
    means = [results[model]["clr_rmse"][h]["mean"] for h in HKEYS]
    stds  = [results[model]["clr_rmse"][h]["std"]  for h in HKEYS]
    return np.array(means), np.array(stds)

def _skill_array(model):
    return np.array([results[model]["skill"][h] for h in HKEYS])


# ── Fig 1 — CLR-RMSE vs horizon ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.4))

for name in results:
    means, stds = _rmse_arrays(name)
    ax.plot(HORIZONS, means,
            color=COLORS[name], marker=MARKERS[name],
            linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name],
            markersize=7, label=name, zorder=3 if name == "JEPA (ours)" else 2)
    ax.fill_between(HORIZONS, means - stds, means + stds,
                    color=COLORS[name], alpha=0.12, zorder=1)

ax.set_xlabel("Forecast horizon (steps)")
ax.set_ylabel("CLR-RMSE (↓ better)")
ax.set_title("Temporal benchmark — CLR-RMSE vs horizon\n(MDSINE2 hold-one-subject-out, 30 subjects, 3 seeds)")
ax.set_xticks(HORIZONS)
ax.legend(framealpha=0.9, loc="upper left")
ax.set_xlim(0.5, 11)
ax.set_ylim(0, 1.45)

# annotate JEPA at h=10
jepa_h10 = results["JEPA (ours)"]["clr_rmse"]["h10"]["mean"]
ax.annotate(f"JEPA: {jepa_h10:.3f}",
            xy=(10, jepa_h10), xytext=(8.2, jepa_h10 + 0.08),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["JEPA (ours)"], lw=1.4),
            color=COLORS["JEPA (ours)"], fontsize=9.5, fontweight="bold")

fig.tight_layout()
fig.savefig(f"{OUT}/glv_benchmark_jepa_rmse.png", dpi=150)
plt.close(fig)
print("saved glv_benchmark_jepa_rmse.png")


# ── Fig 2 — Skill vs persistence ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.4))

for name in ["gLV-L2 (Ridge)", "gLV-net (MLP)", "JEPA (ours)"]:
    skill = _skill_array(name)
    ax.plot(HORIZONS, skill,
            color=COLORS[name], marker=MARKERS[name],
            linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name],
            markersize=7, label=name, zorder=3 if name == "JEPA (ours)" else 2)

ax.axhline(1.0, color=COLORS["Persistence"], linewidth=1.5,
           linestyle=(0, (4, 2)), label="Persistence (=1.0)", zorder=1)
ax.fill_between(HORIZONS, 1.0, 1.0, color=COLORS["Persistence"], alpha=0.0)

ax.set_xlabel("Forecast horizon (steps)")
ax.set_ylabel("Skill score  (pers. RMSE / model RMSE, ↑ better)")
ax.set_title("Skill vs persistence\n(>1 = beats no-change baseline)")
ax.set_xticks(HORIZONS)
ax.legend(framealpha=0.9, loc="upper left")
ax.set_xlim(0.5, 11)
ax.set_ylim(0.8, 3.1)

# shade "beats persistence" zone
ax.axhspan(1.0, 3.1, color="#d4edda", alpha=0.3, zorder=0, label=None)
ax.text(10.4, 1.05, "beats\npersistence", fontsize=7.5, color="#2d6a4f", va="bottom")

# annotate JEPA improvement over gLV-net at h=10
j10 = results["JEPA (ours)"]["skill"]["h10"]
g10 = results["gLV-net (MLP)"]["skill"]["h10"]
ax.annotate(f"+{(j10-g10)*100/g10:.0f}% vs\ngLV-net",
            xy=(10, j10), xytext=(8.0, j10 + 0.08),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["JEPA (ours)"], lw=1.4),
            color=COLORS["JEPA (ours)"], fontsize=9, fontweight="bold")

fig.tight_layout()
fig.savefig(f"{OUT}/glv_benchmark_jepa_skill.png", dpi=150)
plt.close(fig)
print("saved glv_benchmark_jepa_skill.png")


# ── Fig 3 — Grouped bar chart (slides format) ─────────────────────────────────
models_bar = ["Persistence", "gLV-L2 (Ridge)", "gLV-net (MLP)", "JEPA (ours)"]
n_models   = len(models_bar)
n_horizons = len(HORIZONS)
width = 0.18
x = np.arange(n_horizons)

fig, ax = plt.subplots(figsize=(8, 4.5))

for i, name in enumerate(models_bar):
    means, stds = _rmse_arrays(name)
    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, means, width, yerr=stds,
                  label=name, color=COLORS[name],
                  error_kw=dict(elinewidth=1.2, capsize=3),
                  edgecolor="white", linewidth=0.4,
                  zorder=3 if name == "JEPA (ours)" else 2)
    if name == "JEPA (ours)":
        for bar in bars:
            bar.set_edgecolor("#c1121f")
            bar.set_linewidth(1.6)

ax.set_xlabel("Forecast horizon (steps)")
ax.set_ylabel("CLR-RMSE (↓ better)")
ax.set_title("CLR-RMSE by model and horizon  ·  MDSINE2-HOSO protocol")
ax.set_xticks(x)
ax.set_xticklabels([f"h={h}" for h in HORIZONS])
ax.legend(framealpha=0.9)
ax.set_ylim(0, 1.45)

fig.tight_layout()
fig.savefig(f"{OUT}/glv_benchmark_jepa_bars.png", dpi=150)
plt.close(fig)
print("saved glv_benchmark_jepa_bars.png")


# ── Fig 4 — JEPA gain over gLV-net (%) ───────────────────────────────────────
jepa_means = np.array([results["JEPA (ours)"]["clr_rmse"][h]["mean"] for h in HKEYS])
net_means  = np.array([results["gLV-net (MLP)"]["clr_rmse"][h]["mean"] for h in HKEYS])
l2_means   = np.array([results["gLV-L2 (Ridge)"]["clr_rmse"][h]["mean"] for h in HKEYS])
pers_means = np.array([results["Persistence"]["clr_rmse"][h]["mean"]    for h in HKEYS])

gain_over_net  = (net_means  - jepa_means) / net_means  * 100
gain_over_l2   = (l2_means   - jepa_means) / l2_means   * 100
gain_over_pers = (pers_means - jepa_means) / pers_means * 100

fig, ax = plt.subplots(figsize=(6.5, 4.0))

x = np.arange(n_horizons)
w = 0.25
ax.bar(x - w, gain_over_pers, w, label="vs Persistence",     color="#9e9e9e", alpha=0.8)
ax.bar(x,     gain_over_l2,   w, label="vs gLV-L2 (Ridge)",  color="#f4a261", alpha=0.9)
ax.bar(x + w, gain_over_net,  w, label="vs gLV-net (MLP)",   color="#457b9d", alpha=0.9)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Forecast horizon (steps)")
ax.set_ylabel("RMSE reduction by JEPA (%)")
ax.set_title("JEPA improvement over baselines\n(positive = JEPA better)")
ax.set_xticks(x)
ax.set_xticklabels([f"h={h}" for h in HORIZONS])
ax.legend(framealpha=0.9)

# value labels
for rect in ax.patches:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, h + 0.4,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

ax.set_ylim(0, max(gain_over_pers) * 1.18)
fig.tight_layout()
fig.savefig(f"{OUT}/glv_benchmark_jepa_gain.png", dpi=150)
plt.close(fig)
print("saved glv_benchmark_jepa_gain.png")


# ── Fig 5 — Dashboard (2×2 summary) ──────────────────────────────────────────
fig = plt.figure(figsize=(12, 8))
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.34)

# top-left: CLR-RMSE
ax1 = fig.add_subplot(gs[0, 0])
for name in results:
    means, stds = _rmse_arrays(name)
    ax1.plot(HORIZONS, means,
             color=COLORS[name], marker=MARKERS[name],
             linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name],
             markersize=6, label=name, zorder=3 if name == "JEPA (ours)" else 2)
    ax1.fill_between(HORIZONS, means - stds, means + stds,
                     color=COLORS[name], alpha=0.12)
ax1.set_title("CLR-RMSE vs horizon (↓ better)", fontsize=10, fontweight="bold")
ax1.set_xlabel("Horizon (steps)"); ax1.set_ylabel("CLR-RMSE")
ax1.set_xticks(HORIZONS); ax1.set_xlim(0.5, 11)
ax1.legend(fontsize=8, framealpha=0.9)

# top-right: skill
ax2 = fig.add_subplot(gs[0, 1])
for name in ["gLV-L2 (Ridge)", "gLV-net (MLP)", "JEPA (ours)"]:
    skill = _skill_array(name)
    ax2.plot(HORIZONS, skill,
             color=COLORS[name], marker=MARKERS[name],
             linestyle=LINESTYLES[name], linewidth=LINEWIDTHS[name],
             markersize=6, label=name)
ax2.axhline(1.0, color=COLORS["Persistence"], linewidth=1.4, linestyle=(0, (4, 2)), label="Persistence")
ax2.axhspan(1.0, 3.2, color="#d4edda", alpha=0.25)
ax2.set_title("Skill vs persistence (↑ better)", fontsize=10, fontweight="bold")
ax2.set_xlabel("Horizon (steps)"); ax2.set_ylabel("Skill score")
ax2.set_xticks(HORIZONS); ax2.set_xlim(0.5, 11); ax2.set_ylim(0.8, 3.1)
ax2.legend(fontsize=8, framealpha=0.9)

# bottom-left: bars h=1 and h=10
ax3 = fig.add_subplot(gs[1, 0])
horizon_pairs = [("h1", "h=1"), ("h10", "h=10")]
models_b = ["Persistence", "gLV-L2 (Ridge)", "gLV-net (MLP)", "JEPA (ours)"]
x3 = np.arange(len(horizon_pairs))
wb = 0.18
for i, name in enumerate(models_b):
    vals = [results[name]["clr_rmse"][hk]["mean"] for hk, _ in horizon_pairs]
    errs = [results[name]["clr_rmse"][hk]["std"]  for hk, _ in horizon_pairs]
    off  = (i - 1.5) * wb
    ax3.bar(x3 + off, vals, wb, yerr=errs, label=name, color=COLORS[name],
            error_kw=dict(elinewidth=1.1, capsize=2.5),
            edgecolor="white", linewidth=0.3)
ax3.set_title("CLR-RMSE at short & long horizon", fontsize=10, fontweight="bold")
ax3.set_xticks(x3); ax3.set_xticklabels([lbl for _, lbl in horizon_pairs])
ax3.set_ylabel("CLR-RMSE"); ax3.legend(fontsize=7.5, framealpha=0.9)

# bottom-right: JEPA % gain table rendered as heatmap-style bars
ax4 = fig.add_subplot(gs[1, 1])
categories   = ["vs Persistence", "vs gLV-L2", "vs gLV-net"]
gains_matrix = np.array([gain_over_pers, gain_over_l2, gain_over_net])  # (3, 4)
cmap = plt.cm.Blues
im = ax4.imshow(gains_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=65)
ax4.set_xticks(range(4)); ax4.set_xticklabels([f"h={h}" for h in HORIZONS])
ax4.set_yticks(range(3)); ax4.set_yticklabels(categories, fontsize=9)
ax4.set_title("JEPA RMSE reduction over baselines (%)", fontsize=10, fontweight="bold")
for r in range(3):
    for c in range(4):
        ax4.text(c, r, f"{gains_matrix[r, c]:.1f}%",
                 ha="center", va="center", fontsize=9,
                 color="white" if gains_matrix[r, c] > 40 else "black")
fig.colorbar(im, ax=ax4, shrink=0.85, label="% improvement")

fig.suptitle("JEPA Microbiome — Benchmark temporal MDSINE2-HOSO\n"
             "30 subjects · 3 seeds · gLV synthetic (32 species)",
             fontsize=12, fontweight="bold", y=1.01)

fig.savefig(f"{OUT}/glv_benchmark_jepa_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved glv_benchmark_jepa_dashboard.png")


# ── persist JSON ─────────────────────────────────────────────────────────────
out_json = {
    "metadata": {
        "protocol": "MDSINE2 hold-one-subject-out",
        "n_subjects": 30,
        "seeds": [0, 1, 2],
        "horizons": HORIZONS,
        "metric": "CLR-RMSE",
    },
    "results": results,
    "jepa_gain_pct": {
        "vs_persistence": dict(zip([f"h{h}" for h in HORIZONS], gain_over_pers.tolist())),
        "vs_glv_l2":      dict(zip([f"h{h}" for h in HORIZONS], gain_over_l2.tolist())),
        "vs_glv_net":     dict(zip([f"h{h}" for h in HORIZONS], gain_over_net.tolist())),
    },
}
with open(os.path.join(os.path.dirname(__file__), "glv_benchmark_jepa.json"), "w") as f:
    json.dump(out_json, f, indent=2)
print("saved glv_benchmark_jepa.json")
print("\nAll done.")
