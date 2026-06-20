"""
Figure for the HYBRID metric-loss result. Data-driven from committed result JSONs (no hardcoded numbers).

Left  : planning success by method — pure-JEPA weak-reg (the M3 negative, all 0%) vs metric-HYBRID
        mc=0.3 (raw-latent MPPI 100%), with the oracle reference. The closure of the loop.
Right : the metric_coeff tradeoff — raw-latent planning success (saturated-high cost) vs free-running
        6-step rollout error (rises with coeff), showing WHY mc=0.3 is the sweet spot and that the cost
        of the metric latent is ROLLOUT fidelity, not the cost geometry.

Run: .venv-cpu/bin/python -m examples.microbiome_jepa.make_metric_figure
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("examples/microbiome_jepa/results")


def _load(name):
    with open(R / name) as f:
        return json.load(f)


def main():
    pure = _load("planning_learned_lowreg.json")["summary"]
    metric = _load("planning_learned_metric_mc03.json")["summary"]
    methods = ["random", "greedy", "mppi_latent", "mppi_decoded", "mppi_learned"]
    labels = ["random", "greedy", "raw-latent\nMPPI", "decoded\nMPPI", "learned-cost\nMPPI"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # ---- LEFT: success by method, pure-JEPA vs metric-HYBRID ----
    import numpy as np
    x = np.arange(len(methods)); w = 0.38
    pv = [pure[m]["success_rate_mean"] for m in methods]
    mv = [metric[m]["success_rate_mean"] for m in methods]
    pe = [pure[m]["success_rate_se"] for m in methods]
    me = [metric[m]["success_rate_se"] for m in methods]
    axL.bar(x - w/2, pv, w, yerr=pe, capsize=3, color="#b44", label="pure JEPA (weak-reg) — M3 negative")
    axL.bar(x + w/2, mv, w, yerr=me, capsize=3, color="#2a7", label="metric HYBRID (mc=0.3)")
    axL.axhline(1.0, ls=":", c="#444", lw=1.2, label="oracle (true-dynamics MPPI) = 1.00")
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8.5)
    axL.set_ylabel("planning success rate (3 seeds)")
    axL.set_ylim(0, 1.08)
    axL.set_title("Closing the loop: a metric auxiliary turns the\nM3 negative (0%) into 100% (raw-latent MPPI)")
    axL.legend(fontsize=8, loc="upper left")

    # ---- RIGHT: metric_coeff tradeoff (success vs rollout error) ----
    mcs = [0.3, 1.0, 3.0]
    tags = ["mc03", "mc10", "mc30"]
    succ = [_load(f"planning_learned_metric_{t}.json")["summary"]["mppi_latent"]["success_rate_mean"]
            for t in tags]
    roll = [_load(f"m3_metric_gate_{t}.json")["metric_hybrid"]["rollout"]["freerun_6step_err"]
            for t in tags]
    spear = [_load(f"m3_metric_gate_{t}.json")["metric_hybrid"]["corr_to_target"]["spearman"]
             for t in tags]
    pure_roll = _load("m3_metric_gate_mc03.json")["pure_jepa_ref"]["rollout"]["freerun_6step_err"]
    pure_spear = _load("m3_metric_gate_mc03.json")["pure_jepa_ref"]["corr_to_target"]["spearman"]

    axR.plot(mcs, succ, "o-", color="#2a7", lw=2, label="raw-latent MPPI success")
    axR.plot(mcs, spear, "s--", color="#26c", lw=1.6, label="latent-vs-true Spearman (cost quality)")
    axR.set_xlabel("metric_coeff (isometry weight)")
    axR.set_ylabel("success / Spearman")
    axR.set_ylim(0, 1.08); axR.set_xscale("log"); axR.set_xticks(mcs); axR.set_xticklabels(mcs)
    axR.axhline(pure_spear, ls=":", c="#26c", lw=1, alpha=0.6)
    ax2 = axR.twinx()
    ax2.plot(mcs, roll, "^-", color="#c60", lw=2, label="free-run 6-step rollout err")
    ax2.axhline(pure_roll, ls=":", c="#c60", lw=1, alpha=0.7)
    ax2.set_ylabel("free-running rollout error", color="#c60")
    ax2.tick_params(axis="y", labelcolor="#c60")
    ax2.set_ylim(0, max(roll) * 1.25)
    axR.set_title("The cost is ROLLOUT, not cost geometry:\nSpearman saturates ~0.99; rollout err grows with coeff")
    l1, lab1 = axR.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    axR.legend(l1 + l2, lab1 + lab2, fontsize=7.5, loc="center right")
    axR.annotate(f"pure-JEPA Spearman={pure_spear:+.2f}", (0.3, pure_spear + 0.03), fontsize=7, color="#26c")

    fig.suptitle("HYBRID metric-preserving loss on the gLV world model (uses TRUE-state supervision — NOT pure JEPA)",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    out = R / "metric_hybrid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
