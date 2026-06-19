"""
HEADLINE driver — the IDM-ablation collapse-and-recovery experiment on gLV.

For each seed and each arm (idm_coeff = 1.0 "on" vs 0.0 "off") it:
  1. trains the Layer-B world model (examples/microbiome_jepa/train_worldmodel.run),
  2. freezes the encoder and runs the collapse probes (eval_collapse) on HELD-OUT gLV trajectories
     (same fixed species embeddings as training; a held-out sim_seed),
  3. records fast (dynamics) vs slow (initial-state) decodability + training Lpred/Lvar.
Then it aggregates mean +/- standard error per arm, prints a table, and saves a JSON of every raw
number plus a grouped-bar figure. Hypothesis: IDM ON raises fast_r2_delta (dynamics recovered) while
slow_r2_init is similar -> collapse-and-recovery.

Run (CPU smoke):
  .venv-cpu/bin/python -m examples.microbiome_jepa.run_ablation \
      --seeds 0 --epochs 3 --n_traj 64 --eval_n_traj 48 --out checkpoints/microbiome_jepa/ablation_smoke
Real run (3 seeds, more epochs) — cheap on gLV, runs on CPU or GPU:
  ... --seeds 0,1,2 --epochs 80 --n_traj 512 --eval_n_traj 128 --out checkpoints/microbiome_jepa/ablation
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eb_jepa.datasets.microbiome.traj import GLVTrajConfig, GLVTrajDataset
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config
from examples.microbiome_jepa.eval_collapse import probe_encoder
from examples.microbiome_jepa.train_worldmodel import run as train_run

logger = get_logger(__name__)

ARMS = [("idm_on", 1.0), ("idm_off", 0.0)]
PROBE_KEYS = ["fast_r2_action", "fast_r2_delta", "fast_r2_state", "slow_r2_init", "fast_minus_slow", "feat_std"]


def _agg(values):
    """mean and standard error over a list."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)


def run(
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    seeds: str = "0,1,2",
    epochs: int = 80,
    n_traj: int = 512,
    eval_n_traj: int = 128,
    batch_size: int = 64,
    out: str = "checkpoints/microbiome_jepa/ablation",
    # --- optional sweep knobs (None => use the yaml value). For tuning the collapse regime
    #     (Sobal: weak variance-reg + strong temporal-smoothness tempts slow-feature collapse that
    #     IDM then rescues) and for shrinking the model when iterating fast.
    sim_coeff_t: float = None,
    cov_coeff: float = None,
    std_coeff: float = None,
    d_model: int = None,
    n_species: int = None,
    pool: str = None,
    use_amp: bool = None,
):
    # fire turns "0,1,2" into a tuple (0,1,2) and "0" into int 0 — handle all of int/list/tuple/str.
    if isinstance(seeds, (list, tuple)):
        seed_list = [int(s) for s in seeds]
    elif isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip() != ""]
    sweep_ov = {}
    if sim_coeff_t is not None: sweep_ov["model.regularizer.sim_coeff_t"] = sim_coeff_t
    if cov_coeff is not None: sweep_ov["model.regularizer.cov_coeff"] = cov_coeff
    if std_coeff is not None: sweep_ov["model.regularizer.std_coeff"] = std_coeff
    if d_model is not None: sweep_ov["model.d_model"] = d_model
    if n_species is not None:
        sweep_ov["data.n_species"] = n_species
        sweep_ov["data.n_max"] = n_species
    if pool is not None: sweep_ov["model.pool"] = pool
    if use_amp is not None: sweep_ov["training.use_amp"] = use_amp
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []  # one dict per (seed, arm)

    for seed in seed_list:
        for arm, coeff in ARMS:
            ov = {
                "meta.seed": seed,
                "model.regularizer.idm_coeff": coeff,
                "optim.epochs": epochs,
                "data.n_traj": n_traj,
                "data.batch_size": batch_size,
                "data.seed": seed,
                "logging.tqdm_silent": True,
                "logging.log_wandb": False,
                **sweep_ov,
            }
            cfg = load_config(fname, ov, quiet=True)
            logger.info(f"=== train seed={seed} arm={arm} (idm_coeff={coeff}) epochs={epochs} ===")
            metrics, jepa = train_run(cfg=cfg, return_model=True)
            device = next(jepa.parameters()).device

            # Held-out eval trajectories: SAME fixed species embeddings (emb_seed) as training,
            # SAME gLV params, but a held-out sim_seed so trajectories are unseen.
            eval_cfg = GLVTrajConfig(
                n_traj=eval_n_traj,
                T=int(cfg.data.T),
                n_species=int(cfg.data.n_species),
                n_candidate=int(cfg.data.n_candidate),
                dt=float(cfg.data.dt),
                noise_std=float(cfg.data.get("noise_std", 0.0)),
                emb_seed=int(cfg.data.get("emb_seed", 0)),
                sim_seed=10_000 + seed,  # held out from training sim_seed
            )
            eval_ds = GLVTrajDataset(eval_cfg)
            probes = probe_encoder(jepa.encoder, eval_ds, device, seed=seed)

            rec = {
                "seed": seed, "arm": arm, "idm_coeff": coeff,
                "train_pred": metrics.get("pred"), "train_std_loss": metrics.get("std_loss"),
                "train_idm_loss": metrics.get("idm_loss"), "used_stub_glv": eval_ds.used_stub,
                **{k: probes[k] for k in PROBE_KEYS},
            }
            records.append(rec)
            logger.info(f"  -> fast_r2_delta={probes['fast_r2_delta']:.3f} "
                        f"slow_r2_init={probes['slow_r2_init']:.3f} "
                        f"feat_std={probes['feat_std']:.3f}")

    # ---- aggregate per arm ----
    summary = {}
    for arm, _ in ARMS:
        arm_recs = [r for r in records if r["arm"] == arm]
        summary[arm] = {k: _agg([r[k] for r in arm_recs]) for k in PROBE_KEYS}

    # ---- print table ----
    print("\n================ IDM ABLATION (gLV collapse-and-recovery) ================")
    print(f"seeds={seed_list} epochs={epochs} n_traj={n_traj} eval_n_traj={eval_n_traj} "
          f"stub_glv={records[0]['used_stub_glv'] if records else '?'}")
    hdr = "metric".ljust(16) + "".join(a.ljust(20) for a, _ in ARMS)
    print(hdr)
    for k in PROBE_KEYS:
        row = k.ljust(16)
        for arm, _ in ARMS:
            m, se = summary[arm][k]
            row += f"{m:+.3f} ± {se:.3f}".ljust(20)
        print(row)
    print("Interpretation: IDM ON should raise fast_r2_delta (dynamics recovered); slow_r2_init similar.")

    # ---- save raw + summary JSON ----
    res_path = out_dir / "ablation_results.json"
    with open(res_path, "w") as f:
        json.dump({"seeds": seed_list, "epochs": epochs, "n_traj": n_traj,
                   "eval_n_traj": eval_n_traj, "config_overrides": sweep_ov,
                   "records": records, "summary": summary}, f, indent=2)
    print(f"\nsaved raw numbers -> {res_path}")

    # ---- figure: grouped bars, fast (dynamics) vs slow (identity), idm on vs off ----
    fig_keys = ["fast_r2_action", "fast_r2_delta", "slow_r2_init"]
    x = np.arange(len(fig_keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (arm, _) in enumerate(ARMS):
        means = [summary[arm][k][0] for k in fig_keys]
        ses = [summary[arm][k][1] for k in fig_keys]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=ses, capsize=4,
               label=arm, color=("#2a7" if arm == "idm_on" else "#c44"))
    ax.set_xticks(x)
    ax.set_xticklabels(["fast: action\n(intervention)", "fast: Δstate\n(dynamics)", "slow: init\n(identity)"])
    ax.set_ylabel("linear-probe R²  (held-out gLV)")
    ax.set_title(f"IDM ablation on gLV — dynamics vs slow-feature decodability\n"
                 f"(mean ± s.e., {len(seed_list)} seed(s), {epochs} epochs)")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "ablation_collapse_recovery.png"
    fig.savefig(fig_path, dpi=140)
    print(f"saved figure    -> {fig_path}")
    return {"summary": summary, "results_json": str(res_path), "figure": str(fig_path)}


if __name__ == "__main__":
    fire.Fire(run)
