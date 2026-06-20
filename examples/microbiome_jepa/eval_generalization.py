"""
EXP1 GENERALIZATION eval — does the M3 metric closure (raw-latent MPPI ~ oracle) hold across DIFFERENT
gLV instances, not just the headline toy system?

Per instance, on the SAME (src,tgt) episodes and the SAME tol, we run:
  * ORACLE  = true-dynamics state-space MPPI (perfect model) — the controllability reference,
  * mppi_latent = LEARNED raw-latent-distance MPPI on the HYBRID mc=0.3 world model (the clean closure
    test: no decoder, no learned cost — exactly the metric the closure claims to supply),
  * random / greedy baselines.
Both planners use identical MPPI settings; the only difference is true vs learned dynamics + cost. The
claim holds iff mppi_latent reaches (or nears) the oracle's success/final on each instance.

INTEGRITY: every number is from real MPC rollouts on the true GLVSimulator; all seeded. Oracle settings
match oracle_K_sweep / screen_instances (n_samples=96, temperature=1.0). Learned-MPPI settings match the
committed headline planning eval (n_samples=128, temperature=1.0, horizon=6).

Run (CPU, after fetching the gen checkpoints):
  .venv-cpu/bin/python -m examples.microbiome_jepa.eval_generalization
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from examples.microbiome_jepa.plan_glv import (
    MPPIConfig, build_glv_and_encoder, build_world_model, mppi_plan, _greedy_action,
)
from examples.microbiome_jepa.oracle_K_sweep import _oracle_mppi_action

# instance name -> (structural knobs for the gLV system + the checkpoint dir tag). K = n_species.
INSTANCES = {
    "g4_s24":            dict(n_species=24, n_candidate=24, n_guilds=4, comp_strong=-2.5, comp_weak=-0.4),
    "g3_s18":            dict(n_species=18, n_candidate=18, n_guilds=3, comp_strong=-2.5, comp_weak=-0.4),
    "g5_s30":            dict(n_species=30, n_candidate=30, n_guilds=5, comp_strong=-2.5, comp_weak=-0.4),
    "g3_s24_strongcomp": dict(n_species=24, n_candidate=24, n_guilds=3, comp_strong=-3.5, comp_weak=-0.25),
}
WEAKREG = {"model.d_model": 128, "model.regularizer.sim_coeff_t": 4,
           "model.regularizer.cov_coeff": 1, "model.regularizer.std_coeff": 0.25,
           "model.regularizer.metric_coeff": 0.3}


@torch.no_grad()
def eval_instance(name, knobs, ckpt_root, seeds=(0, 1, 2), n_episodes=12, mpc_steps=20,
                  horizon=6, tol_frac=0.15, device="cpu",
                  orc_samples=96, lat_samples=128, n_elites=16, n_iters=3, init_std=0.25, temperature=1.0):
    dev = torch.device(device)
    overrides = {f"data.{k}": v for k, v in knobs.items()}
    overrides.update(WEAKREG)
    ckpt = f"{ckpt_root}/plan_model_gen_{name}/latest.pth.tar"
    jepa, cfg, K = build_world_model("examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml", ckpt, dev,
                                     overrides=overrides)
    sim, state_enc = build_glv_and_encoder(cfg, dev)
    attr = sim.attractors
    n_attr = len(attr)
    inter = [float(np.linalg.norm(attr[i] - attr[j])) for i in range(n_attr) for j in range(n_attr) if i != j]
    tol = tol_frac * float(np.mean(inter))
    amax = float(sim.config.action_max)
    lat_cfg = MPPIConfig(horizon=horizon, n_samples=lat_samples, n_elites=n_elites, n_iters=n_iters,
                         init_std=init_std, temperature=temperature, cumulative=True)

    methods = ["random", "greedy", "oracle", "mppi_latent"]
    recs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tgen = torch.Generator(device=dev).manual_seed(seed)
        pairs = []
        for _ in range(n_episodes):
            s = int(rng.integers(n_attr)); t = int(rng.integers(n_attr - 1)); t += int(t >= s)
            pairs.append((s, t))
        for (src, tgt) in pairs:
            target = attr[tgt]
            for m in methods:
                x = sim.reset(attractor=src).astype(np.float32)
                start = float(np.linalg.norm(x - target))
                z_tgt = state_enc.encode(jepa, target).flatten(1)
                warm = None
                for _ in range(mpc_steps):
                    if m == "random":
                        a = rng.uniform(0.0, amax, size=K).astype(np.float32)
                    elif m == "greedy":
                        a = _greedy_action(sim, x, target, amax)
                    elif m == "oracle":
                        a = _oracle_mppi_action(sim, x, target, amax, K, horizon, orc_samples, n_elites,
                                                n_iters, init_std, temperature, rng)
                    else:  # mppi_latent (learned raw-latent MPPI)
                        z0 = state_enc.encode(jepa, x)
                        a_t, mean_plan = mppi_plan(jepa.predictor, z0, z_tgt, K, amax, lat_cfg,
                                                   mean_init=warm, generator=tgen)
                        warm = torch.zeros_like(mean_plan); warm[:-1] = mean_plan[1:]
                        a = a_t.detach().cpu().numpy().astype(np.float32)
                    x = sim.step(a).astype(np.float32)
                    if np.linalg.norm(x - target) < tol:
                        break
                fd = float(np.linalg.norm(x - target))
                recs.append({"seed": seed, "method": m, "success": bool(fd < tol),
                             "final_dist": fd, "start_dist": start})

    summary = {}
    for m in methods:
        per_seed = [float(np.mean([r["success"] for r in recs if r["method"] == m and r["seed"] == s]))
                    for s in seeds]
        allm = [r for r in recs if r["method"] == m]
        summary[m] = {"success_mean": float(np.mean(per_seed)),
                      "success_se": float(np.std(per_seed, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0,
                      "final_dist": float(np.mean([r["final_dist"] for r in allm]))}
    res = {"instance": name, "knobs": knobs, "K": K, "tol": tol, "n_attractors": n_attr,
           "seeds": list(seeds), "n_episodes": n_episodes,
           "mean_start_dist": float(np.mean([r["start_dist"] for r in recs])),
           "summary": summary,
           "crosses_tol": bool(summary["mppi_latent"]["success_mean"] > 0),
           "near_oracle": bool(summary["mppi_latent"]["success_mean"] >= 0.9 * summary["oracle"]["success_mean"])}
    o, l = summary["oracle"], summary["mppi_latent"]
    print(f"[{name:18s}] tol={tol:.3f} start={res['mean_start_dist']:.2f} | "
          f"ORACLE {o['success_mean']*100:.0f}%/{o['final_dist']:.2f}  "
          f"mppi_latent {l['success_mean']*100:.0f}%±{l['success_se']*100:.0f}/{l['final_dist']:.2f}  "
          f"-> {'CLOSES (near-oracle)' if res['near_oracle'] else ('crosses tol' if res['crosses_tol'] else 'FAILS')}")
    return res


def run(ckpt_root: str = "checkpoints", device: str = "cpu",
        out: str = "examples/microbiome_jepa/results"):
    results = {name: eval_instance(name, knobs, ckpt_root, device=device)
               for name, knobs in INSTANCES.items()}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "exp1_generalization.json").write_text(json.dumps(results, indent=2))
    # markdown
    md = ["# EXP1 generalization across gLV instances (MEASURED)\n",
          "HYBRID mc=0.3 world model per instance. `mppi_latent` = LEARNED raw-latent-distance MPPI "
          "(the clean closure test). ORACLE = true-dynamics MPPI (controllability ref). Same episodes "
          "(3 seeds x 12) + tol per instance.\n",
          "| instance | guilds/S/K | tol | oracle succ/final | mppi_latent succ/final | crosses tol | near-oracle |",
          "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        o, l = r["summary"]["oracle"], r["summary"]["mppi_latent"]
        kn = r["knobs"]
        md.append(f"| {name} | {kn['n_guilds']}/{kn['n_species']}/{r['K']} | {r['tol']:.3f} | "
                  f"{o['success_mean']*100:.0f}% / {o['final_dist']:.2f} | "
                  f"{l['success_mean']*100:.0f}%±{l['success_se']*100:.0f} / {l['final_dist']:.2f} | "
                  f"{'yes' if r['crosses_tol'] else 'NO'} | {'yes' if r['near_oracle'] else 'no'} |")
    (Path(out) / "exp1_generalization.md").write_text("\n".join(md))
    print("\n".join(md[2:]))
    n_close = sum(r["near_oracle"] for r in results.values())
    print(f"\nclosure holds (near-oracle) on {n_close}/{len(results)} instances")
    print(f"saved -> {out}/exp1_generalization.json + .md")
    return results


if __name__ == "__main__":
    import fire
    fire.Fire(run)
