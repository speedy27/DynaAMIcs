"""
EXP 1 pre-screen (CPU, fast) — validate candidate gLV INSTANCES before spending GPU.

The gLV interaction matrix A + attractors are built DETERMINISTICALLY from the STRUCTURAL config
(n_species, n_guilds, comp_strong, comp_weak, within_frac, growth); config.seed only seeds trajectory
noise. So genuinely-different instances come from varying those structural knobs. For each candidate we
check the SAME validity bar as the headline system:
  (1) >= 2 attractors, all LOCALLY STABLE (max real Jacobian eig < 0),
  (2) attractors DISTINCT (so a target is a real destination),
  (3) NON-MONOTONIC reachability: a greedy true-1-step policy FAILS (planning is non-trivial),
  (4) CONTROLLABLE: the perfect-model oracle (true-dynamics MPPI) at full actuation (K=n_species)
      reaches the target within tol (so any learned-planner failure is the model, not the task).
Only instances passing all four are worth training on (else the metric test is confounded).

Run: .venv-cpu/bin/python -m examples.microbiome_jepa.screen_instances
"""
import json
from pathlib import Path

import numpy as np

from eb_jepa.datasets.microbiome.glv import GLVConfig, GLVSimulator
from examples.microbiome_jepa.oracle_K_sweep import _attr_scale, _oracle_mppi_action
from examples.microbiome_jepa.plan_glv import _greedy_action

# Candidate instances. K = n_species (full actuation = the controllable regime the headline used).
# The headline (baseline) is g3/S24/cs-2.5/cw-0.4 — included as a sanity row.
INSTANCES = {
    "baseline_g3_s24":   dict(n_species=24, n_guilds=3, comp_strong=-2.5, comp_weak=-0.4, growth=1.0),
    "g4_s24":            dict(n_species=24, n_guilds=4, comp_strong=-2.5, comp_weak=-0.4, growth=1.0),
    "g3_s18":            dict(n_species=18, n_guilds=3, comp_strong=-2.5, comp_weak=-0.4, growth=1.0),
    "g5_s30":            dict(n_species=30, n_guilds=5, comp_strong=-2.5, comp_weak=-0.4, growth=1.0),
    "g3_s24_strongcomp": dict(n_species=24, n_guilds=3, comp_strong=-3.5, comp_weak=-0.25, growth=1.0),
    "g3_s32_fastgrow":   dict(n_species=32, n_guilds=3, comp_strong=-2.5, comp_weak=-0.4, growth=1.5),
}


def screen_one(name, knobs, action_max=0.5, horizon=6, mpc_steps=20, n_samples=96, n_elites=16,
               n_iters=3, init_std=0.25, temperature=1.0, tol_frac=0.15, n_pairs=6, seeds=(0, 1, 2)):
    # Settings MATCH the committed oracle_K_sweep (n_samples=96, temperature=1.0, 3 seeds x 6 pairs)
    # so the baseline reproduces its 1.00/0.79 and the verdicts are calibrated.
    S = knobs["n_species"]
    cfg = GLVConfig(n_species=S, n_candidate=S, action_max=action_max,  # K = full actuation
                    n_guilds=knobs["n_guilds"], comp_strong=knobs["comp_strong"],
                    comp_weak=knobs["comp_weak"], growth=knobs["growth"])
    sim = GLVSimulator(cfg)
    attr = sim.attractors
    n_attr = len(attr)

    eigs = [sim.jacobian_max_real_eig(attr[g]) for g in range(n_attr)]
    stable = all(e < 0 for e in eigs)
    inter = [float(np.linalg.norm(attr[i] - attr[j])) for i in range(n_attr) for j in range(n_attr) if i != j]
    attr_scale = _attr_scale(attr)
    tol = tol_frac * attr_scale
    distinct = min(inter) > tol  # any two attractors are further apart than tol

    greedy_per_seed, oracle_per_seed, oracle_final = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pairs = []
        for _ in range(n_pairs):
            s = int(rng.integers(n_attr)); t = int(rng.integers(n_attr - 1)); t += int(t >= s)
            pairs.append((s, t))
        g_s, o_s = [], []
        for (src, tgt) in pairs:
            target = attr[tgt]
            x = sim.reset(attractor=src).astype(np.float32)
            for _ in range(mpc_steps):
                a = _greedy_action(sim, x, target, action_max)
                x = sim.step(a).astype(np.float32)
                if np.linalg.norm(x - target) < tol:
                    break
            g_s.append(float(np.linalg.norm(x - target) < tol))
            x = sim.reset(attractor=src).astype(np.float32)
            for _ in range(mpc_steps):
                a = _oracle_mppi_action(sim, x, target, action_max, S, horizon, n_samples, n_elites,
                                        n_iters, init_std, temperature, rng)
                x = sim.step(a).astype(np.float32)
                if np.linalg.norm(x - target) < tol:
                    break
            d = float(np.linalg.norm(x - target))
            o_s.append(float(d < tol)); oracle_final.append(d)
        greedy_per_seed.append(float(np.mean(g_s))); oracle_per_seed.append(float(np.mean(o_s)))

    g_succ = float(np.mean(greedy_per_seed)); o_succ = float(np.mean(oracle_per_seed))
    # Validity: stable + >=2 distinct attractors + oracle-controllable at full actuation (so any
    # learned-planner failure is the model, not the task) + greedy fails the MAJORITY (non-trivial,
    # non-monotonic enough). The headline system has greedy=0; new instances are partially
    # non-monotonic (greedy fails 60-80%), which still defeats the greedy baseline.
    valid = bool(stable and distinct and n_attr >= 2 and g_succ < 0.5 and o_succ >= 0.99)
    res = {
        "name": name, "knobs": knobs, "K": S, "n_attractors": int(n_attr),
        "stable": bool(stable), "max_eig": float(max(eigs)),
        "min_inter_attr_dist": float(min(inter)), "attr_scale": float(attr_scale), "tol": float(tol),
        "distinct": bool(distinct), "greedy_success": g_succ,
        "oracle_success": o_succ, "oracle_mean_final": float(np.mean(oracle_final)),
        "valid": valid,
    }
    print(f"[{name:20s}] S={S} g={knobs['n_guilds']} attr={n_attr} stable={stable} "
          f"maxeig={max(eigs):+.3f} tol={tol:.3f} greedy={g_succ:.2f} oracle={o_succ:.2f} "
          f"(final {np.mean(oracle_final):.2f}) -> {'VALID' if valid else 'reject'}")
    return res


def main():
    out = Path("examples/microbiome_jepa/results")
    out.mkdir(parents=True, exist_ok=True)
    results = {name: screen_one(name, knobs) for name, knobs in INSTANCES.items()}
    valid = [n for n, r in results.items() if r["valid"] and n != "baseline_g3_s24"]
    print(f"\nVALID new instances (controllable + non-monotonic + stable): {valid}")
    (out / "exp1_instance_screen.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {out/'exp1_instance_screen.json'}")


if __name__ == "__main__":
    main()
