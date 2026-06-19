"""
BIG BET — CPU gate. Re-run the planning ORACLE (MPPI on the TRUE gLV dynamics, i.e. a PERFECT model)
across a sweep of action-panel sizes K. The M3 diagnosis showed K=6 is structurally uncontrollable
(oracle 0% success, ~4.0 final dist, even with 4x actions + 3x horizon). The big-bet question:

    Does a LARGER candidate panel K make the task controllable for a PERFECT planner?

K is a pure actuation knob: the gLV attractors depend only on the guild structure / interaction
matrix, NOT on n_candidate. So `tol` and the target states are IDENTICAL across the sweep, and only
the number of dose-able species (and the action dimension) changes. That makes this a clean
controllability curve: at a fixed per-step action budget, how many actuators does a perfect planner
need to reach the target attractor?

Decision rule for the bet:
  * If a larger K reaches the target (oracle success > 0): the task is controllable -> retrain the
    world model at that K and re-run the LEARNED latent-MPPI (M3 may flip to diagnosed-AND-fixed).
  * If even K = n_species stays ~0%: the negative is a fundamental property of THIS gLV/horizon/tol
    spec, not just the small panel -> report that stronger, more complete characterization.

Pure numpy + the gLV sim. No torch, no checkpoint. MPPI is vectorized over samples on the TRUE
dynamics, matching diagnose_planning._oracle_mppi_action exactly (so K=6 must reproduce ~4.0).

Run (local CPU):  .venv-cpu/bin/python -m examples.microbiome_jepa.oracle_K_sweep
INTEGRITY: every number printed/saved is measured from this run; nothing fabricated.
"""

import json
from itertools import permutations
from pathlib import Path

import fire
import numpy as np

from eb_jepa.datasets.microbiome.glv import GLVConfig, GLVSimulator


def _attr_scale(attractors):
    d = [np.linalg.norm(attractors[i] - attractors[j])
         for i, j in permutations(range(len(attractors)), 2)]
    return float(np.mean(d))


def _batched_true_step(sim, X, actions):
    """Batched TRUE gLV step. X [N,S], actions [N,K] -> Xn [N,S].

    Mirrors GLVSimulator._step_from exactly (apply clipped action delta to the candidate species,
    then RK4-integrate steps_per_action of dt), but over a batch of N states at once. Deterministic
    (process noise is omitted in planning rollouts — this is the perfect-model oracle).
    """
    cfg = sim.config
    cand = sim._candidate_index
    Amat = sim._A      # [S,S]
    r = sim._r         # [S]
    m = cfg.immigration
    amax = cfg.action_max
    dt = cfg.dt

    a = np.clip(actions, -amax, amax)
    X = X.copy()
    X[:, cand] = np.maximum(X[:, cand] + a, 0.0)

    def deriv(Xb):
        # (A @ x)_i per row == (Xb @ A.T); + immigration m
        return Xb * (r[None] + Xb @ Amat.T) + m

    for _ in range(cfg.steps_per_action):
        k1 = deriv(X)
        k2 = deriv(np.maximum(X + 0.5 * dt * k1, 0.0))
        k3 = deriv(np.maximum(X + 0.5 * dt * k2, 0.0))
        k4 = deriv(np.maximum(X + dt * k3, 0.0))
        X = np.clip(X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1e6)
    return X


def _oracle_mppi_action(sim, x0, target, action_max, K, H, n_samples, n_elites, n_iters,
                        init_std, temperature, rng):
    """Vectorized state-space MPPI on the TRUE gLV dynamics. Returns planned first action [K].

    Identical objective to diagnose_planning._oracle_mppi_action (cumulative true-state L2 cost,
    softmax-weighted elite refinement), just batched over samples for speed.
    """
    mean = np.zeros((H, K), dtype=np.float32)
    std = np.full((H, K), init_std, dtype=np.float32)
    for _ in range(n_iters):
        noise = rng.standard_normal((n_samples, H, K)).astype(np.float32)
        actions = np.clip(mean[None] + std[None] * noise, -action_max, action_max)  # [n,H,K]
        X = np.tile(x0[None], (n_samples, 1)).astype(np.float64)                     # [n,S]
        costs = np.zeros(n_samples, dtype=np.float64)
        for t in range(H):
            X = _batched_true_step(sim, X, actions[:, t])
            costs += np.linalg.norm(X - target[None], axis=1)                        # cumulative
        elite_idx = np.argsort(costs)[:n_elites]
        ec = costs[elite_idx]
        w = np.exp(temperature * (costs.min() - ec))
        w = w / (w.sum() + 1e-9)
        ea = actions[elite_idx]                                                      # [n_el,H,K]
        mean = (w[:, None, None] * ea).sum(0).astype(np.float32)
        var = (w[:, None, None] * (ea - mean[None]) ** 2).sum(0)
        std = np.sqrt(var).clip(min=0.02).astype(np.float32)
    return np.clip(mean[0], -action_max, action_max)


def _run_one_K(K, n_species, n_guilds, action_max, horizon, mpc_steps, n_samples, n_elites,
               n_iters, init_std, temperature, tol_frac, seed):
    cfg = GLVConfig(n_species=n_species, n_candidate=K, n_guilds=n_guilds, action_max=action_max)
    sim = GLVSimulator(cfg)
    attractors = sim.attractors
    attr_scale = _attr_scale(attractors)
    tol = tol_frac * attr_scale
    pairs = list(permutations(range(n_guilds), 2))
    rng = np.random.default_rng(seed)

    n_succ, finals, starts, per_pair = 0, [], [], []
    for (src, tgt) in pairs:
        target = attractors[tgt]
        x = sim.reset(attractor=src).astype(np.float64)
        start = float(np.linalg.norm(x - target))
        starts.append(start)
        reached = False
        for _ in range(mpc_steps):
            a = _oracle_mppi_action(sim, x, target, action_max, K, horizon, n_samples, n_elites,
                                    n_iters, init_std, temperature, rng)
            x = sim.step(a).astype(np.float64)
            if np.linalg.norm(x - target) < tol:
                reached = True
                break
        fd = float(np.linalg.norm(x - target))
        finals.append(fd)
        n_succ += int(reached)
        per_pair.append({"src": int(src), "tgt": int(tgt), "start": start, "final": fd,
                         "reached": bool(reached)})
    return {
        "K": int(K), "n_species": int(n_species), "n_candidate": int(K),
        "action_max": float(action_max), "horizon": int(horizon), "mpc_steps": int(mpc_steps),
        "tol": float(tol), "attr_scale": float(attr_scale),
        "success_rate": n_succ / len(pairs), "n_success": int(n_succ), "n_pairs": len(pairs),
        "mean_start_dist": float(np.mean(starts)), "mean_final_dist": float(np.mean(finals)),
        "per_pair": per_pair,
    }


def _mean_se(vals):
    a = np.asarray(vals, dtype=float)
    n = len(a)
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)


def _parse_seeds(seeds):
    if isinstance(seeds, (list, tuple)):
        return [int(s) for s in seeds]
    if isinstance(seeds, int):
        return [seeds]
    return [int(s) for s in str(seeds).strip("() ").split(",") if str(s).strip() != ""]


def run(
    K_list=(6, 9, 12, 15, 18, 21, 24),
    n_species: int = 24,
    n_guilds: int = 3,
    action_max: float = 0.5,
    horizon: int = 6,
    mpc_steps: int = 20,
    n_samples: int = 96,
    n_elites: int = 16,
    n_iters: int = 3,
    init_std: float = 0.25,
    temperature: float = 1.0,
    tol_frac: float = 0.15,
    seeds="0,1,2",
    out: str = "examples/microbiome_jepa/results",
    tag: str = "",
):
    if isinstance(K_list, (int, float)):
        K_list = [int(K_list)]
    else:
        K_list = [int(k) for k in K_list]
    seed_list = _parse_seeds(seeds)

    print(f"[oracle-K] n_species={n_species} n_guilds={n_guilds} action_max={action_max} "
          f"horizon={horizon} mpc_steps={mpc_steps} n_samples={n_samples} tol_frac={tol_frac} "
          f"seeds={seed_list}")
    print(f"[oracle-K] sweeping K in {K_list}  (K=6 should reproduce the committed diagnosis ~4.0)")

    # rows[K] aggregates over seeds; each seed is one independent MPPI RNG.
    rows = []
    for K in K_list:
        per_seed = [_run_one_K(K, n_species, n_guilds, action_max, horizon, mpc_steps, n_samples,
                               n_elites, n_iters, init_std, temperature, tol_frac, seed)
                    for seed in seed_list]
        sr_m, sr_se = _mean_se([r["success_rate"] for r in per_seed])
        fd_m, fd_se = _mean_se([r["mean_final_dist"] for r in per_seed])
        rows.append({
            "K": int(K), "tol": per_seed[0]["tol"], "attr_scale": per_seed[0]["attr_scale"],
            "mean_start_dist": per_seed[0]["mean_start_dist"],
            "success_rate_mean": sr_m, "success_rate_se": sr_se,
            "mean_final_dist_mean": fd_m, "mean_final_dist_se": fd_se,
            "per_seed_success_rate": [r["success_rate"] for r in per_seed],
            "per_seed_mean_final_dist": [r["mean_final_dist"] for r in per_seed],
            "n_pairs": per_seed[0]["n_pairs"],
        })

    tol0 = rows[0]["tol"]
    same_tol = all(abs(r["tol"] - tol0) < 1e-9 for r in rows)
    print(f"\n  K    succ_rate (mean±se)   mean_final (mean±se)   "
          f"(start {rows[0]['mean_start_dist']:.2f}, tol {tol0:.3f}"
          f"{'' if same_tol else ' [WARN tol varies]'})")
    print("  " + "-" * 64)
    for r in rows:
        print(f"  {r['K']:>2}   {r['success_rate_mean']:.3f} ± {r['success_rate_se']:.3f}        "
              f"{r['mean_final_dist_mean']:.3f} ± {r['mean_final_dist_se']:.3f}")

    best = max(rows, key=lambda r: (r["success_rate_mean"], -r["mean_final_dist_mean"]))
    controllable = best["success_rate_mean"] > 0.0
    # smallest K with any success (the controllability onset)
    onset = next((r["K"] for r in rows if r["success_rate_mean"] > 0.0), None)
    verdict = ("CONTROLLABLE: onset at K=%d, full success at K=%d -> retrain world model at K=%d"
               % (onset, best["K"], best["K"])) if controllable else \
              ("UNCONTROLLABLE even at K=%d (=all species): the negative is fundamental to this "
               "gLV/horizon/tol spec" % max(r["K"] for r in rows))
    print(f"\n[oracle-K] VERDICT: {verdict}")

    res = {"rows": rows, "best_K": best["K"], "best_success_rate": best["success_rate_mean"],
           "controllability_onset_K": onset, "controllable": controllable, "verdict": verdict,
           "same_tol_across_K": same_tol,
           "settings": {"n_species": n_species, "n_guilds": n_guilds, "action_max": action_max,
                        "horizon": horizon, "mpc_steps": mpc_steps, "n_samples": n_samples,
                        "n_elites": n_elites, "n_iters": n_iters, "tol_frac": tol_frac,
                        "seeds": seed_list}}
    Path(out).mkdir(parents=True, exist_ok=True)
    fn = Path(out) / (f"oracle_K_sweep{('_' + tag) if tag else ''}.json")
    with open(fn, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[oracle-K] saved -> {fn}")

    # ---- figure: controllability curve (oracle success + final dist vs K) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ks = [r["K"] for r in rows]
        sr = [r["success_rate_mean"] for r in rows]
        sr_se = [r["success_rate_se"] for r in rows]
        fd = [r["mean_final_dist_mean"] for r in rows]
        fd_se = [r["mean_final_dist_se"] for r in rows]
        start = rows[0]["mean_start_dist"]

        fig, ax1 = plt.subplots(figsize=(7, 4.3))
        c1, c2 = "#2a7", "#c84"
        ax1.errorbar(Ks, sr, yerr=sr_se, marker="o", lw=2, color=c1, capsize=3,
                     label="oracle success rate")
        ax1.set_xlabel("action panel size K  (dose-able species; n_species=%d)" % n_species)
        ax1.set_ylabel("oracle success rate", color=c1)
        ax1.set_ylim(-0.03, 1.05)
        ax1.tick_params(axis="y", labelcolor=c1)
        ax1.axhline(0.0, ls=":", c="gray", lw=0.8)

        ax2 = ax1.twinx()
        ax2.errorbar(Ks, fd, yerr=fd_se, marker="s", lw=2, color=c2, capsize=3, ls="--",
                     label="mean final distance")
        ax2.axhline(tol0, ls=":", c=c2, lw=1.0)
        ax2.axhline(start, ls="-.", c="gray", lw=0.8)
        ax2.set_ylabel("mean final dist to target  (tol=%.2f, start=%.2f)" % (tol0, start),
                       color=c2)
        ax2.tick_params(axis="y", labelcolor=c2)

        ax1.set_title("gLV planning controllability vs actuation (PERFECT model / true dynamics)\n"
                      "task is uncontrollable until ~all species are dose-able — root cause of the "
                      "M3 negative")
        ax1.set_xticks(Ks)
        fig.tight_layout()
        fig_path = Path(out) / (f"oracle_K_sweep{('_' + tag) if tag else ''}.png")
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        print(f"[oracle-K] figure -> {fig_path}")
        res["figure"] = str(fig_path)
    except Exception as e:
        print(f"[oracle-K] figure skipped: {type(e).__name__}: {e}")

    return res


if __name__ == "__main__":
    fire.Fire(run)
