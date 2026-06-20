"""Smoke test for the gLV simulator (WS5). Run with the CPU venv:

    /Users/bnz/DynaAMIcs/.venv-cpu/bin/python eb_jepa/datasets/microbiome/_smoke_glv.py

Checks (per the WS5 brief):
1. build GLVSimulator(GLVConfig()), assert attractors.shape[0] >= 2;
2. from each attractor, rollout with zero actions for T=200 stays within tol of that attractor;
3. generate_trajectories(n=8, T=20) returns states [8,21,S], actions [8,20,K], float32, finite, >=0;
4. demonstrate_non_monotonicity() reports fraction_nonmonotonic > 0 and prints the evidence.
"""

import os
import sys
import tempfile

import numpy as np

# Allow running as a plain script (resolve the package import without installing).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from eb_jepa.datasets.microbiome.glv import (  # noqa: E402
    GLVConfig,
    GLVSimulator,
    demonstrate_non_monotonicity,
)


def main() -> int:
    print("=" * 78)
    print("WS5 gLV simulator smoke test")
    print("=" * 78)

    cfg = GLVConfig()
    sim = GLVSimulator(cfg)
    S, K = sim.n_species, sim.action_dim
    print(f"n_species={S}  action_dim(K)={K}  n_guilds={cfg.n_guilds}")
    print(f"candidate_index={sim.candidate_index.tolist()}")

    # --- 1. >= 2 attractors ---
    attr = sim.attractors
    print(f"\n[1] attractors.shape = {attr.shape}")
    assert attr.shape[0] >= 2, f"need >=2 attractors, got {attr.shape[0]}"
    assert attr.shape[1] == S
    # report stability (Jacobian max real eigenvalue) and per-attractor dominant guild
    for g in range(attr.shape[0]):
        maxeig = sim.jacobian_max_real_eig(attr[g])
        dom = attr[g][sim.guild == g].mean()
        off = attr[g][sim.guild != g].max()
        print(f"    attractor {g}: jacobian_max_real_eig={maxeig:+.4f} (stable={maxeig < 0}) "
              f"dominant_guild_mean={dom:.3f} max_offguild={off:.2e}")
        assert maxeig < 0, f"attractor {g} is NOT locally stable (maxeig={maxeig})"
    print("    OK: >=2 attractors, all locally stable.")

    # --- 2. zero-action rollout stays at each attractor (T=200) ---
    print("\n[2] zero-action rollout (T=200) drift from each attractor:")
    zero_actions = np.zeros((200, K), dtype=np.float64)
    max_drift = 0.0
    for g in range(attr.shape[0]):
        traj = sim.rollout(attr[g], zero_actions)
        drift = float(np.linalg.norm(traj[-1] - attr[g]))
        max_drift = max(max_drift, drift)
        print(f"    attractor {g}: ||state[200] - attractor|| = {drift:.3e}")
    tol_stay = 1e-3
    assert max_drift < tol_stay, f"attractor not stable under zero action: max drift {max_drift} >= {tol_stay}"
    print(f"    OK: max drift {max_drift:.3e} < {tol_stay}.")

    # --- 3. generate_trajectories shapes/dtypes/finiteness/non-negativity ---
    print("\n[3] generate_trajectories(n=8, T=20):")
    batch = sim.generate_trajectories(n=8, T=20, seed=123)
    states, actions = batch["states"], batch["actions"]
    print(f"    states.shape={states.shape} dtype={states.dtype}")
    print(f"    actions.shape={actions.shape} dtype={actions.dtype}")
    assert states.shape == (8, 21, S), states.shape
    assert actions.shape == (8, 20, K), actions.shape
    assert states.dtype == np.float32 and actions.dtype == np.float32
    assert np.isfinite(states).all() and np.isfinite(actions).all(), "non-finite values"
    assert (states >= 0).all(), "states must be non-negative"
    print("    OK: shapes, dtype=float32, finite, states >= 0.")

    # determinism check (same seed -> identical batch)
    batch2 = sim.generate_trajectories(n=8, T=20, seed=123)
    assert np.array_equal(states, batch2["states"]) and np.array_equal(actions, batch2["actions"])
    print("    OK: deterministic given seed.")

    # --- 4. non-monotonicity demonstration ---
    print("\n[4] demonstrate_non_monotonicity():")
    # write the optional figure to a temp dir so the smoke test does not pollute the source tree.
    png = os.path.join(tempfile.gettempdir(), "glv_nonmonotonic.png")
    evidence = demonstrate_non_monotonicity(cfg, png_path=png)
    frac = evidence["fraction_nonmonotonic"]
    print(f"    fraction_nonmonotonic (greedy fails, detour succeeds) = "
          f"{evidence['n_nonmonotonic']}/{evidence['n_pairs']} = {frac:.3f}")
    print(f"    fraction_strict_moveaway (distance peaks above start)  = "
          f"{evidence['n_strict_moveaway']}/{evidence['n_pairs']} = {evidence['fraction_strict_moveaway']:.3f}")
    print(f"    reach tol = {evidence['tol']:.3f}  (attractor scale = {evidence['attractor_scale']:.3f})")
    print("    per-pair evidence (src->tgt):")
    for p in evidence["per_pair"]:
        s, t = p["pair"]
        print(f"      {s}->{t} via gate guild {p['gate_guild']}: "
              f"greedy_min={p['greedy_min_dist']:.2f}(reach={p['greedy_reaches']}) "
              f"detour_min={p['detour_min_dist']:.2f}(reach={p['detour_reaches']}) "
              f"start={p['start_dist']:.2f} detour_peak={p['detour_peak_before_min']:.2f} "
              f"req_increase={p['required_distance_increase']:+.2f} "
              f"NONMONO={p['is_nonmonotonic']} STRICT_MOVEAWAY={p['is_strict_moveaway']}")
    if "pair" in evidence:
        print(f"    headline non-monotonic pair: {evidence['pair']} "
              f"greedy_min_dist={evidence['greedy_min_dist']:.3f} "
              f"optimal(detour)_min_dist={evidence['optimal_min_dist']:.3f} "
              f"required_distance_increase={evidence['required_distance_increase']:+.3f}")
    if "png_path" in evidence:
        print(f"    saved figure: {evidence['png_path']}")
    assert frac > 0, "fraction_nonmonotonic must be > 0 (dynamics must be non-monotonic)"
    print("    OK: fraction_nonmonotonic > 0.")

    print("\n" + "=" * 78)
    print("ALL SMOKE CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
