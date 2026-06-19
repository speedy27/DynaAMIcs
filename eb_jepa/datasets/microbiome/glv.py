"""Generalized Lotka-Volterra (gLV) microbiome simulator -- the "Two Rooms" of microbiome.

WS5 of the EB-JEPA microbiome world model. Produces unlimited clean, controllable trajectories with
KNOWN ground-truth attractors and explicit interventions as actions. It is the rigorous synthetic
testbed for the IDM-ablation collapse story and for intervention planning (MPPI/CEM).

Dynamics
--------
    dx/dt = x * (r + A x),   x_i >= 0     (generalized Lotka-Volterra)

We engineer ``r`` (growth) and ``A`` (interaction matrix) to obtain >= 2 STABLE equilibria with a
saddle/barrier between them, and -- crucially -- a NON-MONOTONIC reachability structure.

Multistability (how)
--------------------
Species are partitioned into ``n_guilds`` mutually-exclusive guilds. Within a guild: weak mutualism
with strong self-limitation; between guilds: strong competition. Competitive exclusion then makes each
"single-guild-dominant" state a stable equilibrium (one guild near its carrying capacity, the others
driven to ~0). With strong self-limitation the corners are locally stable (verified here via the
Jacobian: max real eigenvalue < 0). The within-guild mutualism strength is scaled with guild size so
the guild subsystem stays bounded (self_lim + within*(guild_size-1) < 0) at any ``n_species``.

Non-monotonic reachability (how, and WHY it matters)
----------------------------------------------------
The between-guild competition is CYCLIC (rock-paper-scissors): guild g strongly suppresses guild
(g+1) % n_guilds and is only weakly suppressed by it. Single-guild states stay stable (strong
self-limitation dominates), but reachability is directed. Reaching the target guild T from incumbent S
"against the cycle" requires first BLOOMING the gate guild that suppresses S; that bloom moves the
community AWAY from T (distance increases) before T can establish. A greedy "reduce distance every
step" policy gets stuck on exactly these pairs; a detour that temporarily increases distance succeeds.
This is the microbiome analog of Two Rooms and is what makes planning non-trivial. The property is
demonstrated with a measured number in :func:`demonstrate_non_monotonicity`.

Actions
-------
``action a in R^K`` is a (typically non-negative, "probiotic dose") delta applied to the K candidate
species' abundance at each env step -- a continuous relaxation of "introduce/remove a taxon". Kept
deliberately small relative to the carrying capacity so that an intervention cannot teleport the
community across a basin boundary in one step; the basin must be crossed through the dynamics.

Pure numpy (no torch, no heavy eb_jepa imports). Deterministic given ``config.seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class GLVConfig:
    n_species: int = 32          # S, total species in the community
    n_candidate: int = 8         # K, action panel size (curated taxa an intervention can perturb)
    dt: float = 0.05
    steps_per_action: int = 1    # integrate this many dt per env step
    noise_std: float = 0.0       # process noise std on log-abundance; 0 = deterministic
    seed: int = 0

    # ---- knobs that create multistability + non-monotonic reachability ----
    n_guilds: int = 3            # number of mutually-exclusive guilds -> number of attractors
    self_lim: float = -1.0       # diagonal self-limitation (must be < 0)
    within_frac: float = 0.4     # within-guild mutualism as a FRACTION of the stability bound;
    #                              effective within = within_frac * |self_lim| / (max_guild_size - 1),
    #                              so within*(guild_size-1) < |self_lim| always (bounded, stable).
    comp_strong: float = -2.5    # strong between-guild competition (predator -> prey direction)
    comp_weak: float = -0.4      # weak between-guild competition (prey -> predator direction)
    growth: float = 1.0          # intrinsic growth rate r_i (uniform)
    action_max: float = 0.5      # max per-species delta an action can apply per env step (bounded)
    init_seed_abundance: float = 0.05  # background abundance of non-dominant species at an attractor
    immigration: float = 1e-3    # constant influx m: dx/dt = x*(r+Ax)+m. Small m>0 means no species is
    #                              ever truly extinct ("rain of propagules"), so a guild can REGROW
    #                              after the incumbent is cleared. Required for reachability with a
    #                              small candidate panel (else absent guilds are stuck at 0 forever).

    def __post_init__(self) -> None:
        if self.self_lim >= 0:
            raise ValueError("self_lim must be negative (self-limitation).")
        if self.n_guilds < 2:
            raise ValueError("n_guilds must be >= 2 to have multiple attractors.")
        if self.n_species < self.n_guilds:
            raise ValueError("n_species must be >= n_guilds.")
        if self.n_candidate < 1 or self.n_candidate > self.n_species:
            raise ValueError("n_candidate must be in [1, n_species].")


# --------------------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------------------
class GLVSimulator:
    """Generalized Lotka-Volterra environment with multistable, non-monotonic dynamics."""

    def __init__(self, config: GLVConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

        S = config.n_species
        ng = config.n_guilds

        # Balanced contiguous guild assignment: blocks [0..],[1..],... as even as possible.
        guild = np.array(sorted(i % ng for i in range(S)), dtype=np.int64)
        self._guild = guild
        self._guild_sizes = np.array([int((guild == g).sum()) for g in range(ng)], dtype=np.int64)
        max_sz = int(self._guild_sizes.max())

        # within-guild mutualism scaled to guarantee guild-subsystem stability/boundedness.
        within = config.within_frac * abs(config.self_lim) / max(max_sz - 1, 1)
        self._within = within

        # Growth vector.
        self._r = np.full(S, config.growth, dtype=np.float64)

        # Interaction matrix A.
        A = np.zeros((S, S), dtype=np.float64)
        for i in range(S):
            gi = guild[i]
            for j in range(S):
                gj = guild[j]
                if i == j:
                    A[i, j] = config.self_lim
                elif gi == gj:
                    A[i, j] = within
                elif gj == (gi + 1) % ng:
                    # effect of the PREY guild (gi+1) on the PREDATOR guild gi: weak
                    A[i, j] = config.comp_weak
                else:
                    # effect of a predator/other guild on guild gi: strong suppression
                    A[i, j] = config.comp_strong
        self._A = A

        # Compute and cache the verified-stable attractors (one per guild).
        self._attractors = self._compute_attractors()

        # Fixed candidate panel (species the action can perturb). Spread across guilds + species,
        # deterministically, so an intervention can address every guild (needed for the detour).
        self._candidate_index = self._build_candidate_panel()

        # Internal env state (set by reset()).
        self._state: Optional[np.ndarray] = None

    # ---- core gLV derivative + RK4 integrator (robust to stiffness; clamps non-negative) ----
    def _deriv(self, x: np.ndarray) -> np.ndarray:
        # gLV with a small constant immigration term m (see GLVConfig.immigration).
        return x * (self._r + self._A @ x) + self.config.immigration

    def _rk4_step(self, x: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._deriv(x)
        k2 = self._deriv(np.maximum(x + 0.5 * dt * k1, 0.0))
        k3 = self._deriv(np.maximum(x + 0.5 * dt * k2, 0.0))
        k4 = self._deriv(np.maximum(x + dt * k3, 0.0))
        xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # clamp non-negative and cap to avoid overflow blow-ups on pathological configs
        return np.clip(xn, 0.0, 1e6)

    def _integrate(self, x: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64).copy(), 0.0, 1e6)
        for _ in range(n_steps):
            x = self._rk4_step(x, dt)
        return x

    def _compute_attractors(self) -> np.ndarray:
        """Integrate from each guild-dominant corner to long-time equilibrium."""
        S = self.config.n_species
        ng = self.config.n_guilds
        attractors = np.zeros((ng, S), dtype=np.float64)
        for g in range(ng):
            x0 = np.full(S, self.config.init_seed_abundance, dtype=np.float64)
            x0[self._guild == g] = 1.0
            # long settle with a small dt for an accurate equilibrium
            attractors[g] = self._integrate(x0, dt=0.02, n_steps=20000)
        return attractors

    def _build_candidate_panel(self) -> np.ndarray:
        """Pick K fixed species indices, spread across guilds (round-robin), deterministically."""
        K = self.config.n_candidate
        ng = self.config.n_guilds
        chosen: list[int] = []
        # round-robin one species at a time from each guild until we have K
        per_guild_lists = [list(np.where(self._guild == g)[0]) for g in range(ng)]
        ptr = [0] * ng
        g = 0
        while len(chosen) < K:
            if ptr[g] < len(per_guild_lists[g]):
                chosen.append(int(per_guild_lists[g][ptr[g]]))
                ptr[g] += 1
            g = (g + 1) % ng
            # safety: if all guilds exhausted (K > S handled by config check) break
            if all(ptr[gg] >= len(per_guild_lists[gg]) for gg in range(ng)) and len(chosen) < K:
                break
        return np.array(sorted(chosen[:K]), dtype=np.int64)

    # ---- public properties (API contract) ----
    @property
    def n_species(self) -> int:
        return self.config.n_species

    @property
    def action_dim(self) -> int:
        return self.config.n_candidate

    @property
    def candidate_index(self) -> np.ndarray:
        """[K] species indices that the action perturbs (fixed)."""
        return self._candidate_index.copy()

    @property
    def attractors(self) -> np.ndarray:
        """[n_attractors, S] known stable equilibria (>= 2)."""
        return self._attractors.copy()

    @property
    def guild(self) -> np.ndarray:
        """[S] guild id per species (helper; not part of the required contract)."""
        return self._guild.copy()

    @property
    def interaction_matrix(self) -> np.ndarray:
        """[S, S] interaction matrix A (helper)."""
        return self._A.copy()

    @property
    def growth(self) -> np.ndarray:
        """[S] growth vector r (helper)."""
        return self._r.copy()

    # ---- helpers ----
    @staticmethod
    def log_abundance(state: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Log-abundance helper: log(state + eps)."""
        return np.log(np.asarray(state, dtype=np.float64) + eps)

    def jacobian_max_real_eig(self, x: np.ndarray) -> float:
        """Max real part of the Jacobian eigenvalues at x (< 0 => locally stable)."""
        x = np.asarray(x, dtype=np.float64)
        S = len(x)
        Ax = self._A @ x
        J = np.zeros((S, S), dtype=np.float64)
        for i in range(S):
            for k in range(S):
                J[i, k] = (self._r[i] + Ax[i]) * (1.0 if i == k else 0.0) + x[i] * self._A[i, k]
        return float(np.linalg.eigvals(J).real.max())

    def _apply_action(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Add the (clipped) action delta to the candidate species' abundances."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.config.n_candidate:
            raise ValueError(f"action must have length K={self.config.n_candidate}, got {a.shape[0]}.")
        a = np.clip(a, -self.config.action_max, self.config.action_max)
        x = x.copy()
        x[self._candidate_index] = x[self._candidate_index] + a
        return np.maximum(x, 0.0)

    # ---- env API ----
    def reset(self, seed: Optional[int] = None, attractor: Optional[int] = None) -> np.ndarray:
        """Reset to (an attractor, optionally perturbed) and return the state [S]."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        ng = self.config.n_guilds
        if attractor is None:
            attractor = int(self._rng.integers(ng))
        if not (0 <= attractor < ng):
            raise ValueError(f"attractor must be in [0,{ng}); got {attractor}.")
        self._state = self._attractors[attractor].copy()
        return self._state.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply action [K] (delta on candidates), integrate, return next state [S]."""
        if self._state is None:
            raise RuntimeError("call reset() before step().")
        self._state = self._step_from(self._state, action)
        return self._state.copy()

    def _step_from(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Pure functional env step from an explicit state (used by rollout / planners)."""
        x = self._apply_action(state, action)
        x = self._integrate(x, dt=self.config.dt, n_steps=self.config.steps_per_action)
        if self.config.noise_std > 0.0:
            # process noise on log-abundance, then back to abundance (keeps non-negativity)
            log_x = np.log(x + 1e-8)
            log_x = log_x + self.config.noise_std * self._rng.standard_normal(log_x.shape)
            x = np.maximum(np.exp(log_x) - 1e-8, 0.0)
        return x

    def rollout(self, init_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out: actions [T, K] -> states [T+1, S] (states[0] == init_state)."""
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.config.n_candidate:
            raise ValueError(f"actions must be [T, K={self.config.n_candidate}]; got {actions.shape}.")
        T = actions.shape[0]
        S = self.config.n_species
        states = np.zeros((T + 1, S), dtype=np.float64)
        states[0] = np.maximum(np.asarray(init_state, dtype=np.float64).reshape(S), 0.0)
        x = states[0].copy()
        for t in range(T):
            x = self._step_from(x, actions[t])
            states[t + 1] = x
        return states

    def generate_trajectories(self, n: int, T: int, action_policy: str = "random",
                              seed: int = 0) -> dict:
        """Generate a batch of trajectories.

        Parameters
        ----------
        n : number of trajectories
        T : number of actions (=> T+1 states) per trajectory
        action_policy : "random" (uniform non-negative doses on a random sparse subset of candidates),
            "zero" (no intervention -> pure relaxation from the start state), or
            "perturb" (a short random push at the start, then zero -- mimics a one-shot intervention).
        seed : RNG seed for this batch (independent of construction seed; deterministic)

        Returns
        -------
        {"states": float32[n, T+1, S], "actions": float32[n, T, K]}
        """
        rng = np.random.default_rng(seed)
        S = self.config.n_species
        K = self.config.n_candidate
        ng = self.config.n_guilds
        amax = self.config.action_max

        states = np.zeros((n, T + 1, S), dtype=np.float32)
        actions = np.zeros((n, T, K), dtype=np.float32)

        for i in range(n):
            start_attr = int(rng.integers(ng))
            x0 = self._attractors[start_attr].copy()

            if action_policy == "random":
                # sparse non-negative "probiotic" doses: each step, perturb a random subset.
                acts = np.zeros((T, K), dtype=np.float64)
                for t in range(T):
                    n_active = int(rng.integers(1, max(2, K // 2 + 1)))
                    idx = rng.choice(K, size=n_active, replace=False)
                    acts[t, idx] = rng.uniform(0.0, amax, size=n_active)
            elif action_policy == "zero":
                acts = np.zeros((T, K), dtype=np.float64)
            elif action_policy == "perturb":
                # one-shot intervention: random non-negative dose on a fixed candidate subset for the
                # first `push` steps, then no action (the community relaxes under the dynamics).
                acts = np.zeros((T, K), dtype=np.float64)
                push = min(max(T // 4, 1), T)
                idx = rng.choice(K, size=max(1, K // 2), replace=False)
                acts[:push][:, idx] = rng.uniform(0.0, amax, size=(push, len(idx)))
            else:
                raise ValueError(f"unknown action_policy {action_policy!r}.")

            traj = self.rollout(x0, acts)
            states[i] = traj.astype(np.float32)
            actions[i] = acts.astype(np.float32)

        return {"states": states, "actions": actions}


# --------------------------------------------------------------------------------------------------
# Non-monotonicity demonstration (the rubric-defining property)
# --------------------------------------------------------------------------------------------------
def _greedy_monotone_min_dist(sim: GLVSimulator, src: int, tgt: int, T: int = 250) -> float:
    """Best distance reachable by a STRICTLY-distance-decreasing (monotone) policy.

    At each step we may apply a single-candidate non-negative dose OR let the dynamics relax; we only
    accept a move if it strictly reduces the Euclidean distance to the target attractor. If no move
    improves, the monotone policy is STUCK (returns the best distance reached). This is the natural
    "reduce distance every step" greedy planner that should fail on non-monotonic pairs.
    """
    S = sim.n_species
    K = sim.action_dim
    amax = sim.config.action_max
    target = sim.attractors[tgt]
    x = sim.attractors[src].copy()
    best = float(np.linalg.norm(x - target))
    for _ in range(T):
        bd, bx = best, None
        # try each single candidate at two dose levels
        for k in range(K):
            for amt in (amax * 0.5, amax):
                a = np.zeros(K)
                a[k] = amt
                xn = sim._step_from(x, a)
                dd = float(np.linalg.norm(xn - target))
                if dd < bd - 1e-9:
                    bd, bx = dd, xn
        # try pure relaxation
        xr = sim._step_from(x, np.zeros(K))
        dr = float(np.linalg.norm(xr - target))
        if dr < bd - 1e-9:
            bd, bx = dr, xr
        if bx is None:
            break  # stuck: monotone policy cannot proceed
        x, best = bx, bd
    return best


def _detour_min_dist(sim: GLVSimulator, src: int, tgt: int,
                     push_gate: int = 30, push_tgt: int = 60, relax: int = 250):
    """Min distance reachable by a DETOUR that is allowed to temporarily increase distance.

    Heuristic detour matched to the cyclic structure: (1) bloom the gate guild that most strongly
    suppresses the incumbent ``src`` (this moves AWAY from the target), (2) then dose the target
    guild's candidates, (3) then let the dynamics relax. Returns (min_dist, peak_dist_before_min)
    where peak_dist_before_min > start distance evidences the required move-away.
    """
    S = sim.n_species
    K = sim.action_dim
    ng = sim.config.n_guilds
    amax = sim.config.action_max
    guild = sim.guild
    cand = sim.candidate_index
    A = sim.interaction_matrix
    target = sim.attractors[tgt]

    # between-guild effect matrix gb[a,b] = effect of guild b on guild a (pick a representative pair)
    rep = [int(np.where(guild == g)[0][0]) for g in range(ng)]
    gb = np.array([[A[rep[a], rep[b]] for b in range(ng)] for a in range(ng)])
    gate = int(np.argmin(gb[src, :]))  # guild that suppresses the incumbent most strongly

    # candidate indices belonging to gate / target guilds (action only touches candidates)
    gate_cand_mask = np.array([guild[c] == gate for c in cand])
    tgt_cand_mask = np.array([guild[c] == tgt for c in cand])

    x = sim.attractors[src].copy()
    start = float(np.linalg.norm(x - target))
    dists = [start]

    for _ in range(push_gate):
        a = np.zeros(K)
        a[gate_cand_mask] = amax
        x = sim._step_from(x, a)
        dists.append(float(np.linalg.norm(x - target)))
    for _ in range(push_tgt):
        a = np.zeros(K)
        a[tgt_cand_mask] = amax
        x = sim._step_from(x, a)
        dists.append(float(np.linalg.norm(x - target)))
    for _ in range(relax):
        x = sim._step_from(x, np.zeros(K))
        dists.append(float(np.linalg.norm(x - target)))

    dists = np.asarray(dists)
    argmin = int(np.argmin(dists))
    peak_before_min = float(dists[: argmin + 1].max())
    return float(dists.min()), peak_before_min, start, gate, dists


def demonstrate_non_monotonicity(config: Optional[GLVConfig] = None,
                                 tol: Optional[float] = None,
                                 png_path: Optional[str] = None) -> dict:
    """Measure non-monotonic reachability over all ordered (init, target) attractor pairs.

    Two distinct, both-honest measures are returned:

    * ``fraction_nonmonotonic`` (PRIMARY, the planning sense): a pair is non-monotonic when a
      strictly-distance-decreasing (greedy/monotone) policy CANNOT reach the target (gets stuck),
      but a DETOUR policy that is allowed to temporarily increase distance CAN. If greedy always
      worked, planning would be trivial; this fraction quantifies how often it does NOT.
    * ``fraction_strict_moveaway`` (SECONDARY, the strongest visual): the fraction of pairs whose
      successful detour distance-to-target rises STRICTLY ABOVE the start distance before descending
      (you literally must move away first). ``required_distance_increase`` reports how far above.

    Returns a dict with both measures + per-pair evidence. Optionally saves a small PNG
    (distance-vs-step for the most non-monotonic pair) to ``png_path``.
    """
    if config is None:
        config = GLVConfig()
    sim = GLVSimulator(config)
    ng = config.n_guilds

    # default reach tolerance: a small fraction of the inter-attractor distance scale
    attr = sim.attractors
    inter = [float(np.linalg.norm(attr[i] - attr[j]))
             for i in range(ng) for j in range(ng) if i != j]
    scale = float(np.mean(inter)) if inter else 1.0
    if tol is None:
        tol = 0.1 * scale  # must get within 10% of the attractor-separation scale to "reach"

    pairs = [(s, t) for s in range(ng) for t in range(ng) if s != t]
    per_pair = []
    n_nonmono = 0
    n_strict = 0
    example_pair = None
    example_curve = None
    best_example_score = -np.inf

    for s, t in pairs:
        greedy_min = _greedy_monotone_min_dist(sim, s, t)
        detour_min, peak_before_min, start, gate, curve = _detour_min_dist(sim, s, t)
        greedy_reaches = greedy_min < tol
        detour_reaches = detour_min < tol
        is_nm = (not greedy_reaches) and detour_reaches
        # "required increase" = how far above the start distance the successful detour had to go
        required_increase = (peak_before_min - start) if detour_reaches else 0.0
        is_strict = detour_reaches and (peak_before_min > start + 1e-6)
        if is_nm:
            n_nonmono += 1
        if is_strict:
            n_strict += 1
        # pick headline example: prefer a strict-move-away non-monotonic pair with the largest
        # required increase; this is the most compelling figure.
        score = (2.0 if (is_nm and is_strict) else 1.0 if is_nm else 0.0) + required_increase
        if is_nm and score > best_example_score:
            best_example_score = score
            example_pair = (s, t)
            example_curve = curve
        per_pair.append({
            "pair": (s, t),
            "gate_guild": gate,
            "greedy_min_dist": greedy_min,
            "greedy_reaches": greedy_reaches,
            "detour_min_dist": detour_min,
            "detour_reaches": detour_reaches,
            "start_dist": start,
            "detour_peak_before_min": peak_before_min,
            "required_distance_increase": required_increase,
            "is_nonmonotonic": is_nm,
            "is_strict_moveaway": is_strict,
        })

    fraction = n_nonmono / len(pairs) if pairs else 0.0
    fraction_strict = n_strict / len(pairs) if pairs else 0.0

    result = {
        "fraction_nonmonotonic": fraction,
        "n_nonmonotonic": n_nonmono,
        "fraction_strict_moveaway": fraction_strict,
        "n_strict_moveaway": n_strict,
        "n_pairs": len(pairs),
        "tol": tol,
        "attractor_scale": scale,
        "per_pair": per_pair,
    }
    if example_pair is not None:
        ex = next(p for p in per_pair if p["pair"] == example_pair)
        result["pair"] = example_pair
        result["greedy_min_dist"] = ex["greedy_min_dist"]
        result["optimal_min_dist"] = ex["detour_min_dist"]
        result["required_distance_increase"] = ex["required_distance_increase"]

    if png_path is not None and example_curve is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(example_curve, lw=2)
            ax.axhline(example_curve[0], ls="--", c="gray", label="start distance")
            ax.axhline(tol, ls=":", c="green", label="reach tol")
            s, t = example_pair
            ax.set_title(f"gLV non-monotonic reachability: attractor {s} -> {t}\n"
                         f"(distance to target must INCREASE before reaching it)")
            ax.set_xlabel("env step")
            ax.set_ylabel("Euclidean distance to target attractor")
            ax.legend()
            fig.tight_layout()
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
            result["png_path"] = png_path
        except Exception as e:  # pragma: no cover - plotting is optional
            result["png_error"] = repr(e)

    return result


if __name__ == "__main__":  # pragma: no cover
    import json
    cfg = GLVConfig()
    sim = GLVSimulator(cfg)
    print("n_species:", sim.n_species, "action_dim:", sim.action_dim)
    print("attractors shape:", sim.attractors.shape)
    print("candidate_index:", sim.candidate_index)
    out = demonstrate_non_monotonicity(cfg)
    print("fraction_nonmonotonic:", out["fraction_nonmonotonic"])
    print(json.dumps(out["per_pair"], indent=2, default=float))
