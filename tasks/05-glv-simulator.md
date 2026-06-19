# WS5 — gLV simulator (the "Two Rooms" of microbiome)

Owner: sub-agent. Integrator: orchestrator. Read CLAUDE.md first (esp. "Data plan" item 2 and the
"non-monotonic attractors" requirement). Work ONLY in the file below. Smoke on `.venv-cpu`. Do NOT
commit/push; the orchestrator integrates.

## Goal
A generalized Lotka-Volterra (gLV) simulator that produces unlimited clean, controllable microbiome
trajectories with KNOWN ground-truth attractors and explicit interventions as actions. It is our
rigorous testbed for the IDM-ablation collapse story and for intervention planning.

CRITICAL (rubric-defining): the dynamics MUST be **non-monotonic** in reachability — for some
(initial state, target attractor) pairs, the optimal action sequence must first move AWAY from the
target (increase distance) before reaching it. If a greedy "reduce distance every step" policy always
works, planning is trivial and the result is worthless. You must DEMONSTRATE this property with a
number/figure. (This is the microbiome analog of Two Rooms.)

## File to create
`eb_jepa/datasets/microbiome/glv.py` (the package `__init__.py` already exists — do not touch it).

## API contract (other workstreams build against this — keep it EXACTLY)
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class GLVConfig:
    n_species: int = 32        # S, total species in the community
    n_candidate: int = 8       # K, action panel size (the curated taxa an intervention can perturb)
    dt: float = 0.05
    steps_per_action: int = 1  # integrate this many dt per env step
    noise_std: float = 0.0     # process noise (on log-abundance); 0 = deterministic
    seed: int = 0
    # ... any extra knobs needed to create multistability + non-monotonic reachability

class GLVSimulator:
    def __init__(self, config: GLVConfig): ...
    @property
    def n_species(self) -> int: ...
    @property
    def action_dim(self) -> int: ...          # == config.n_candidate (K)
    @property
    def candidate_index(self) -> np.ndarray: ... # [K] species indices the action perturbs (fixed)
    @property
    def attractors(self) -> np.ndarray: ...   # [n_attractors, S] known stable equilibria (>=2)
    def reset(self, seed=None, attractor=None) -> np.ndarray: ...   # -> state [S] (abundances >=0)
    def step(self, action: np.ndarray) -> np.ndarray: ...          # action [K] -> next state [S]
    def rollout(self, init_state: np.ndarray, actions: np.ndarray) -> np.ndarray: ...
        # actions [T, K] -> states [T+1, S]
    def generate_trajectories(self, n: int, T: int, action_policy: str = "random",
                              seed: int = 0) -> dict: ...
        # -> {"states": float32[n, T+1, S], "actions": float32[n, T, K]}

def demonstrate_non_monotonicity(config: GLVConfig | None = None) -> dict:
    """Return evidence: e.g. {"pair": (init_basin, target_attractor),
    "greedy_min_dist": float, "optimal_min_dist": float, "fraction_nonmonotonic": float, ...}
    showing that on a measurable fraction of (init, target) pairs the distance-to-target must
    increase before the target is reached. Also save a small PNG to the path given (optional)."""
```

Notes:
- Action semantics: `action a in R^K` is a delta applied to the K candidate species' abundance (or
  growth) each env step — a continuous relaxation of "add/remove a probiotic taxon". This is the
  clean ground-truth action for the IDM (predict a from z_t,z_{t+1}) and for MPPI/CEM planning.
- Keep states non-negative (clamp/relu); track abundance (and expose log-abundance helper).
- Multistability: design `r` (growth) and `A` (interaction matrix) to yield >=2 stable equilibria
  with a saddle/barrier so transitions can be non-monotonic. Document how you achieved it.
- Pure numpy (and optionally torch); NO import of heavy eb_jepa modules. <= ~350 lines.

## Smoke test (must pass on .venv-cpu; paste the output in your final report)
Write `eb_jepa/datasets/microbiome/_smoke_glv.py` (or run inline) that:
1. builds `GLVSimulator(GLVConfig())`, asserts `attractors.shape[0] >= 2`;
2. from each attractor, `rollout` with zero actions for T=200 stays within tol of that attractor
   (attractors are actually stable);
3. `generate_trajectories(n=8, T=20)` returns shapes `states [8,21,S]`, `actions [8,20,K]`, dtype
   float32, finite, non-negative states;
4. `demonstrate_non_monotonicity()` reports `fraction_nonmonotonic > 0` and prints the evidence.
Run: `/Users/bnz/DynaAMIcs/.venv-cpu/bin/python eb_jepa/datasets/microbiome/_smoke_glv.py`

## Definition of done
Deterministic given seed; >=2 verified-stable attractors; non-monotonicity demonstrated with a number;
`generate_trajectories` shapes/dtypes correct; smoke passes; report what you built, the attractor
structure, and the non-monotonicity evidence (with the numbers you measured — do NOT fabricate).
