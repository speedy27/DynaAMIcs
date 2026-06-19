# WS3 — Intervention planning on gLV (the Layer-B application result)

Owner: sub-agent. Integrator: orchestrator. Read CLAUDE.md (esp. action-space + planning notes) and
the "Integrity and honesty" rules. Work ONLY in the file below. Smoke on `.venv-cpu`. Do NOT
commit/push; the orchestrator integrates.

## Goal
Given a TRAINED gLV world model (encoder + GRU predictor from Layer B), plan a sequence of interventions
(continuous K-dim delta-abundance on the candidate panel) to drive the community from a start state to a
TARGET attractor, by minimizing latent distance to the target representation. Compare to baselines.
This makes our novelty (action-conditioned world model + planning) concrete.

## File to create
`examples/microbiome_jepa/plan_glv.py` (do NOT touch other files).

## Recommended approach: LEAN latent-space MPPI (do NOT use GCAgent)
The recon found `eb_jepa/planning.py` MPPIPlanner/GCAgent assume TENSOR observations
(`GCAgent.unroll` does `obs_init.repeat(...)`), which breaks our dict-obs. So implement a small MPPI
directly in latent space (cleaner + avoids that):
- Encode start community -> `z0 = jepa.encode(obs)[:, :, 0]` shape `[1, D, 1, 1]` (flatten to `[1,D]`).
- Encode target community (the target attractor rendered as a community) -> `z_tgt [1, D]`.
- ROLL FORWARD IN LATENT SPACE with the predictor (no re-encode): the RNNPredictor signature is
  `forward(state[B,D,1,1,1], action[B,K,1]) -> [B,D,1,1,1]` (action = GRU input, state = hidden;
  K = action_dim). For a batch of N candidate action sequences `a[N,K,H]`, roll H steps from z0.
- MPPI: sample N sequences ~ N(mean, std) over horizon H; cost = latent distance to z_tgt; support BOTH
  cumulative (sum over steps) and final-only cost (a flag — CLAUDE.md says cumulative helps); update
  mean by exp-weighted elites (temperature); iterate; return the first action. (Mirror the math in
  `planning.py` MPPIPlanner lines ~1299-1338, but in latent space.)
- MPC loop: execute the planned first action in the gLV ENV (`GLVSimulator.step`), re-encode the new
  community, replan, until horizon or success.

## Inputs / how to build the model + env + encode a community
- Rebuild the world model from the training cfg + checkpoint exactly like `probe_downstream.py` /
  `train_worldmodel.py` do: `load_config(fname)` -> build `SetTransformerEncoder` + `RNNPredictor`
  (hidden=D, action_dim=K) -> load `model_state_dict`/`encoder_state_dict` from the checkpoint saved by
  `train_worldmodel.run` (it calls `save_checkpoint(..., model=jepa, encoder_state_dict=...)`). Read
  `eb_jepa/training_utils.py:load_checkpoint` for the format.
- gLV env: `eb_jepa.datasets.microbiome.glv.GLVSimulator(GLVConfig(...))` — `reset`, `step(action[K])`,
  `attractors [n_attr, S]`, `candidate_index`. Use the SAME gLV params as training.
- Encode a single gLV state `x[S]` into the encoder's obs dict: REUSE the token construction in
  `eb_jepa/datasets/microbiome/traj.py` (`GLVTrajDataset._build_tokens` + its fixed seeded species
  embeddings + the SAME `zscore`). Easiest: instantiate a `GLVTrajDataset` with the SAME emb_seed and
  borrow `_species_emb`, `zscore`, and the CLR+mask logic to turn `x[S]` -> obs `{"otu":[1,1,S,F],
  "mask":[1,1,S]}`. Document this clearly.

## Baselines (report success rate of each)
- `random`: random actions each step.
- `greedy`: 1-step action that most reduces TRUE state-space distance to target (oracle-ish, in state
  space) — the baseline that the gLV's NON-MONOTONICITY should defeat.
- `final_only`: our MPPI but final-state cost only (ablates cumulative).
- `mppi` (ours): cumulative latent-distance MPPI.

## Success metric
Episode success = final gLV state within tolerance of the target attractor (e.g. relative L2 in
abundance space below a threshold tied to inter-attractor distance). Report success rate over
N episodes (varied start/target) x seeds, mean +/- s.e. Save JSON + a small figure (success rate per
method) like run_ablation.py.

## Smoke test (.venv-cpu; paste output)
`examples/microbiome_jepa/_smoke_plan.py`: build a TINY random world model (no checkpoint) + tiny gLV,
run 2 episodes of the full MPC loop for `mppi` + `random`, assert: shapes correct, MPPI returns finite
actions, env steps, success flag computed, JSON written. A random model will NOT succeed — that is fine
and expected; the smoke proves the HARNESS runs, not that planning works.
Run: `/Users/bnz/DynaAMIcs/.venv-cpu/bin/python examples/microbiome_jepa/_smoke_plan.py`

## Definition of done
Harness runs end-to-end on CPU with a random model (smoke); a fire `run(fname, checkpoint, ...)` entry
plans with a trained model and reports per-method success rates to JSON + figure. INTEGRITY: report only
measured success rates; if planning with the trained model does not beat baselines, say so honestly
(a negative result is acceptable and informative). Report file:line of the MPPI + MPC loop + the
state->obs encoding, and any assumption you could not verify.
