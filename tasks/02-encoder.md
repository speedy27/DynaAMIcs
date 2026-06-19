# WS2 — Set-transformer encoder + (optional) imposter-repulsion loss

Owner: sub-agent. Integrator: orchestrator. Read CLAUDE.md first (esp. "Architecture and eb_jepa
contracts" and the new "Repo reality check"). Work ONLY in the two files below, APPEND-ONLY (add new
classes; do NOT modify or break existing classes — the suite must still pass). Smoke on `.venv-cpu`.
Do NOT commit/push; the orchestrator integrates.

## Files to edit (append new classes only)
- `eb_jepa/architectures.py`  → add `SetTransformerEncoder`
- `eb_jepa/losses.py`         → add `ImposterRepulsionLoss` (optional, high-reward)

## THE OBS/TOKEN CONTRACT (shared verbatim with WS1 — you consume exactly this)
- Input obs is a DICT: `{"otu": FloatTensor[B, T, N_max, F], "mask": BoolTensor[B, T, N_max]}`
  (mask True = real OTU, False = pad). `F = 385` (384 ProkBERT + 1 z-scored CLR log-abundance).
  Features arrive ALREADY CLR'd + per-dim z-scored (WS1 does that). You just embed + attend + pool.
- Output MUST be `[B, D, T, 1, 1]` (one community vector per timepoint; H'=W'=1), matching
  ImpalaEncoder's contract so the predictor/regularizer/planning machinery work unchanged.

## Deliverable 1 — `SetTransformerEncoder` (architectures.py)
Permutation-invariant over OTUs, NO positional encoding.
```python
class SetTransformerEncoder(nn.Module):
    def __init__(self, token_dim=385, d_model=256, n_heads=4, n_layers=4,
                 dim_feedforward=512, dropout=0.0, pool="mean", mlp_output_dim=None): ...
        # expose self.mlp_output_dim = d_model (or a final projection dim) so the builder can set
        # the RNNPredictor hidden_size = encoder output dim D. Provide self.final_ln if you add one.
    def forward_set(self, tokens, mask):   # tokens [B*, N_max, F], mask [B*, N_max] -> [B*, D]
        # project F->d_model; TransformerEncoder with src_key_padding_mask=~mask; masked
        # permutation-invariant pool over N_max (masked mean, or PMA/attention pooling) -> [B*, D]
    def forward(self, obs):                 # obs dict -> [B, D, T, 1, 1]
        # fold T into batch: [B*T, N_max, F]; call forward_set; reshape to [B, D, T, 1, 1]
```
Hard requirements:
- Output shape EXACTLY `[B, D, T, 1, 1]`.
- PERMUTATION INVARIANCE: permuting the N_max OTU order (and mask together) yields the same output
  within 1e-5.
- MASK INVARIANCE: changing values in padded (mask=False) slots does NOT change the output.
- Must accept variable N_max and a fully-masked-except-one community without NaN.
- D configurable; the builder will set predictor hidden_size = D and feed `state [B,D,T,1,1]` to
  `VC_IDM_Sim_Regularizer` (expects [B,C,T,H,W]) and per-timestep `[B,D,1,1,1]` to `RNNPredictor`.

## Deliverable 2 (optional, do if time) — `ImposterRepulsionLoss` (losses.py)
Carry Susagi's central idea into the JEPA as a regularizer: in masked-prediction mode, the predicted
representation of a masked REAL OTU should be FAR from the representation of a hard IMPOSTER OTU
(an OTU close in DNA-embedding space but not in the community). Signature:
```python
class ImposterRepulsionLoss(nn.Module):
    def __init__(self, margin=...): ...
    def forward(self, pred_rep, real_rep, imposter_rep) -> Tensor   # scalar; hinge/triplet style
```
Return a finite scalar; keep it self-contained so the builder can add it as a gated sub-loss and log
it. (The imposter SAMPLING lives in WS1/data; this is just the loss term.)

## Smoke test (must pass on .venv-cpu; paste output)
Script `eb_jepa/_smoke_encoder.py` that:
1. builds `SetTransformerEncoder(token_dim=385, d_model=128)`; feeds obs with B=4,T=3,N_max=16,F=385
   (random tokens, random mask with >=1 real per row); asserts output shape EXACTLY `[4,128,3,1,1]`;
2. permutation-invariance: permute N_max dim of otu+mask the same way → max abs diff < 1e-5;
3. mask-invariance: randomize values in padded slots → output unchanged (<1e-5);
4. plug the output `state` into `VC_IDM_Sim_Regularizer(cov_coeff=1,std_coeff=1,sim_coeff_t=1)` →
   returns finite (weighted, unweighted, dict); take `state[:,:,0:1]` → reshape `[4,128,1,1,1]`,
   build `RNNPredictor(hidden_size=128, action_dim=4, final_ln=nn.Identity())`, call with
   `action=torch.randn(4,4,1)` → returns `[4,128,1,1,1]` finite;
5. `ImposterRepulsionLoss` on random reps → finite scalar.
Run: `/Users/bnz/DynaAMIcs/.venv-cpu/bin/python eb_jepa/_smoke_encoder.py`
Then confirm you didn't break anything:
`/Users/bnz/DynaAMIcs/.venv-cpu/bin/python -m pytest tests/test_jepa_output_formats.py -q`

## Definition of done
Output is exactly `[B,D,T,1,1]`; permutation- and mask-invariance verified numerically; output composes
with `VC_IDM_Sim_Regularizer` and `RNNPredictor` (D==hidden_size) without shape errors; existing tests
still pass; append-only edits. Report the shapes/diffs you actually measured (no fabrication).
