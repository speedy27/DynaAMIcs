# WS1 — Data: OTU-set + trajectory datasets, CLR/z-score, init_data branch

Owner: sub-agent. Integrator: orchestrator. Read CLAUDE.md first (esp. "Data plan", and the pitfalls
on CLR + per-feature z-score). Work ONLY in the files listed. Smoke on `.venv-cpu`. Do NOT
commit/push; the orchestrator integrates.

## Files to create / edit
- CREATE `eb_jepa/datasets/microbiome/transforms.py`
- CREATE `eb_jepa/datasets/microbiome/otu_data.py`
- CREATE `eb_jepa/datasets/microbiome/traj.py`
- EDIT  `eb_jepa/datasets/utils.py` — add a `microbiome` branch ONLY (early return). Do NOT alter the
  two_rooms / maze code paths. This is the only shared file you touch; keep the diff minimal.
(The package `eb_jepa/datasets/microbiome/__init__.py` already exists — do not modify it.)

## THE OBS/TOKEN CONTRACT (shared verbatim with WS2 — the encoder consumes exactly this)
- Per-OTU token feature dim `F = D_emb + 1`, with `D_emb = 384` (ProkBERT). Token =
  `concat( z(ProkBERT_384) , z(CLR_log_abundance) )`, i.e. all F dims PER-DIMENSION z-scored
  (fit mean/std on train, persist them). CLR (centered log-ratio) is applied to relative abundances
  BEFORE z-scoring (pseudocount for zeros).
- A "community at one timepoint" = a set of up to `N_max` OTU tokens + a boolean presence mask.
- A dataset sample returns `obs` as a DICT of TIME-MAJOR tensors (so TrajSlicerDataset can slice
  along time by `v[start:end:frameskip]`):
    `obs = {"otu": FloatTensor[T, N_max, F], "mask": BoolTensor[T, N_max]}`  (True = real, False = pad)
  For static Layer-A samples, `T = 1`.
- After the default DataLoader collate this becomes
    `{"otu": [B, T, N_max, F], "mask": [B, T, N_max]}`.
- The set-transformer encoder (WS2) maps this dict -> state `[B, D, T, 1, 1]`. You do NOT build the
  encoder; you only produce the obs dict in this exact shape/dtype.

## Reference the REAL data format from the Susagi clone (read, don't guess)
Real data is on the cluster at `/lustre/work/vivatech-dynamics/bbenziada/datasets/susagi/data/`
(NOT on this Mac). Match its format by reading the local Susagi clone:
- `/Users/bnz/Microbiome-Modelling/scripts/utils.py` — ProkBERT h5 loading (key scheme), the
  A97/B97 `otus.rename.map1` resolver (`load_otu_rename_map`, `build_otu_key_resolver`),
  `build_sample_embeddings`.
- `/Users/bnz/Microbiome-Modelling/scripts/jepa/data.py` — how `samples-otus.97.mapped` is parsed
  into per-sample OTU sets + abundances; DIABIMMUNE longitudinal sample grouping.
Key real paths (for the cluster code path / configs):
- embeddings: `…/susagi/data/model/prokbert_embeddings.h5` (670 MB, 384-d, float32, keyed by OTU id)
- corpus: `…/susagi/data/microbeatlas/samples-otus.97.mapped` (20 GB) + `otus.rename.map1` (41 MB)
- downstream: `…/susagi/data/{gingivitis,IBS,infants,diabimmune,snowmelt}/*.csv|*.tsv`

## Deliverables
1. `transforms.py`:
   - `clr(abund: Tensor, pseudocount=...) -> Tensor` (centered log-ratio; handle zeros/sparsity).
   - `PerDimZScore` (fit/transform; stores mean/std; serializable) over the F-dim token features.
   - `augment_community(tokens, mask, *, subsample_frac, jitter_std, dropout_p, generator)` for
     two-view SSL: random OTU subsample, gaussian jitter on the log-abundance feature, OTU dropout.
   - `mask_otus(tokens, mask, frac, generator)` for I-JEPA masked prediction: returns
     (visible_tokens, visible_mask, target_idx) so the predictor can predict masked OTU reps.
2. `otu_data.py`:
   - `load_prokbert_embeddings(path) -> (emb: ndarray[N_otu, 384], otu_id_to_row: dict)` + the
     A97/B97 resolver (reuse Susagi logic).
   - `OTUSampleDataset(Dataset)`: builds per-sample OTU sets + abundances → tokens `[N_max, F]` +
     mask. Configurable `mode in {"two_view","masked","single"}`:
       * two_view → `__getitem__` returns `(view1_obs, view2_obs)` each `{"otu":[1,N_max,F],
         "mask":[1,N_max]}` (two augmentations of the same community) for Layer-A VICReg/BCS.
       * masked → returns the masked split for I-JEPA.
       * single → returns one community (for probing/eval).
     Expose attrs: `.token_dim (F)`, `.n_max`, `.zscore` (the fitted PerDimZScore).
   - A SYNTHETIC fallback so CPU smoke needs NO real data: if the data dir is absent, generate random
     OTU embeddings + sparse abundances with the SAME shapes/dtypes. A real-data code path keyed off
     the embeddings-h5 path / `data_dir` config.
3. `traj.py`:
   - `GLVTrajDataset(TrajDataset)` wrapping WS5's `GLVSimulator.generate_trajectories` (API in
     tasks/05-glv-simulator.md — build against that interface; import lazily so a missing glv.py
     doesn't break import). For SYNTHETIC species (no DNA), the per-species token =
     `concat( fixed_seeded_species_embedding[s] (dim 384) , z(log_abundance_s) )`, mask = species
     present (abundance>eps). This lets the SAME set-transformer consume gLV communities.
     `__getitem__(i)` returns the 5-tuple `(obs_dict, act[T,K], state[T,S], reward[T], extra)` and the
     class exposes `.proprio_dim`, `.action_dim (=K)`, `.state_dim`, and `get_seq_length`. Verify the
     5-tuple/collate convention against `eb_jepa/datasets/two_rooms/wall_dataset.py` +
     `eb_jepa/datasets/traj_dset.py` (TrajSlicerDataset slices obs dict items along time).
4. `utils.py` microbiome branch: at the TOP of `init_data`, `if env_name == "microbiome": return
   init_microbiome_data(cfg_data, device)` returning `(train_loader, val_loader, config, None)`. Put
   `init_microbiome_data` in `otu_data.py` or `traj.py`. Keep config a simple object with the attrs
   the example needs (`batch_size`, `size`, ...). Do not disturb `_resolve_env`/`create_env` for
   two_rooms/maze (a microbiome `create_env` for planning is WS3's job; leave a TODO).

## Smoke test (must pass on .venv-cpu; paste output)
A script `eb_jepa/datasets/microbiome/_smoke_data.py` that, with NO real data present:
1. builds `OTUSampleDataset(mode="two_view", synthetic=True)`, wraps in DataLoader(batch_size=4),
   pulls one batch, asserts `view1["otu"].shape == [4,1,N_max,F]`, `view1["mask"].shape==[4,1,N_max]`,
   F==385, dtypes float32/bool, and that view1 != view2 (augmentation differs);
2. asserts CLR+z-score: on the fitted train features, per-dim mean≈0, std≈1 (within tol);
3. builds `GLVTrajDataset` (use a tiny stub if glv.py absent — but prefer the real one once WS5 lands)
   through `TrajSlicerDataset(num_frames=4)`, pulls a slice, asserts `obs["otu"].shape==[4,N_max,F]`,
   `act.shape==[4,K]`;
4. `init_data("microbiome", cfg_data={...})` returns 4-tuple with working train/val loaders.
Run: `/Users/bnz/DynaAMIcs/.venv-cpu/bin/python eb_jepa/datasets/microbiome/_smoke_data.py`

## Definition of done
All four smoke checks pass on `.venv-cpu`; obs dict matches the contract EXACTLY; CLR+z-score verified
numerically; two_rooms/maze paths untouched; report shapes/dtypes you actually observed (no fabrication)
and note anything about the real-data format you couldn't verify without the cluster files.
