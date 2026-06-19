# WS4 — Downstream probe vs Susagi MLP baseline + sequencing-tech-invariance

Owner: sub-agent. Integrator: orchestrator. Read CLAUDE.md first (esp. "Strategy: Layer A downstream",
"Pitfalls", the verified-references novelty note, and "Integrity and honesty"). Work ONLY in the files
listed. Smoke on `.venv-cpu`. Do NOT commit/push; the orchestrator integrates.

## Goal (the Layer-A "result" for the rubric)
Given a TRAINED Layer-A set-JEPA encoder, evaluate the community representation on real Susagi
downstream tasks with a linear probe (+ small MLP), and COMPARE to the Susagi MLP baseline. Plus a
sequencing-technology-INVARIANCE probe (cited, from Cell-JEPA's argument): a representation that carries
LESS technical nuisance is better.

## Files to create (do NOT touch main.py, train_worldmodel.py, eval_collapse.py, datasets/*, architectures.py, losses.py)
- CREATE `examples/microbiome_jepa/probe_downstream.py`
- CREATE `examples/microbiome_jepa/baselines_port.py`  (thin port of the Susagi MLP baseline for comparison)

## Inputs / APIs you build against (already in the repo)
- WS1 data: `eb_jepa/datasets/microbiome/otu_data.py` — `OTUSampleDataset(mode="single", ...)` yields one
  community per sample as obs `{"otu":[1,N_max,F], "mask":[1,N_max]}` (F=385). It has a SYNTHETIC fallback
  (no real data needed for smoke) and a real-data path keyed off `data_dir`/`embeddings_h5`. READ this file
  for its exact config + how it loads real samples; reuse its loader, do not rewrite it.
- WS2 encoder: `eb_jepa.architectures.SetTransformerEncoder` — `forward(obs)->[B,D,T,1,1]`. Encode a
  community to a feature vector `z = encoder(obs).flatten(1)` -> `[B, D]`.
- Real data (CLUSTER ONLY, not on this Mac) lives at
  `/lustre/work/vivatech-dynamics/bbenziada/datasets/susagi/data/`. Match the label/format by READING the
  local Susagi clone:
  * infants env:  `/Users/bnz/Microbiome-Modelling/scripts/infants/predict_env.py` (+ `data/infants/meta_withbirth.csv` = SampleID,Env; `abundance.csv` = OTU x sample matrix)
  * IBS:          `/Users/bnz/Microbiome-Modelling/scripts/IBS/predict_ibs.py` (`data/IBS/final_metadata.csv` = run_id,country,ibs)
  * MLP baselines: `/Users/bnz/Microbiome-Modelling/scripts/*/base_lines_mlp.py` and `run_mlp_baselines.sh`
  * sequencing tech labels: `data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt`
    (and Susagi `scripts/utils.py` for how technology/biome terms are parsed). Tech = HiSeq/MiSeq etc.

## Deliverables
1. `probe_downstream.py`:
   - `encode_samples(encoder, dataset, device) -> (Z[N,D], labels)` — encode a labeled OTU-sample set.
   - `linear_probe(Z, y, groups=None, task="clf"|"reg") -> metrics` — grouped/stratified CV; report
     accuracy + macro ROC-AUC for classification (match Susagi's metric). Also a small-MLP probe option.
   - `tech_invariance(Z, tech_labels) -> accuracy` — train a classifier to predict sequencing tech from
     the representation; LOWER accuracy = better (less technical nuisance). Designed to compare our JEPA
     rep vs the Susagi imposter rep (so accept any [N,D] rep + labels).
   - `run(checkpoint, fname, task=...)` (fire entry): rebuild the encoder from the training cfg (`fname`)
     + load `encoder_state_dict` from `checkpoint`, build `OTUSampleDataset(mode="single", task-specific
     labels)`, run linear_probe + tech_invariance, print + save JSON. Synthetic fallback when no data_dir.
2. `baselines_port.py`: a minimal, faithful port of the Susagi MLP baseline (sklearn MLPClassifier with
   their CV protocol) operating on raw abundances, so we report OUR probe vs THEIR baseline on the SAME
   split. Cite the source script in a comment. Keep it small.

## Smoke test (must pass on .venv-cpu; paste output)
`examples/microbiome_jepa/_smoke_probe.py` (or inline) with SYNTHETIC data (no real files):
1. build a random `SetTransformerEncoder`, a synthetic `OTUSampleDataset(mode="single")` with fake
   labels + fake tech labels; `encode_samples` -> Z[N,D];
2. `linear_probe` returns finite accuracy/AUC; `tech_invariance` returns finite accuracy;
3. `baselines_port` MLP runs on synthetic abundances and returns finite metrics.
Run: `/Users/bnz/DynaAMIcs/.venv-cpu/bin/python examples/microbiome_jepa/_smoke_probe.py`

## Definition of done
Synthetic smoke passes; real-data code path implemented + keyed off `data_dir` (clearly flagged
UNVERIFIED since real data is cluster-only); metrics match Susagi's (accuracy + macro AUC) so the
comparison is apples-to-apples; tech-invariance probe works on any rep. Report exact synthetic outputs,
the file:line of each function, the Susagi scripts you matched, and every real-data assumption you could
not verify locally. Do NOT fabricate any accuracy/AUC — only report numbers your code actually produced.
