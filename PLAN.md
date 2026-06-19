# PLAN.md — EB-JEPA Microbiome World Model

Living source of truth for orchestration. Read CLAUDE.md first (strategy, contracts, integrity rules).
Last updated: 2026-06-19 by orchestrator (initial plan).

## Status at a glance
- **Orientation: DONE.** Code contracts verified against the real repo (see CLAUDE.md "Repo reality
  check"). Susagi parts bin mapped (`/Users/bnz/Microbiome-Modelling`).
- **Local smoke env: DONE.** `.venv-cpu` (CPU torch 2.12.1) works; `tests/` 16/16 passed on it.
- **Cluster: BLOCKED on user authorization.** Dalia work dir is empty; clone+setup.sh (builds the two
  arch venvs) and any job submission write to a shared allocation, so they need explicit go-ahead.
  Reservation `Vivatech` expires **2026-06-21T00:00 (~2 days)** — this is the binding time constraint.
- **Data: NOT downloaded.** Susagi data lives on Zenodo (DOI 10.5281/zenodo.18679373) + HF
  (basilboy/microbiome-model). Cluster has egress (github/zenodo/hf all 200). Download is a cluster op
  (same authorization).

## The binding constraints (design around these)
1. ~2 days of GPU. Optimize for ONE clean rubric-maximizing result, not breadth. The headline is the
   gLV IDM-ablation collapse-and-recovery story (synthetic, clean, controllable) + a Layer-A probe that
   beats the MLP baseline. Everything else is stretch.
2. Mac can't run CUDA torch → build + validate ALL logic on `.venv-cpu` first; cluster only for scale,
   speed, 3-seed sweeps, real DIABIMMUNE, and final figures.
3. Partition by FILE. One integration owner for `examples/microbiome_jepa/main.py` (orchestrator).

## Workstreams, file ownership, dependency order
Integration owner of `examples/microbiome_jepa/main.py` and `eval.py`: **orchestrator (me)**.
Everyone else delivers self-contained nn.Modules / functions / datasets that I wire in. No two agents
edit the same file at once.

| WS | Name | Owns (files) | Depends on | Status |
|----|------|--------------|-----------|--------|
| WS5 | gLV simulator | `eb_jepa/datasets/microbiome/glv.py` (new) | — | not started |
| WS1 | Data | `eb_jepa/datasets/microbiome/otu_data.py`, `transforms.py`, `traj.py` (new); the `microbiome` branch in `datasets/utils.py` (coordinated edit) | WS5 for traj format | not started |
| WS2 | Encoder + regularizers | new classes in `eb_jepa/architectures.py` (SetTransformerEncoder), `eb_jepa/losses.py` (optional imposter-repulsion term) | shape contract only | not started |
| WS3 | World model + planning | `eb_jepa/planning.py` (microbiome goal objective), action-encoder + IDM wiring spec | WS5, WS2 | not started |
| WS4 | Eval + figures | `examples/microbiome_jepa/probes.py`, `figures.py`, baseline port from Susagi (new files) | WS1, WS2 | not started |
| WS0 | Orchestration/integration | `examples/microbiome_jepa/main.py`, `eval.py`, `cfgs/`, `PLAN.md`, cluster, merges | all | in progress |

Rationale for ownership: `architectures.py` and `losses.py` are shared infra but WS2 only APPENDS new
classes — single owner avoids conflicts. `datasets/utils.py` gets one small `microbiome` branch; the
orchestrator lands that edit to keep the dispatcher uncontended. The set-transformer encoder is shared
by Layer A and Layer B (built once, in WS2).

## Dependency order / milestones
- **M0 (now): bootstrap.** Cluster clone+setup (blocked on auth) ‖ build WS5 gLV + WS1 data scaffold +
  WS2 encoder locally and CPU-smoke each in isolation.
- **M1: Layer A end-to-end (the safety net).** `examples/microbiome_jepa/main.py` trains a static
  set-JEPA (two-view VICReg or masked prediction) on gLV-derived OTU sets + real OTU sets, no collapse,
  CPU smoke at `optim.epochs=2`. Then a linear probe. → first thing that must always run.
- **M2: Layer A result.** Linear probe beats the Susagi MLP baseline on ≥1 downstream task; 3-seed
  error bars (cluster). Targets to beat (MEASURED by Susagi, in their repo result files — verify
  provenance): infants env CV acc 0.5486±0.0309 / macro AUC 0.9124±0.0069; IBS cross-country AUC matrix.
- **M3: Layer B world model.** AC temporal JEPA on gLV with IDM + Lsim; planning reaches target
  attractors above random/greedy/final-state-only.
- **M4: Headline.** IDM-ablation collapse-and-recovery curve (without IDM the encoder collapses onto
  slow features; with IDM it captures community dynamics). The single most rubric-maximizing figure.
- **M5: reality-check + report.** DIABIMMUNE probe; 1–2pp report + 3-min demo, one-command runnable.

## Decisions taken (and why) — update if changed
- **USER AUTHORIZED (2026-06-19): full cluster autonomy** (bootstrap, data download, submit jobs without
  per-action prompts) **AND full real-corpus pretraining** (download everything; pretrain set-JEPA on
  real unlabeled MicrobeAtlas samples at scale, in addition to gLV).
- **gLV-synthetic still owns the headline ablation.** Clean, controllable, known ground truth, cheap on
  CPU → the IDM-ablation collapse-and-recovery curve runs there. Real corpus is the SCALE pretraining +
  downstream probe + DIABIMMUNE reality-check. Both, now that scale is greenlit.
- **Token = concat(ProkBERT 384-d DNA embedding, CLR log-abundance), per-dim z-scored.** NOTE: Susagi's
  own model does NOT concatenate abundance (DNA-embedding token only); adding abundance is OUR JEPA
  enrichment. Call this out in the report as a deliberate design choice.
- **Data: single 6.2 GB `data.zip` on Zenodo** (record 18679373, "Dataset for Evaluations of Community
  Stability Microbiome Model"). NOT 19 GB (that was the uncompressed MicrobeAtlas estimate). Downloading
  to `$WORK/datasets/susagi/` now; Susagi repo also cloned on cluster for the baseline/eval port (WS4).

## Open questions / blockers (for the human)
1. **Authorize cluster operation?** I need a go-ahead to (a) clone the repo to /lustre/work and run
   setup.sh, (b) submit SLURM jobs (training/eval) with `--reservation=Vivatech`, (c) download the
   Susagi data to /lustre/work. All reversible; none touches other users' files. Without this I can
   still build + CPU-smoke all Layer A/B logic, but cannot produce GPU results or 3-seed numbers.
2. **Data scope:** OK to defer the 19 GB corpus (gLV-first plan above), or do you want real-data
   pretraining at scale?

## Known integration TODOs (discovered during build)
- **pyproject.toml lacks `h5py` and `pandas`** — WS1's real-data loader needs both. Add them and
  re-sync the aarch64 cluster venv BEFORE any real-data (M2) run. (CPU smoke uses synthetic, so this
  didn't surface locally.)
- WS1 real-data format VERIFIED 2026-06-19 against the real cluster files: `samples-otus.97.mapped` =
  `>SRR.SRS` blocks then `90_;96_;97_<id>\t<count>` (abundance = 2nd field, as WS1 assumed); downstream
  `IBS/final_metadata.csv` = run_id,country,ibs; `infants/meta_withbirth.csv` = SampleID,Env (+ an
  OTU×sample `infants/abundance.csv`). The infants probe must read the abundance MATRIX (samples as
  columns) — note for WS4.
- fire override syntax is `--key value` (bare `key=value` binds to the positional `cfg`). Fixed in docs.

## Integrity ledger (measured vs expected)
- MEASURED (2026-06-19):
  * Local CPU tests: test_loss_equivalences + test_jepa_output_formats = 16/16.
  * Cluster: both venvs built; GPU pytest (job 74351) = 21 passed in 20s on aarch64 GB200; 22 GB data
    verified (prokbert_embeddings.h5 670MB/384-d, samples-otus.97.mapped 20GB, downstream CSVs).
  * WS2 encoder smoke: output [4,128,3,1,1], perm-inv 9.5e-7, mask-inv 0.0, composes w/ regularizer+RNN.
  * WS1 data smoke (4/4): obs (B,1,N_max,385); CLR+z-score per-dim mean~3e-8/std~1.0; init_data ok.
  * M1 Layer A CPU smoke (SYNTHETIC data, 2 epochs, tiny): loss 1.41->1.29, invariance 0.138->0.079,
    cov_loss ~0.005-0.01, feat_std ~0.91 STABLE (no collapse). This validates the PIPELINE only — it
    is NOT a scientific result (synthetic data, 2 epochs). Real-data training + seeds are M2.
- UNVERIFIED: WS1 real-data loader path (the 22GB corpus is cluster-only; verify on cluster before M2).
  Susagi baseline numbers (infants/IBS) quoted from their result files — re-verify provenance in WS4
  before claiming "beat the baseline".
- No fabricated numbers anywhere. Every figure above came from a real run named here.
