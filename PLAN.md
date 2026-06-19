# PLAN.md — EB-JEPA Microbiome World Model

Living source of truth for orchestration. Read CLAUDE.md first (strategy, contracts, integrity rules).
Last updated: 2026-06-19 by orchestrator (after Layer A + Layer B harness built & CPU-green).

## Status at a glance
- **Build phase: DONE & CPU-green, committed/pushed to `bnz`.**
  - Layer A static set-JEPA (`examples/microbiome_jepa/main.py`, commit ef608b3): two-view VICReg/BCS,
    runs end-to-end, no collapse (feat_std ~0.91). SYNTHETIC CPU smoke — not a science result yet.
  - Layer B AC world model (`examples/microbiome_jepa/train_worldmodel.py`, commit cd75b67): IDM-ablation
    harness (idm_coeff on/off), both arms run end-to-end on CPU. NOT a result: under-trained smoke shows
    pred~0.001/feat_std~0.04 in BOTH arms — distinguishing them needs real training + a collapse metric.
  - WS5 gLV (`glv.py`): 3 stable attractors, non-monotonicity demonstrated (greedy fails 6/6).
- **Cluster: FULLY OPERATIONAL.** Both arch venvs built (h5py 3.16 + pandas 3.0 confirmed in the aarch64
  venv); 22 GB data downloaded+verified; GPU pytest 21/21. Repo on cluster at ef608b3 — **needs `git pull`
  to cd75b67 before Layer B GPU runs.** Reservation `Vivatech` expires **2026-06-21T00:00**.
- **NEXT (science phase):** (M4) build a collapse metric (dynamics vs slow-feature decodability probe),
  tune so idm-on vs idm-off diverges, then 3-seed GPU ablation = the headline figure. (M2) real-data
  Layer A probe vs Susagi MLP baseline.
- Gotcha logged: verify cluster venv imports with `uv run python` / the venv python (on a compute node),
  NOT bare `python` — job 74382 "FAILED" was only that false-negative check.

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

## Lit-review course-corrections (2026-06-19, from verified references)
- NOVELTY (report/slides framing, no code change): static masked-set JEPA for omics already exists
  (GeneJepa/Cell-JEPA/JEPA-DNA; GeneJepa self-calls "world model"). DO NOT claim "first JEPA for
  biology"; lead with the ACTION-CONDITIONED TEMPORAL world model + INTERVENTION PLANNING (our white
  space). Encoder = enabling substrate, not headline. See CLAUDE.md thesis pt 4.
- NEW EVAL METRIC (M2, real data, cheap + cited): sequencing-technology-invariance probe. Train a
  classifier to predict sequencing tech (HiSeq/MiSeq, from Susagi sample metadata
  data/microbeatlas/sample_terms_mapping...biome_tech) from (a) our JEPA community rep vs (b) the
  Susagi imposter rep; LOWER accuracy = better (rep carries less technical nuisance). Cell-JEPA's core
  argument; Susagi's own rollouts show tech separation. Sibling to the gLV fast/slow collapse probe.
- CONTINGENCY LEVERS (only if needed; don't switch mid-experiment):
  * If VICReg/VC_IDM tuning is fiddly in the ablation -> eb_jepa ships BCS (SIGReg), single hyperparam;
    LeWM shows pred + Gaussian-isotropy alone is stable. (BCS is two-view, Layer A; for Layer B the
    sequence reg is VC_IDM — keep unless std/cov tuning blocks us.)
  * If the real-data Layer A probe underperforms vs Susagi MLP -> add Fourier features on the CLR
    abundance concatenated to the ProkBERT token (GeneJepa tokenizer trick). Else skip.
- Forecasting baselines to position against (temporal/planning): gLV/cLV, pNODE, MicroProphet (all real).

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
  * M4 HEADLINE — IDM ablation, gLV, GPU job 74610, 3 seeds, 80 epochs (eval_collapse standardized probe):
    - Induce-collapse regime (sim=4,cov=1,std=0.25): fast_r2_action IDM-on 0.748±0.051 vs off 0.520±0.021
      (Δ+0.229, on>off in ALL 3 seeds, non-overlapping); Δstate +0.082; slow saturated ~0.99 both.
    - Default regime (sim=1,cov=25,std=1): fast_r2_action 0.364±0.020 vs 0.291±0.041 (Δ+0.073, 2/3 seeds,
      1 reversed) — modest, seed-noisy; NOT overclaimed. Figures in examples/microbiome_jepa/results/.
    - Finding = regime-dependence: strong VICReg partially substitutes for IDM; in the collapse-prone
      regime IDM robustly rescues the intervention/dynamics signal.
  * M3 PLANNING — gLV latent-MPPI, GPU job 74718, 3 seeds, 12 ep/seed: HONEST NEGATIVE. 0% success for
    ALL methods (random/greedy/final_only/mppi) at tol=0.15*attr_scale; MPPI does NOT beat baselines
    (final dist 4.88 vs random 4.58 / greedy 4.51; start 6.64). Reported as-is; likely latent-vs-state
    geometry mismatch and/or weak bounded actions. Headline (M4) stands independently.
- UNVERIFIED: WS1 real-data loader path (the 22GB corpus is cluster-only; verify on cluster before M2).
  Susagi baseline numbers (infants/IBS) quoted from their result files — re-verify provenance in WS4
  before claiming "beat the baseline".
- No fabricated numbers anywhere. Every figure above came from a real run named here.
