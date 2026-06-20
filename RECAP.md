# RECAP — Tout ce qu'on a ajouté au fork eb_jepa

> Mémo anti-perte. Inventaire de **ce qui a été créé / fait** sur le fork `eb_jepa`
> pour le hackathon. Mis à jour le **2026-06-20**. Branche d'intégration : `adrien`
> (merge de `tristan`). Upstream intact : `origin/main`.

---

## 🗓️ Journal des itérations

> **Convention** : à chaque nouvelle itération, on **préfixe ici** une entrée datée
> (la plus récente en haut). Format : `Ajouté` / `Modifié` / `Fait & vérifié` /
> `Résultats` / `À suivre`. Les sections 0–10 plus bas restent l'**inventaire de
> référence** (mis à jour quand une brique devient permanente).

### 2026-06-20 · itération 7 — Rapport HTML final Microbiome + Tahoe + pitch 10 minutes

**Ajouté**
- `artifacts/report/build_research_report.py` — générateur Python du **rapport de recherche HTML**
  final : structure ESIEE complète (Data → Architecture → Training → Inference → Evaluation),
  checklist Gold Standard, pitch 10 minutes, processus d'itération, limites honnêtes et références.
- `artifacts/report/assets/` — 25 PNG copiés/générés : figures microbiome F1–F8 + figures FCGR/
  Met2Img + figures Tahoe (collapse, skill, UMAP, modules, ablations, scaling/screening).
- Figures Tahoe générées depuis JSON réels : `tahoe_perturb_ablation.png`,
  `tahoe_encoder_ablation.png`, `tahoe_scaling_screening.png`.

**Modifié**
- `artifacts/report/index.html` — remplacé par un rapport **HTML autonome** (20 images inlinées en
  base64), lisible, clair et orienté jury.
  Il répond précisément aux demandes : graphiques/représentations Microbiome + Tahoe, architecture,
  protocole de test, résultats vs baselines, ablations, inference/planning, pitch 10 minutes et références.

**Fait & vérifié**
- Build : `uv run --no-sync python artifacts/report/build_research_report.py` ✅.
- Contrôles HTML : 0 placeholder `Figure manquante`, 20 `<img>` intégrées, 0 référence externe
  `assets/*.png`, sections `Gold Standard`, `Pitch 10 minutes`, `Evaluation Microbiome`,
  `Evaluation Tahoe` présentes.
- `get_errors` clean sur le générateur.

**Résultats intégrés**
- Microbiome : normalisation ON/OFF, bio-loss/no-reg, PCA latents, CV subject-grouped, spectre
  dynamique, trajectoires, FCGR vs ProkBERT.
- Tahoe : perturbation skill/ablation, SetTransformer vs MLP, SIGReg collapse, scaling, zero-shot,
  screening top drugs.

**À suivre**
- Optionnel : transformer ce rapport en deck `.pptx`/slides 10 min ou ajouter F9 Gingivitis
  (drop-out/colonise ROC) si le suivi per-OTU est branché.

### 2026-06-20 · itération 6 — +3 figures (Susagi blog + corpus) : infant CV, spectre, trajectoires

**Ajouté à `examples/microbiome/make_figures.py`** (8 figures au total) :
- F6 `infant_cv` — protocole « Infants » de Susagi : probe embeddings gelés, **GroupKFold par
  sujet** (sans fuite), JEPA vs raw mean-ProkBERT, âge R² + diversité R² (mean±std).
- F7 `dynamics_spectrum` — opérateur de transition latente fitté (z'≈W[z,a]), |valeurs propres|
  triées + cercle unité (style Koopman-JEPA Fig.4).
- F8 `latent_trajectories` — trajectoires latentes par sujet (PCA, colorées âge) + histogramme du
  pas latent (style Temporal Straightening Fig.2).
- Loader enrichi `load_latents_full` (pooled + séquence + transitions + descripteur brut + subject ids).

**Fait & vérifié**
- Régénéré sur GPU, 8 PNG inspectés.
- ⚠️ Bug d'honnêteté attrapé : la CV par-fenêtre **fuit** (fenêtres d'un même sujet dans 2 folds)
  → AUROC=1.0 & raw « bat » JEPA (faux). Corrigé en **GroupKFold(sujet)**.

**Résultats réels (ckpt ab_norm_on, val, subject-grouped CV)**
- Âge R² : **raw −1.56±0.53 vs JEPA +0.31±0.38** (JEPA généralise, raw sur-apprend l'identité sujet).
- Spectre dynamique : opérateur **contractif** (|λ|<0.62, 0 mode |λ|>0.9) → dynamique faible/amortie.
- Pas latent moyen ≈ 1.52 (les latents bougent ; skill<1 = direction non prédite).

**À suivre**
- Option Gingivitis (drop-out/colonise ROC) : suivi per-OTU depuis le cache brut + probe
  `z_t ⊕ emb_OTU → présent à t+1`. Faisable, à brancher si voulu.

### 2026-06-20 · itération 5 — Figures « deck-grade » d'éval/ablation (5 figures, données réelles)

**Ajouté**
- `examples/microbiome/make_figures.py` — générateur des **5 figures reference-grade** (calquées sur
  VICReg / MAE / I-JEPA / DINOv2 / β-VAE+DreamerV3), **100% à partir d'artefacts réels** (log
  d'entraînement, checkpoint, `metrics.json` d'ablation). Skip propre + note si une source manque
  (jamais de valeurs inventées).
- `artifacts/figures/F1..F5.png` : F1 collapse panel · F2 ablation grid · F3 before/after à compute
  égal · F4 PCA des latents (âge/diversité/phénotype) · F5 scaling + comparaison contrôlée.

**Fait & vérifié**
- Généré sur GPU local depuis `artifacts/train_log.txt` (run 50-ep) + ckpt `ab_norm_on` + ablation
  `ab_norm_{on,off}`. Les 5 PNG inspectés (layout F1/F3 corrigé).
- ⚠️ Piège Windows : `CUDA_VISIBLE_DEVICES=""` casse l'init cudnn du GRU (`min()` vide) → garder le GPU.

**Résultats réels mis en figure**
- Ablation **per-feature norm** (15 ep, seed 0) : skill **1.078 vs 0.151**, AUROC **0.907 vs 0.537**,
  age_r2 **0.427 vs 0.247**, effrank **4.38 vs 3.90** → la normalisation par-feature (exigée par le
  rubric) est le levier qui fait passer skill>1.
- Run 50-ep : age_r2 **0.54** (horloge OK) mais **tvar=0 / skill<1** = collapse temporel (l'insight).

**À suivre**
- Régénérer F5 avec les vrais 3 seeds (1/1000/10000) quand le sweep cluster est rapatrié → error bars.

### 2026-06-20 · itération 4 — Intégration FCGR (microbiome2img) dans la classe JEPA

**Ajouté**
- `eb_jepa/architectures.py` → `FCGRSetEncoder` — encoder communauté **DNA-as-image** :
  chaque OTU = image FCGR aplatie dans les canaux (`S²+1`), CNN par token + pooling
  pondéré par l'abondance. Hérite de `TemporalBatchMixin` → **drop-in exact** du
  `SetEncoder` (contrat `[B, D, T, 1, 1]`). L'abondance sert seulement de poids (jamais
  feature CNN) → pas de souci de normalisation inter-canaux.
- `examples/microbiome2img/synth.py` — cohorte FCGR **synthétique** : panel d'OTU en
  clades → images FCGR (z-scorées par pixel), trajectoires d'abondance **action-conditionnées**
  (un « régime » booste un clade par step). Émet le **même dict** que `MicrobiomeDataset`.
- `examples/microbiome2img/main.py` + `cfgs/train.yaml` — câble la **classe `JEPA` de la
  lib** (FCGRSetEncoder + RNNPredictor + VC_IDM_Sim_Regularizer + SquareLossSeq + pertes
  bio), mêmes métriques que `examples/microbiome` (skill / effrank / tvar / age_r2 / probe).
- `tests/test_microbiome2img.py` (4 tests) + `tests/conftest.py` (rend `examples`
  importable sous pytest).

**Modifié**
- `examples/microbiome2img/README.md` — statut « Full JEPA integration » → ✅.

**Fait & vérifié**
- `pytest tests/test_microbiome2img.py tests/test_microbiome.py` → **12/12 ✅**.
- Smoke GPU end-to-end (`main.py`, device=cuda, 0.62M params) : tourne, sauve ckpt+metrics.

**Résultats (synthétique, 25–30 ep, honnête)**
- z-score du panel FCGR : **effrank 1.0 → ~4.6** (corrige le collapse de *features* ; piège
  CLAUDE.md = probas FCGR ~1/256 trop petites pour le CNN).
- **Latent utile** : `pheno_auroc ≈ 0.94–0.96` (le probe linéaire sépare le phénotype).
- ⚠️ **Collapse temporel** persistant (`tvar≈0`, `skill≈0`) malgré tvar_coeff↑ : même
  *slow-feature collapse* que l'exemple microbiome (l'encodeur encode l'identité sujet,
  pas la dynamique). On-thesis, pas un bug d'intégration.

**À suivre**
- Vrai head-to-head image-CGR vs ProkBERT : brancher le panel `synth.py` sur un FASTA réel.
- Creuser le skill du world-model FCGR (couplage échelle `z` ↔ LayerNorm du prédicteur ;
  tester un Δt-stride / des margins) — chantier séparé de l'intégration.

### 2026-06-20 · itération 3 — Ablation encoder Tahoe + fix no-op IDM + récup résultats

**Ajouté**
- `examples/tahoe/encoder_ablation.py` — driver d'**ablation encoder MLP ↔ Set-Transformer**
  (mêmes seeds, même protocole linear-probe que `main.py`). Sort `results.json`
  (macro-F1 mean±std par tâche) + figure `encoder_ablation.png` (style deck).
  Inclut `--selftest` (agrégation + figure sur données synthétiques, sans cache/GPU).

**Modifié**
- `examples/tahoe/main.py` — `run()` **retourne** maintenant les métriques de probe
  finales (`{encoder, seed, genes, reg, metrics}`) ; rétro-compatible (le CLI ignore
  le retour). Permet au driver d'ablation de réutiliser l'entraînement exact.
- `examples/tahoe/experiments.py` — **fix du no-op IDM** : le terme inverse-dynamics
  passait `idm(z_ctrl, z_pert)` (deux tenseurs **données figées**) → gradient jamais
  remonté dans le prédicteur, donc `IDM(known)` était **byte-identique** à `baseline`.
  Corrigé en `idm(z_ctrl, z_pred)` (transition **prédite**), fidèle à
  `InverseDynamicsLoss` de la lib → le gradient atteint `pred`.

**Fait & vérifié**
- `encoder_ablation --selftest` ✅ (table + `encoder_ablation.png` générés).
- Smoke synthétique IDM (jetable, supprimé après) : `baseline` vs `IDM` ne sont
  **plus identiques** (diff `0 → ≠0`) → prouve que le terme IDM backprop dans `pred`.
- `get_errors` clean sur `main.py`, `encoder_ablation.py`, `experiments.py`.
- Récupéré le `results.json` du job **74812** (lancé sous le compte **`tlecourto`**,
  d'où son absence chez `ascazzola`) → local `artifacts/tahoe/exp/results.json`.

**Résultats — Tahoe perturbation (job 74812, `cache_pert_small`, 3 seeds, skill_meanshift)**
- Ablation pertes : baseline **1.745** · +pathway **1.740** · +perturbsig **1.149** ·
  full **1.159** · IDM(known) **1.745** (= baseline → **c'était le no-op**, désormais corrigé).
- ⚠️ Finding honnête : les pertes bio **dégradent** le skill mean-shift ici (le MSE nu
  est le meilleur). Le deck affiche ~1.20× (≈ full), ce qui masque le baseline plus fort.
- Scaling monotone `1.07 → 1.14 → 1.16` ; zero-shot généralise (seen `1.161` / unseen
  `1.157`) ; screening plausible (Homoharringtonine, Selinexor…).

**À suivre**
- **Relancer** l'ablation `experiments.py` sur cluster → vrais chiffres `IDM(known)`
  (≠ baseline) avant de mettre l'IDM dans le deck.
- Lancer `encoder_ablation.py` sur le cache panel-gènes (`tlecourto/tahoe/cache.pt`)
  → table + figure MLP vs Set-Transformer (3 seeds).
- Câbler `artifacts/tahoe/exp/results.json` dans une figure d'ablation pour le deck.

### 2026-06-20 · itération 2 — Viz HTML + tests + runs cluster réels

**Ajouté**
- `architecture_microbiome.html` + `architecture_tahoe.html` (racine) — pages
  d'architecture interactives (encoder / predictor / énergie + type de donnée +
  « ce que permet le world model »), style cohérent, liées entre elles.
- `examples/tahoe/_make_synth_cache.py` — générateur de **cache Tahoe synthétique**
  (smoke local ; les vraies données vivent sur le cluster).
- Note mémoire repo `/memories/repo/cluster.md` (SSH, $WORK, plafond GPU, staging).

**Fait & vérifié**
- `uv run pytest tests/` → **29 passed**. Gotcha : en `--no-sync` il faut
  `PYTHONPATH=$PWD`. 6 warnings `return-not-None` dans `tests/test_jepa_output_formats.py`.
- Vrai run **microbiome local** 5 ep (GPU) → collapse temporel visible (`skill=0`, `tvar=0`).
- **Cluster** : repo mis à jour `09b65f2 → c33453e` (apporte le fix collapse + le code Tahoe).
- **Microbiome 3 seeds** lancés (jobs `75100`/`75101`/`75102`, full 50 ep).
- **Tahoe stagé** (vraies données, 0 copie) : symlinks `$WORK/tahoe/{cache.pt 2.7G,
  cache_pert.pt 1.5G, cache_pert_small.pt 197M}` → `vivatech-dynamics/tlecourto/tahoe`.
- **Tahoe lancé** : `75125` (cell-state) + `75126` (perturbation).

**Résultats — microbiome seed 1 (`75100`, 50 ep, COMPLETED 4 min 41)**
- World model : `skill_vs_identity` **1.11** (> 1, bat « no-change ») **mais** `tvar ≈ 0`
  et `tvarL` saturé à 0.99 → latent encore **temporellement quasi-plat** ; le
  `TemporalVarianceLoss` à coeff 1.0 **ne lève pas** `tvar` (à monter).
- Baselines (probe sujet-disjoint) : raw `age_r2=0.398` · random-encoder `0.570` ·
  **JEPA entraîné `0.606`** (best, mais gain faible vs random → set-pooling). T1D : échec
  (JEPA `0.345` < hasard).

**À suivre**
- Lancer l'ablation **`tvar0`** (3 seeds) + variante **`tvar_coeff=10`**.
- Analyser Tahoe `75125`/`75126` quand ils finissent (cell-state probe drug/moa/cell_line ; perturb skill).

---

## 0. En une phrase

On a porté la recette **EB-JEPA** (prédiction dans l'espace latent + anti-collapse,
**sans reconstruction**) sur **deux nouvelles modalités biologiques** :

1. **Microbiome-JEPA** — world model action-conditionné sur communautés bactériennes
   intestinales longitudinales (DIABIMMUNE).
2. **Tahoe-JEPA** — JEPA cell-state sur single-cell Tahoe-100M, avec volet
   **perturbation médicamenteuse** (world model conditionné par le médicament).

Les deux réutilisent la classe `JEPA` de la lib ; on a ajouté les **encodeurs**,
**régularisateurs bio**, **datasets**, **exemples**, **tests**, **outillage cluster**
et un **deck jury**.

---

## 1. Ajouts à la librairie `eb_jepa/`

### 1.1 Encodeurs — `eb_jepa/architectures.py`

| Classe | Ligne | Rôle | Modalité |
|---|---|---|---|
| `SetEncoder` | L538 | DeepSets **permutation-invariant**, pooling **pondéré par l'abondance** sur tokens OTU | Microbiome (`f_θ`) |
| `SetTransformer` | L454 | Encodeur d'ensemble **par attention** (alternative au DeepSets) | Single-cell / sets |
| `MultiSourceFusion` | L412 | Init **multi-source par gène** (fusionne plusieurs sources d'embedding de gènes) | Tahoe cell-state |
| `RNNPredictor` | L590 | Predictor GRU **action-conditionné** (`is_rnn=True` → rollout + planning) | World model (`g_φ`) |
| `InverseDynamicsModel` | L635 | IDM : prédit l'action depuis `(z_t, z_{t+1})` | Régularisation dynamique |

> `RNNPredictor` / `InverseDynamicsModel` viennent de la lib ; **`SetEncoder`,
> `SetTransformer`, `MultiSourceFusion` sont nos ajouts.**

### 1.2 Pertes & métriques — `eb_jepa/losses.py`

| Classe / fonction | Ligne | Rôle | Statut |
|---|---|---|---|
| `AlphaDiversityLoss` | L349 | une tête doit retrouver la **diversité de Shannon** depuis le latent | **ajout** (microbiome) |
| `PhyloDispersionLoss` | L377 | distances latentes ≈ **soft-UniFrac** (phylogénie tree-free) | **ajout** (microbiome) |
| `TemporalVarianceLoss` | L422 | std par-dim **le long du temps** ≥ marge → **le fix du collapse temporel** | **ajout** (clé) |
| `effective_rank` | L446 | rang effectif de la covariance latente → **moniteur de collapse de features** | **ajout** (métrique) |
| `PathwayCoherenceLoss` | L472 | prior structurel **gene-program** (différenciateur du cell-state JEPA) | **ajout** (Tahoe) |
| `PerturbationSignatureLoss` | L512 | signature de **perturbation** dans le latent | **ajout** (Tahoe) |
| `BCS` (SIGReg) + `epps_pulley` | L567 / L551 | régularisateur **SIGReg / LeJEPA** (anti-collapse alternatif à VICReg) | **ajout** (Tahoe two-view) |
| `VC_IDM_Sim_Regularizer` | L170 | var + cov + temporal-sim + inverse-dynamics (régularisateur world-model) | lib (réutilisé) |
| `InverseDynamicsLoss` | L134 | MSE entre action prédite par l'IDM et action vraie | lib (réutilisé) |

### 1.3 Datasets — `eb_jepa/datasets/`

| Chemin | Contenu |
|---|---|
| `eb_jepa/datasets/microbiome/precompute.py` | **offline** : streame les données brutes (MicrobeAtlas 19 GB + ProkBERT h5 640 MB + métadonnées DIABIMMUNE) → `cache.pt` (24 MB) |
| `eb_jepa/datasets/microbiome/dataset.py` | **training** : `cache.pt` → fenêtres temporelles fixes → DataLoader (n'importe que `numpy`/`torch`) |
| `eb_jepa/datasets/microbiome/cache.pt` | cache compact embarqué dans le repo (293 sujets · 3348 timepoints · 16 811 OTUs) |
| `eb_jepa/datasets/tahoe/dataset.py` | `TahoeConfig` / `TahoeDataset` / `make_loaders` : single-cell **two-view** (dropout + bruit multiplicatif mimant le dropout de séquençage) |

---

## 2. Exemples créés

### 2.1 `examples/microbiome/` (Layer A + world model temporel)

| Fichier | Rôle |
|---|---|
| `main.py` | wire `JEPA` (SetEncoder + RNNPredictor + VC_IDM_Sim_Regularizer) + pertes bio + probes ; entraîne ; dump `metrics.json` |
| `eval.py` | métriques + figure latent-space + collapse (corrélation / eff-rank) |
| `baselines.py` | **ladder de baselines** — représentation (diversité Shannon · rank-abundance · raw mean-ProkBERT · MLP supervisé · encodeur random multi-seed · JEPA multi-ckpt ±std) **+ table dynamique world-model** (identité · mean-shift · AR linéaire · prédicteur JEPA) |
| `aggregate.py` | agrège les runs d'ablation → table + bar chart (mean ± std) |
| `viz.py` | visualisations |
| `run_ablation.sh` | sweep **4 conditions × 3 seeds** (baseline · div+phylo · tvar · full) |
| `ablation_norm.slurm` | **sbatch optimisé** : ablation normalisation ON/OFF × 3 seeds, **6 runs en parallèle sur 1 GPU** (modèle ~0.7M params → un seul GB200 suffit ; laisse 2 slots libres) |
| `train.slurm` | launcher SLURM (GPU) |
| `cfgs/train.yaml` | config + **leviers ablatables** : `normalize_features`, `tvar_margin`, `residual_predictor`, `pred_coeff` |
| `ARCHITECTURE.md` / `README.md` / `architecture.html` | doc complète + explainer visuel |

### 2.2 `examples/tahoe/` (single-cell + perturbation)

| Fichier | Rôle |
|---|---|
| `main.py` | entraînement du cell-state JEPA (two-view, SIGReg + PathwayCoherenceLoss) ; probe final **raw / pca50 / random-enc / JEPA** |
| `experiments.py` | toutes les études contrôlées du EB-JEPA perturbation |
| `perturb.py` | world model **conditionné par le médicament** (drug-perturbation) |
| `probe_emb.py` | probe linéaire sur embeddings gelés |
| `_check.py` / `_subsample.py` | utilitaires data |
| `run.slurm` / `perturb.slurm` | launchers SLURM |
| `cfgs/train.yaml` | config Tahoe |
| `NEXT_STEPS.md` | roadmap (à lire pour la suite) |
| `slides/` | éléments de deck Tahoe |

---

## 3. Tests ajoutés — `tests/`

| Fichier | Lignes | Couvre |
|---|---|---|
| `tests/test_microbiome.py` | +122 | encodeur / pertes bio / wiring `JEPA` microbiome |
| `tests/test_jepa_output_formats.py` | +832 | contrats de forme des sorties `JEPA` |
| `tests/test_loss_equivalences.py` | +320 | équivalences entre formulations de pertes |
| `tests/planning_test.py` | modifié | planning (aligné sur la signature à marge) |
| `tests/eb_jepa_test.py` | supprimé (−283) | ancien test remplacé |

Lancer : `uv run pytest tests/`.

---

## 4. Outillage cluster / reproductibilité (racine)

| Fichier | Rôle |
|---|---|
| `run_seeds.py` | lance un sweep multi-seeds |
| `setup.sh` / `setup.md` | setup environnement + cluster |
| `env.sh` | détection d'équipe + résolution du work dir |
| `slurm_test.sh` | smoke-test SLURM |
| `cluster/` | `DEV_PROCESS.md`, helpers `gpus` / `log` / `qall` / `sq` / `users`, `README.md` |
| `artifacts/` | `train_log.txt`, `ckpt/microbiome_jepa.pt` |
| `checkpoints/microbiome/microbiome_jepa.pt` | checkpoint entraîné (le nôtre, **pas** un poids pré-livré) |

---

## 5. Slides / livrables jury

- Deck LaTeX jury : architecture **TikZ** + schémas **MultiSourceFusion**.
- Figure d'ablation **collapse** (SIGReg vs aucune régularisation).
- `examples/microbiome/architecture.html` : explainer interactif.
- `examples/tahoe/slides/` : volet single-cell.

---

## 6. Références bibliographiques ajoutées — `references/paper/`

~40 papiers JEPA résumés (un `SUMMARY.md` par papier). Les plus pertinents pour notre récit :

- **Collapse / slow features** : `temporal-straightening`, `jepa-slow-features` (Sobal 2022) → théorie derrière `TemporalVarianceLoss`.
- **Anti-collapse** : `lejepa` (SIGReg/BCS), `reconstruction-or-semantics`.
- **Set / perm-invariant JEPA** : `point-jepa`, `stem-jepa`, `s-jepa`.
- **Bio / séquence JEPA** : `protein-jepa`, `polymer-jepa`, `graph-jepa`.
- **Time-series JEPA** : `ts-jepa`, `mts-jepa`, `t-jepa`.
- **World models** : `leworldmodel`, `stable-worldmodel`, `navigation-world-models`, `v-jepa(2)`.

---

## 7. Résultats à ce jour (honnêtes)

- ✅ **Positif** : la représentation microbiome retrouve l'**horloge d'âge**
  (`age_r2 ≈ 0.50` sur sujets held-out), vérifié contre raw mean-ProkBERT et un
  encodeur **non entraîné** → le gain vient bien de l'entraînement.
- 🔬 **Collapse diagnostiqué (l'insight)** : avec VICReg/IDM seuls, le world model
  **collapse temporellement** (`tvar → 0`, `skill ≤ 1`) — VICReg garde la variance
  *batch/feature* (eff-rank haut) mais pas la variance **temporelle** (slow-feature
  collapse de Sobal et al.).
- 🛠️ **Collapse combattu (le fix)** : `TemporalVarianceLoss` applique la hinge VICReg
  **le long du temps**. Comparaison contrôlée `tvar=0` vs full via `run_ablation.sh` → `aggregate.py`.
- 🔬 **Négatif honnête** : le probe **T1D** reste proche du hasard (peu de positifs,
  signal subtil) — gardé dans le rapport.

### 7.1 Baselines durcies (résolution des lacunes de l'audit)

Après audit des baselines, on a **comblé les trous** pour que tout « on bat X » soit
incontestable. Tout est **réellement exécuté** (aucune valeur inventée).

**Microbiome — `examples/microbiome/baselines.py`** : ladder de représentation complète,
même probe linéaire subject-disjoint (Ridge âge + LogReg T1D) :

| Baseline ajoutée | Type | Répond à |
|---|---|---|
| diversité Shannon (stats) | écologie classique, **sans apprentissage** | « le ML bat-il l'indice de diversité de base ? » |
| rank-abundance (courbe triée) | écologie classique, **sans apprentissage** | « bat-il la forme d'abondance de la communauté ? » |
| MLP supervisé sur raw | **supervisé** (le « MLP à battre » de Susagi) | référence non-linéaire supervisée |
| JEPA multi-ckpt (±std) | notre modèle | **barres d'erreur** → fini le run unique |

(+ `raw mean-ProkBERT` et `encodeur random` déjà présents.)

**+ table dynamique world-model** (espace latent, MSE du next-state) : `identité` ·
`mean-shift global` · `AR linéaire W[z,a]` · `prédicteur JEPA`, avec skill = MSE_id / MSE_modèle.
> Sur trajectoires lentes l'identité est une baseline **dure** ; battre **mean-shift ET l'AR
> linéaire** est la vraie preuve de dynamique. **Résultat réel** (3 vrais checkpoints `full`,
> cluster, moyenne ± std) :
> - **Représentation** : JEPA **age_r2 = 0.526 ± 0.009** > encodeur random **0.390 ± 0.103** >
>   rank-abundance 0.206 / diversité 0.169 / raw −0.063 / MLP −0.434 → **le gain vient du
>   training**, pas de l'architecture (positif net, très stable). T1D reste ~chance (0.51).
> - **Dynamique** (seed1, comparaison **juste** 1-pas) : JEPA **1-pas teacher-forced skill 1.355×**
>   (bat largement « rien ne change » 1.0 et mean-shift 1.002) **mais AR linéaire 1.639×** le dépasse
>   encore ; en **rollout 4-pas** le JEPA retombe à **1.112×** (accumulation d'erreur). *Finding
>   honnête* : le prédicteur bat la persistance mais **pas encore un modèle linéaire**, et perd
>   ~0.24 de skill en rollout. **Fix testé** (prédicteur résiduel z+Δ zero-init + `pred_coeff=5`,
>   `full_res` seed1) = **échec honnête** : JEPA 1-pas **1.289×** (≤ 1.355× d'origine), toujours sous
>   AR linéaire 1.637×, et age_r2 tombé à 0.444. **Cause racine identifiée** : l'encodeur est
>   **temporellement collapsé** (`tvar≈8e-4`) → quasi aucune dynamique à prédire, donc une droite
>   closed-form gagne. **Vrai levier** = casser le collapse temporel (`tvar_margin` 1→4), pas le
>   prédicteur. Config mis à jour ; résiduel/pred_coeff rendus ablatables et remis à OFF/1.

**Tahoe — `examples/tahoe/main.py`** : ajout du **control encodeur-random** au probe in-domain
(`raw / pca50 / random-enc / JEPA`), même logique « training vs inductive bias » que le microbiome.

> Reste ouvert (vrai upper-bound externe) : foundation-model gelé (MosaicFM) **déjà dispo** dans
> `examples/tahoe/probe_emb.py`, à brancher dans le probe in-domain principal.

> ✅ Levier **appliqué** (cf. `ARCHITECTURE.md` §3) : **z-score par dimension** des features
> d'entrée de l'encodeur (`NormSetEncoder` dans `examples/microbiome/main.py` ; stats calculées
> sur le train, stockées en buffers/checkpoint). L'**abondance brute** est gardée pour le poids
> + masque de pooling. Activé par défaut (`model.normalize_features`), **ablatable** on/off.
> (CLR non fait : casserait le rôle poids/masque du canal d'abondance.)
>
> **🎯 Résultat d'ablation (norm ON vs OFF, 30 ép, **3 seeds** — job `75889`) — c'est LE fix du collapse temporel :**
>
> | métrique (moy ± std, 3 seeds) | norm **OFF** | norm **ON** |
> |---|---|---|
> | `tvar` (mouvement temporel) | **0.0001** (collapsé) | **0.030** (~300×) |
> | `skill` world-model (rollout) | **0.80 ± 0.15** (< « rien ne change ») | **1.14 ± 0.11** (bat « rien ») |
> | `age_r2` (horloge d'âge) | **0.27 ± 0.05** | **0.47 ± 0.08** |
> | `effrank` | ~4.4 | ~4.4 |
>
> Sans normalisation, le canal d'abondance écrase la variance VICReg et les latents restent
> **figés dans le temps** (collapse) → le world-model fait *pire* que la persistance ; avec, ils
> bougent (~300×) → il **bat la persistance** et l'horloge d'âge **double**, stable sur 3 seeds.
> Côté dynamique 1-pas, l'écart JEPA↔AR-linéaire se **resserre** (1.26 vs 1.35 avec norm, contre
> 1.31 vs 1.60 sans) : le modèle est enfin dans un régime de **vraie dynamique**. (Produit par
> `ablation_norm.slurm` : 6 runs en parallèle sur **1 seul GPU**.)
>
> **Nuance honnête (le compromis)** : le normalisé **répare la dynamique** (tvar 0→0.03, skill
> 0.80→1.14, il bat enfin la persistance) **et** bat le no-norm en représentation (age 0.27→0.47).
> Mais l'ancien `full` (sans norm, **50 ép**) atteignait age **0.526** vs **0.47** pour le normalisé
> (à **30 ép** seulement). Donc :
> - Le normalisé **échange un peu de représentation statique contre une vraie dynamique temporelle**.
> - C'est un **point de fonctionnement différent**, pas strictement « mieux partout ».
> - À tester : norm à **50 ép** / `tvar_margin=2` pour récupérer l'age **tout en** gardant la dynamique.

### 7.2 Ablation pertes bio + démo collapse (job `76022`, **norm ON, 50 ép, 3 seeds**)

Comparaison contrôlée : une perte activée à la fois, **toutes avec normalisation ON**.
Moyenne ± std sur 3 seeds (1 / 1000 / 10000).

| condition | age_r2 | skill (rollout) | effrank | tvar | rôle |
|---|---|---|---|---|---|
| `baseline` (VICReg+IDM seuls) | 0.41 ± 0.09 | 1.12 ± 0.10 | 4.3 | 0.009 | la **norm seule** donne déjà beaucoup |
| `div+phylo` (pertes bio) | 0.41 ± 0.06 | 1.03 ± 0.10 | 4.7 | 0.012 | rang un peu plus riche |
| `tvar` (variance temporelle) | 0.40 ± 0.11 | **1.24 ± 0.04** | **2.3** ⚠️ | **0.35** | meilleure dynamique **mais collapse de rang** |
| **`full` (div+phylo+tvar)** | 0.41 ± 0.08 | **1.15 ± 0.08** | **4.8** | 0.047 | **meilleur équilibre** (dynamique + rang + T1D **0.77**) |
| `collapse_noreg` (std=cov=0) | 0.43 | 1.01 | **1.0** ⚠️ | 0.020 | **démo collapse** : sans VICReg le rang s'effondre |

**Insights :**
- **Démo collapse (la preuve JEPA, exigée par le guide)** : couper var+cov VICReg (`std=cov=0`)
  → **effrank = 1.0** (les 128 dims latentes s'effondrent sur **1** direction ; T1D tombe à **0.33**
  < hasard). « Sans régularizer → collapse », noir sur blanc.
- **`tvar` seul** maximise la dynamique (skill **1.24**) mais **collapse le rang de features**
  (effrank 2.3) — il force la variance temporelle au détriment de la richesse.
- **`full`** combine le meilleur : dynamique > persistance (skill **1.15**), **rang haut** (4.8),
  **meilleur T1D** (0.77) → la config qui **équilibre les deux types de collapse** (features ET temps).
- **Honnête** : l'`age_r2` ≈ 0.41 quelle que soit la condition → c'est la **normalisation** qui porte
  l'age (cf. §7.1) ; les pertes bio portent surtout **dynamique + rang + T1D**. À 50 ép l'age ne
  monte pas vs 30 ép → ~30 ép suffit (léger sur-apprentissage au-delà sur 293 sujets).

---

### 7.3 Head-to-head **texte (ProkBERT) vs image (FCGR)** (job `76274`, 30 ép, 3 seeds)

**Le bon modèle mental (à ne pas confondre).** L'argument JEPA central : on ne change que le
**front-end** ; l'objectif, le régularizer et le moteur `jepa.py` restent **identiques**.

| Composant | Set-JEPA (microbiome brut) | Image-JEPA (Met2Img) | change ? |
|---|---|---|---|
| Input | ensemble d'embeddings OTU | image `[B,C,H,W]` | ✅ |
| Encodeur | set-transformer | ResNet (CNN, fourni) | ✅ |
| Prédicteur | masked-set | identité (2-vues) | ✅ |
| Augmentations | masquage de membres | crops/bruit image | ✅ |
| Objectif + régularizer | JEPA + VICReg/SIGReg | **identique** | ❌ |
| Moteur (`jepa.py`) | **identique** | **identique** | ❌ |

**Deux variantes « image » distinctes — ne pas les confondre :**
- **(a) FCGR-token (ce que j'ai benché)** : chaque OTU → image FCGR → CNN → token, dans le **même
  pipeline set+temporel (RNNPredictor)** que le texte. C'est un **swap de token-encodeur** propre
  (= l'intégration actuelle de `examples/microbiome2img/main.py`).
- **(b) Met2Img-image (la table ci-dessus, cible conceptuelle)** : **un** Met2Img `[B,C,H,W]` par
  échantillon → **ResNet** → **2-vues** (moteur `examples/image_jepa`). **Pas encore benché.**

**Résultat de (a)** — même communautés, même recette, seul le **token** change :

| token | feature | age_r2 | skill | effrank | tvar | T1D |
|---|---|---|---|---|---|---|
| **ProkBERT** (pré-entraîné) | texte | **0.47 ± 0.08** | **1.14 ± 0.11** | **4.4** | 0.030 | **0.77** |
| **FCGR** (CNN from scratch) | image | -0.01 ± 0.02 | 0.00 | **1.0** ⚠️ | 0.000 | 0.50 |

Figure : [`artifacts/figures/fig4_text_vs_image.png`](artifacts/figures/fig4_text_vs_image.png).

**Lecture honnête :** le **token ProkBERT gagne nettement** ; le **token FCGR a collapsé** (effrank 1.0
sur 3 seeds). **Mais ce n'est pas la qualité de la feature** : (i) le bench taxonomie montre que la FCGR
capture bien la phylogénie (~90 %, cf. fig3) ; (ii) le CNN FCGR est appris **from scratch sans le z-score
par pixel** (le panel `synth.py` z-scoré passe d'effrank **1.0 → 4.6**, cf. §itération 4). Donc :
**collapse d'entraînement, pas de signal**. Le texte part d'embeddings **pré-entraînés + normalisés** →
robuste. **Prochaine étape** : (1) z-scorer le panel FCGR réel ; (2) benché la variante (b) Met2Img/ResNet.

---

## 8. Commandes clés

```bash
# Vérifier l'install
uv run pytest tests/

# --- Microbiome ---
# (1, offline) construire le cache depuis les données brutes
python -m eb_jepa.datasets.microbiome.precompute
# (2) entraînement (smoke)
uv run python -m examples.microbiome.main --fname examples/microbiome/cfgs/train.yaml optim.epochs=2 logging.log_wandb=false
# (3) éval
python -m examples.microbiome.eval --ckpt checkpoints/microbiome/microbiome_jepa.pt
# (3b) ladder de baselines + table dynamique (passer plusieurs ckpts -> ±std)
python -m examples.microbiome.baselines --ckpt checkpoints/microbiome/microbiome_jepa.pt
# (4) ablation contrôlée 4×3 seeds → table + bar chart
bash examples/microbiome/run_ablation.sh
python -m examples.microbiome.aggregate --root <ckpt>/microbiome
# (5) ablation normalisation OPTIMISÉE (6 runs // sur 1 GPU : norm ON/OFF × 3 seeds)
sbatch examples/microbiome/ablation_norm.slurm

# --- Tahoe (single-cell) ---
uv run python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml optim.epochs=2 logging.log_wandb=false
python -m examples.tahoe.experiments --cache $WORK/tahoe/cache_pert.pt --fp artifacts/tahoe/drug_fp.pt --out artifacts/tahoe/exp --epochs 12

# --- Cluster ---
sbatch examples/microbiome/train.slurm
```

---

## 9. Historique « fait » (commits clés, branche `adrien`)

```
c33453e  Merge origin/tristan into adrien
9840ee1  SetTransformer encoder (per-gene multi-source init) + NEXT_STEPS roadmap
3106922  exps (Tahoe)
ec0f6d7  Merge: integrate Tahoe + reconcile collapse fix
b14d918  microbiome: temporal-collapse fix + ablation tooling
28af050  Slide redesign (TikZ archi + multi-source fusion) + MultiSourceFusion module
377c5d3  Ablation figure collapse (SIGReg vs none) → deck jury
58b4bab  Tahoe drug-perturbation EB-JEPA world model + deck jury
994cc16  Tahoe-100M cell-state JEPA (SIGReg + PathwayCoherenceLoss) + leviers anti-collapse microbiome
b0104ef  microbiome: baselines (raw/random/JEPA) + launcher SLURM
3e570e5  microbiome: architecture explainer + viz + artifacts
03518f1  Microbiome-JEPA: world model action-conditionné (commit initial modalité)
```

Cumul depuis le fork : **390 fichiers changés, ~33 k insertions**.

---

## 10. Définition of done (rappel rubric, cf. `CLAUDE.md`)

- [x] Encoder set-JEPA entraîné sans collapse + probe `age_r2 ≈ 0.50`.
- [x] Collapse temporel **diagnostiqué + corrigé** (`TemporalVarianceLoss`), avec moniteurs `tvar` / `effrank`.
- [x] 2e modalité (Tahoe single-cell + perturbation) intégrée, recette transférée.
- [x] **Baselines durcies** : ladder représentation + table dynamique (mean-shift / AR linéaire) + control random Tahoe (cf. §7.1).
- [ ] Ablation 4×3 seeds **finalisée** (table + bar chart mean ± std).
- [x] **Normalisation par feature** (`NormSetEncoder`, z-score par dim, abondance brute préservée) + **ablation norm ON/OFF** : ON **casse le collapse temporel** (tvar 0→0.018, skill 0.15→1.08, age 0.25→0.43, T1D 0.54→0.91) ; confirmation 3 seeds via `ablation_norm.slurm` (job 75889).
- [ ] Rapport 1–2 pages + démo 3 min, runnable en une commande.
