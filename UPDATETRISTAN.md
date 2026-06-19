# UPDATE TRISTAN — récap complet du travail (branche `tristan`)

> Document de synthèse : **tout ce qui a été créé / ajouté, les sources de données, et la logique**
> derrière chaque pièce. Pour la liste des tâches restantes → [examples/tahoe/NEXT_STEPS.md](examples/tahoe/NEXT_STEPS.md).
>
> ⚠️ **Contrainte d'honnêteté (non négociable) :** aucune valeur inventée. Toute comparaison = un
> baseline qu'on a **réellement** fait tourner. On ne prétend **pas** battre GeneJEPA sur son
> protocole ; on mesure proprement nos baselines + nos différenciateurs.

---

## 0. Vue d'ensemble — ce qu'on construit

**Headline : un world-model énergie-based (EB-JEPA) de perturbation médicamenteuse.**
`(cellule contrôle, médicament) → cellule perturbée`, sur Tahoe-100M.

Principe JEPA : on **prédit dans l'espace de représentation** (pas de reconstruction de comptages).
Énergie = `‖ g_φ(f_θ(x), q_ω(a)) − f_θ(x') ‖²  +  λ·R(z)`.
- `f_θ` encoder, `g_φ` predictor (drug-conditionné), `q_ω` action-encoder, `R` anti-collapse.

Deux régimes d'encoder coexistent dans le repo (source de confusion fréquente — clarifié) :
| Modèle | Encoder `f_θ` | Anti-collapse |
|---|---|---|
| **Tahoe perturbation world-model** (headline) | **GELÉ** (embeddings MosaicFM pré-calculés) | non (rien à régulariser) |
| Tahoe représentation (`main.py`, PBMC3k) | **ENTRAÎNÉ** | oui (**SIGReg**) |
| Microbiome SetEncoder | **ENTRAÎNÉ** | oui (VICReg / IDM) |

« Encoder gelé » = on n'exécute **aucun** réseau : l'embedding MosaicFM **EST** l'état `z`, on le lit.

---

## 1. Encoders ajoutés — [eb_jepa/architectures.py](eb_jepa/architectures.py) (+178 lignes)

### `SetTransformer` (Perceiver) — l'encoder à pousser
- Chaque **gène = un token**. Token init =
  `learned-id(gène) + Σ_sources W_s·source_s(gène) + value_proj(expression)`.
- M latents apprennent + **cross-attention** vers les K gènes (O(K·M), scalable), puis self-attn, pooling → `z`.
- **`register_gene_source(name, table[K,d])`** : table **gelée** (scGPT / KGE / ESM2), projection **apprise**.
- **Logique** : c'est ce qui **permet le gene-init multi-sources**. Avec un MLP il n'y a pas de tokens →
  l'init par gène n'a nulle part où vivre. Recette inspirée de GeneJEPA (Perceiver + masked-gene).
- Testé : forward/backward OK, sources scGPT(512-d)+KGE(128-d) s'enregistrent, gradients OK, tables gelées.
- Branché dans `main.py` via `model.encoder=settransformer` + `data.gene_sources=<.pt>`.

### `MultiSourceFusion`
- Fusionne plusieurs sources d'embeddings → un latent, avec **fallback appris par source** (n'importe
  quel sous-ensemble peut être présent). Projection Linear/source + concat + MLP.
- **Logique** : permet de fusionner MosaicFM + scGPT + KGE… (niveau cellule) OU plusieurs représentations
  du **médicament** (fingerprint + descripteurs + drug-KGE) côté action.

### `SetEncoder` (DeepSets, microbiome)
- Encoder pondéré-par-abondance, invariant permutation, sur tokens OTU. Sortie `[B,D,T,1,1]` (contrat eb_jepa).

### `CellEncoder` (MLP) — baseline rapide (dans `examples/tahoe/main.py`).

---

## 2. Losses ajoutées — [eb_jepa/losses.py](eb_jepa/losses.py) (+164 lignes)

| Loss | Domaine | Logique |
|---|---|---|
| **`PathwayCoherenceLoss`** | Tahoe | distances cosine par paires dans le latent ≈ distances d'activité des **modules de gènes** → prior de programme biologique (ce que GeneJEPA n'a pas). |
| **`PerturbationSignatureLoss`** | Tahoe | le **shift prédit** (`z_pert − z_ctrl`) doit être cohérent **par médicament** (supervised-contrastive) → force la signature du médicament. |
| `AlphaDiversityLoss`, `PhyloDispersionLoss`, `TemporalVarianceLoss` | microbiome | priors écologiques + anti-collapse temporel. |

> ⚠️ pas d'`ImposterRepulsionLoss` : choix assumé **pur JEPA sans imposter**.

---

## 3. Datasets & pipeline de précalcul — [eb_jepa/datasets/tahoe/](eb_jepa/datasets/tahoe/)

| Fichier | Rôle |
|---|---|
| `precompute_pert.py` | **cache de perturbation** depuis un shard d'embeddings : `X[N,2560]`, drug, cell_line, `is_control` (auto-détecte DMSO/vehicle/control), centroïdes par lignée (pseudo-contrôle), `drug_fp` (Morgan via RDKit, fallback one-hot), modules (KMeans sur les dims). Sur Dalia : 300k cellules, 95 drugs, 50 lignées, 859 contrôles réels. |
| `pert_dataset.py` | `PertDataset` : paires contrôle→perturbé `[D,2,1,1]` + action `[A,2]`. Contrôle = DMSO aléatoire de la même lignée, sinon centroïde. |
| `precompute_pbmc.py` | cache benchmark **PBMC3k** (type cellulaire dans le slot cell_line). |
| `precompute_emb.py` / `precompute.py` | construction des caches depuis embeddings / shards bruts. |
| `dataset.py` | cache cellule two-view (gene dropout + bruit) pour `main.py`. |

---

## 4. Modèles & drivers — [examples/tahoe/](examples/tahoe/)

| Fichier | Rôle |
|---|---|
| **`perturb.py`** | le **world-model** : `FrozenIdentityEncoder` + `RNNPredictor` via `JEPA.unroll`. Pertes `PerturbationSignatureLoss` + `PathwayCoherenceLoss`. `evaluate()` calcule **skill_vs_identity** & **skill_vs_meanshift**. |
| **`main.py`** | cellule **représentation** JEPA (two-view + SIGReg/VICReg + PathwayCoherence + probe vs raw/PCA). Encoder sélectionnable `mlp`/`settransformer`. |
| **`experiments.py`** | driver batch : ablation `baseline/+pathway/+perturbsig/full/IDM` × seeds {1,1000,10000}, scaling, **zero-shot** (drugs held-out via fingerprint), **screening** in-silico. |
| `slides/` | deck jury LaTeX moderne + scripts de figures (`make_figs.py`, `make_umap.py`, `collapse_demo.py`). |
| `*.slurm` | lanceurs Dalia. |

---

## 5. Sources de données & modèles (références)

| Source | Quoi | Usage |
|---|---|---|
| **Tahoe-100M** | 100M cellules scRNA-seq, ~1000 lignées cancéreuses, ~3000 médicaments | dataset principal |
| **Tahoe-x1 / MosaicFM-3B** | embeddings cellulaires pré-calculés (2560-d) | **notre encoder gelé** |
| **GeneJEPA** (Litman 2025, bioRxiv 2025.10.14.682378) | JEPA transcriptome (Perceiver + Fourier + EMA + VICReg, 4×H100) | **concurrent direct** — comparaison honnête, pas de claim SOTA |
| **eb_jepa** | framework EB-JEPA (encoder/predictor/regularizer/unroll) | base de tout |
| **RDKit** | fingerprints Morgan + descripteurs depuis SMILES | actions médicament (94/95 drugs) |
| **PBMC3k** (scanpy) | benchmark type cellulaire | probe transfert |
| **scGPT / KGE / ESM2** | embeddings gène (à télécharger + aligner) | **sources gene-init** (à faire, cf NEXT_STEPS §3) |

---

## 6. Résultats / findings (honnêtes)

- **Headline positif** : le world-model bat **no-effect** (~1.20×) et **mean-shift** (~1.19×).
- **Motivation** : SSL from-scratch apprend l'**identité cellulaire** (F1 0.93), pas le **médicament** (0.02)
  → justifie le world-model conditionné par l'action.
- **PBMC3k** : 0.92 in-domain (≠ comparable au 0.69 transfert-gelé de GeneJEPA — précisé).
- **Ablation collapse** (`collapse_demo.py`) : SIGReg std 1.14 / acc 0.94 **vs** none std 0.002 / acc 0.43.
- **Microbiome** : finding négatif honnête (collapse temporel persistant malgré TemporalVarianceLoss).

---

## 7. À reprendre de la branche `bnz` (Belgacem) — analysé, à porter

| Pièce | Intérêt pour Tahoe |
|---|---|
| **`eval_collapse.py`** (probe fraîche de décodabilité d'action) | ⭐ **probe de drug-decodability** : décoder le médicament depuis `[z_ctrl,z_pert]` → preuve mécanistique que le modèle retient l'intervention. Branche sur l'ablation IDM. Astuce **floored-standardize** à copier. |
| Framing **IDM + dépendance-au-régime** | résultat mécanistique honnête (IDM rescue +0.23 R² en régime collapse). |
| **`plan_glv.py`** (MPPI latent) | → **planning/screening de médicaments** : choisir le drug qui rapproche la cellule d'un état-cible. La démo "whaouh". |
| `diagnose_planning.py` (oracle/solvability) | rigueur : prouver qu'un négatif vient du task-spec, pas du modèle. |
| Structure **`REPORT.md`** (MEASURED/PENDING, négatifs diagnostiqués) | template narratif qui gagne **sans SOTA**. |
| `SetTransformerEncoder` (PMA pooling + masque, contrat 5D) | cross-polliniser : ajouter pooling PMA + support masque à notre `SetTransformer`. |

À **ne pas** prendre : `ImposterRepulsionLoss` (pur JEPA), simulateur gLV / real-data infants (microbiome-spécifique).

---

## 8. Comment lancer

```bash
# Dalia (1 seul GPU dispo → jobs sérialisent ; UN job batché)
sbatch --wrap "uv run python -u -m examples.tahoe.experiments \
  --cache $WORK/tahoe/cache_pert_small.pt --fp $WORK/tahoe/drug_fp.pt \
  --out $WORK/tahoe/exp --epochs 12"          # → results.json

# Local : encoder set-transformer + (option) sources gene-init
python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml \
  data.cache_path=artifacts/tahoe/cache.pt model.encoder=settransformer \
  model.d_model=128 model.n_latents=24 model.depth=2 optim.epochs=10 \
  [data.gene_sources=artifacts/tahoe/gene_sources.pt]
```

Env local : `~/miniconda3/envs/ml_env/bin/python` (torch 2.8, rdkit, scanpy, umap, pyarrow).
Dalia : `$WORK=/lustre/work/vivatech-dynamics/tlecourto`, partition `defq`, `--reservation=Vivatech`,
venv par arch (`eb_jepa_$ARCH`). pyarrow uniquement sur login (x86) → précalcul sur login, train lit le cache.

---

## 9. Prochaine action recommandée
1. **Porter `eval_collapse` → drug-decodability probe** (petit, gros gain narratif), branché sur l'ablation IDM.
2. **Planning/screening latent** (single-step) = démo.
3. Mirrorer le style **REPORT** dans le deck.

(Détail complet des todos → [examples/tahoe/NEXT_STEPS.md](examples/tahoe/NEXT_STEPS.md).)
