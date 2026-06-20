Checklist de soumission — Hack The World(s) (ESIEE)
🎯 0. Le « Gold Standard » (les 6 incontournables — ce qui fait une grande soumission)

Cadrage clair du problème
(problem framing)

Résultats quantitatifs vs baseline

Un vrai insight / une découverte
(positive OU négative)

Discussion honnête des limites

Au moins UNE étude d'ablation
⚠️
obligatoire

Modèle qui tourne + ≥ 1 checkpoint stable sauvegardé

Reproductible
(

config.yaml
propre)
📦 1. Livrables
Modèle JEPA
world model
entraîné sur ta donnée (microbiome)

Présentation de 10 minutes
(⚠️ pas 3 min — c'est 10 ici)
Code reproductible + checkpoint
🗂️ 2. DATA (slide « Data »)

D'où viennent les données ?
(générées / fournies / trouvées →
citer la source
: Susagi/MicrobeAtlas)

Modalité, taille, difficulté
annoncées

À quoi ça ressemble ?
→ stats + échantillons +
une projection PCA/UMAP

Préparation
: normalisation (⚠️
par feature, obligatoire
), encoding/decoding de la cible, augmentation (oui/non + justifiée)
🏗️ 3. ARCHITECTURE (slide « Architecture »)

Modèle inchangé ou modifié ?
Si modifié :
pourquoi, quoi, intuition + papiers
(cite LeJEPA/SIGReg, Susagi, EB-JEPA)

Nombre de paramètres & modules

Quelles (sous-)losses ? comment équilibrées ?
(VICReg/SIGReg/IDM…)

Collapse observé ?
→ montrer la
courbe d'entraînement

Régularisation & stabilité
discutées
🏋️ 4. TRAINING (slide « Training »)

Setup
: batch size, optimiseur, scheduler de LR, nb d'époques

Itération efficace
: une
proxy metric
pour classer les runs tôt (ex. probe-MSE à l'époque 5)

Comparer sans entraînement complet
: scaling laws / ablations
🔮 5. INFERENCE (slide « Inference »)

Stratégie d'inférence
: Mode 1 (réactif) / Mode 2 (planning) / MCTS / actor ?

Performance vs temps d'inférence
(courbe perf/compute)

Astuces d'inférence
éventuelles
⚠️ Pour un world model microbiome : comment tu « utilises » le modèle (rollouts, prédiction d'état, planning d'intervention) — à définir.

📊 6. EVALUATION (slide « Evaluation » — le cœur de ta note)

Robustesse
: ablations +
stabilité aux seeds
(3 seeds : 1 / 1000 / 10000) + scaling laws

Performance
: score sur dataset +
comparaison à la baseline
(Susagi imposteur)

Reach
: application à une situation réelle / « un pas vers l'AGI »
Métriques :
santé (std + rang effectif), probe (AUROC/bal-acc), tâche (dropout AUROC)
🌟 7. Bonus (pour se démarquer)
Haute performance

Tuning d'hyperparamètres

Analyse extensive de stabilité — surtout visualiser le collapse
(heatmaps de covariance)

Dataset fait main
(non-synthétique) ← le microbiome réel coche cette case ✅

Méthode qui scale à d'autres datasets
du domaine (ex. séries temporelles, single-cell)

World model hiérarchique
🎁 8. Ce qui te démarque déjà (gratuit, à mettre en avant)
Souligner que ton « action = perturbation » est un
vrai world model
(track World-Models)
L'ablation
« sans régularizer → collapse »
= preuve de compréhension JEPA
L'angle
imposteur (Susagi) → JEPA
= before/after net
📌 Les 3 pièges à éviter
Pas d'ablation = soumission incomplète (c'est explicitement exigé).
Présentation = 10 min (pas 3) → prévois plus de contenu/structure.
Ne pas tout implémenter : les organisateurs disent « don't try to implement everything » — un seul axe propre + before/after + 1 ablation > plein d'idées à moitié faites.
Tu veux que je transforme cette checklist en plan de slides (10 min, une slide par section Data→Archi→Training→Inference→Eval) avec, pour chacune, ce que tu dois exactement mettre pour ton projet microbiome ?




