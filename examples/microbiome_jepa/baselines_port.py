"""Faithful (minimal) port of the Susagi downstream BASELINES, for an
apples-to-apples comparison against our set-JEPA linear/MLP probe.

WHY THIS FILE EXISTS
--------------------
WS4 reports "OUR JEPA representation + a probe" vs "THE Susagi baseline" on the
SAME cross-validation split and the SAME metrics (accuracy + macro ROC-AUC). To
keep the comparison honest, the *estimators* and the *CV protocol* here are copied
from the Susagi repo rather than re-invented. The Susagi baseline operates on RAW
features (presence/abundance vectors), NOT on a learned embedding — that is exactly
the point of the comparison (does a JEPA embedding beat raw features under an
identical probe?).

SOURCES MATCHED (read locally at /Users/bnz/Microbiome-Modelling):
  * scripts/infants/predict_env.py
      - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      - LogisticRegression(solver='lbfgs', penalty='l2', C=10,
                           multi_class='multinomial', max_iter=2000, class_weight=None)
      - metrics: accuracy_score + roc_auc_score(multi_class='ovr', average='macro')
        for >2 classes, else roc_auc_score(y, proba[:,1]).
    -> reproduced verbatim in `susagi_logreg_cv`.
  * scripts/*/base_lines_mlp.py  (e.g. scripts/diabimmune/base_lines_mlp.py:_make_mlp)
      - MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam',
                      alpha=1e-4, learning_rate_init=1e-3, max_iter=200, random_state=42)
    -> reproduced in `make_susagi_mlp`. (Susagi wraps it in OneVsRestClassifier for a
       *multilabel* colonisation/dropout task; for the single-label tasks here we use
       the bare MLPClassifier, which is sklearn's native multiclass path. We expose a
       `multilabel` flavour too, see `susagi_mlp_grouped_cv`.)
  * scripts/utils.py:eval_masked_ovr  -> GroupKFold(n_splits=5) for grouped CV.

INTEGRITY NOTE: nothing here fabricates a number. Each function runs sklearn on the
arrays you pass and returns measured metrics. The synthetic smoke (in
`_smoke_probe.py`) exercises every path with random data, so any printed metric there
is a genuine model fit on random features (i.e. ~chance), NOT a microbiome result.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier

# Susagi's global seed (scripts/infants/predict_env.py, scripts/utils.py SEED=42).
SUSAGI_SEED = 42


# ---------------------------------------------------------------------------
# Estimators (verbatim hyper-parameters from the Susagi scripts)
# ---------------------------------------------------------------------------
def make_susagi_logreg() -> LogisticRegression:
    """LogReg head exactly as scripts/infants/predict_env.py uses it.

    NOTE: ``multi_class='multinomial'`` is deprecated/removed-ish in very new
    sklearn; we keep it for fidelity but fall back if the kwarg is rejected.
    """
    kwargs = dict(
        solver="lbfgs",
        penalty="l2",
        C=10,
        class_weight=None,
        max_iter=2000,
    )
    try:
        return LogisticRegression(multi_class="multinomial", **kwargs)
    except TypeError:  # pragma: no cover - newer sklearn dropped the kwarg
        return LogisticRegression(**kwargs)


def make_susagi_mlp(hidden_units: int = 128) -> MLPClassifier:
    """Small MLP exactly as scripts/diabimmune/base_lines_mlp.py:_make_mlp.

    (base_hidden=128; same optimiser / alpha / lr / max_iter / seed.)
    """
    return MLPClassifier(
        hidden_layer_sizes=(int(hidden_units),),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=SUSAGI_SEED,
    )


# ---------------------------------------------------------------------------
# Shared metric helper (SAME definition used by probe_downstream so the
# comparison is apples-to-apples).
# ---------------------------------------------------------------------------
def macro_roc_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Macro one-vs-rest ROC-AUC, matching Susagi's convention.

    Susagi (predict_env.py): for >2 classes use multi_class='ovr', average='macro';
    for binary use roc_auc_score(y, proba[:, 1]). Returns NaN if a fold/test set has
    a single class present (AUC undefined), so callers can nan-average over folds.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if n_classes > 2:
            return float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            )
        return float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# CV protocols
# ---------------------------------------------------------------------------
def _fit_predict_proba(clf, X_tr, y_tr, X_te, classes_all):
    """Fit `clf`, return (y_pred, proba_aligned_to_classes_all).

    Aligns predict_proba columns to the global class set so AUC is well-defined
    even if a train fold is missing a class.
    """
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    proba_local = clf.predict_proba(X_te)
    # Align columns to the full class set (zeros for classes absent in this fold).
    proba = np.zeros((X_te.shape[0], len(classes_all)), dtype=float)
    local_classes = list(getattr(clf, "classes_", classes_all))
    for j_local, c in enumerate(local_classes):
        j_global = int(np.where(classes_all == c)[0][0])
        proba[:, j_global] = proba_local[:, j_local]
    return y_pred, proba


def susagi_logreg_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = SUSAGI_SEED,
) -> Dict[str, float]:
    """StratifiedKFold LogReg baseline (port of predict_env.py main loop).

    Returns mean/std accuracy + mean/std macro ROC-AUC over folds.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes_all = np.unique(y)
    n_classes = len(classes_all)
    n_splits = min(n_splits, _max_folds(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        clf = make_susagi_logreg()
        y_pred, proba = _fit_predict_proba(clf, X[tr], y[tr], X[te], classes_all)
        accs.append(float(accuracy_score(y[te], y_pred)))
        aucs.append(macro_roc_auc(y[te], proba, n_classes))
    return _summ(accs, aucs, n_splits, n_classes, model="logreg")


def susagi_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = SUSAGI_SEED,
    hidden_units: int = 128,
) -> Dict[str, float]:
    """StratifiedKFold MLP baseline (Susagi MLP estimator + the predict_env CV).

    The Susagi MLP scripts use the (128,) MLP under GroupKFold for the *multilabel*
    task; for the single-label downstream tasks we keep their estimator but use the
    same StratifiedKFold protocol as their single-label LogReg head, so the only
    thing that differs from `susagi_logreg_cv` is the classifier.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes_all = np.unique(y)
    n_classes = len(classes_all)
    n_splits = min(n_splits, _max_folds(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        clf = make_susagi_mlp(hidden_units)
        y_pred, proba = _fit_predict_proba(clf, X[tr], y[tr], X[te], classes_all)
        accs.append(float(accuracy_score(y[te], y_pred)))
        aucs.append(macro_roc_auc(y[te], proba, n_classes))
    return _summ(accs, aucs, n_splits, n_classes, model="mlp")


def susagi_mlp_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    hidden_units: int = 128,
) -> Dict[str, float]:
    """GroupKFold MLP baseline (matches scripts/utils.py:eval_masked_ovr grouping).

    Use this when samples share a group (e.g. subject / host) that must not straddle
    the train/test boundary — the leakage-safe protocol Susagi uses for its grouped
    AUC numbers.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes_all = np.unique(y)
    n_classes = len(classes_all)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))

    gkf = GroupKFold(n_splits=n_splits)
    accs, aucs = [], []
    for tr, te in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_susagi_mlp(hidden_units)
        y_pred, proba = _fit_predict_proba(clf, X[tr], y[tr], X[te], classes_all)
        accs.append(float(accuracy_score(y[te], y_pred)))
        aucs.append(macro_roc_auc(y[te], proba, n_classes))
    return _summ(accs, aucs, n_splits, n_classes, model="mlp_grouped")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _max_folds(y: np.ndarray) -> int:
    """Largest #folds StratifiedKFold allows (<= smallest class count), >=2."""
    _, counts = np.unique(y, return_counts=True)
    return int(max(2, min(5, counts.min())))


def _summ(accs, aucs, n_splits, n_classes, model: str) -> Dict[str, float]:
    accs = np.asarray(accs, dtype=float)
    aucs = np.asarray([a for a in aucs], dtype=float)
    valid_auc = aucs[np.isfinite(aucs)]
    return {
        "model": model,
        "n_classes": int(n_classes),
        "n_splits": int(n_splits),
        "acc_mean": float(np.mean(accs)) if accs.size else float("nan"),
        "acc_std": float(np.std(accs)) if accs.size else float("nan"),
        "auc_macro_mean": float(np.mean(valid_auc)) if valid_auc.size else float("nan"),
        "auc_macro_std": float(np.std(valid_auc)) if valid_auc.size else float("nan"),
    }


def run_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    model: str = "logreg",
) -> Dict[str, float]:
    """Single entry: dispatch to the requested Susagi baseline protocol.

    model: "logreg" (StratifiedKFold LogReg), "mlp" (StratifiedKFold MLP), or
    "mlp_grouped" (GroupKFold MLP; requires `groups`).
    """
    if model == "logreg":
        return susagi_logreg_cv(X, y)
    if model == "mlp":
        return susagi_mlp_cv(X, y)
    if model == "mlp_grouped":
        if groups is None:
            raise ValueError("model='mlp_grouped' requires groups=")
        return susagi_mlp_grouped_cv(X, y, groups)
    raise ValueError(f"unknown baseline model={model!r}")
