"""WS4 — Downstream linear/MLP probe of the Layer-A set-JEPA encoder + a
sequencing-technology-invariance probe, with the Susagi MLP/LogReg baseline as the
apples-to-apples reference.

THE RUBRIC STORY
----------------
Layer A's "result" is: a FROZEN set-JEPA community embedding, probed linearly on a
real Susagi downstream task, beats the Susagi baseline that uses the same probe on
raw features. We additionally report a *sequencing-technology-invariance* probe
(Cell-JEPA's argument): train a classifier to predict the sequencing technology
(HiSeq / MiSeq / 454 ...) from the representation — LOWER accuracy is BETTER, because
it means the embedding carries less technical nuisance. This doubles as the
microbiome analogue of the Sobal et al. "slow distractor" collapse check: if the
encoder collapses onto the batch/technology feature, tech is trivially decodable.

WHAT'S VERIFIED vs NOT
----------------------
* VERIFIED on this Mac (synthetic): the encoder API
  (`SetTransformerEncoder(obs)->[B,D,T,1,1]`), the `OTUSampleDataset(mode="single")`
  obs contract, encoding to `[N, D]`, and every CV/metric path (see `_smoke_probe.py`).
* UNVERIFIED (real data is CLUSTER-ONLY; the 22 GB corpus and the Susagi label CSVs
  are NOT on this machine — only the Susagi *scripts* are):
    - The real OTU corpus parsing (delegated to WS1 `OTUSampleDataset._build_real`,
      itself flagged unverified there).
    - The mapping from a sample's SRS id to a downstream LABEL (infants Env, IBS, ...)
      and to a sequencing-TECHNOLOGY token. The Susagi label files
      (`data/infants/meta_withbirth.csv`, `data/IBS/final_metadata.csv`,
      `data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt`) do not
      exist locally, so `load_real_labels` is implemented to the documented format but
      has NOT been run against real files. Every such spot is marked `# UNVERIFIED`.

Metrics MATCH Susagi (accuracy + macro one-vs-rest ROC-AUC) and share their definition
with `baselines_port.py`, so OUR probe and THEIR baseline are scored identically on
the SAME split.

Fire entry (smoke uses synthetic; no `--checkpoint` needed):
    .venv-cpu/bin/python -m examples.microbiome_jepa.probe_downstream \
        --checkpoint <exp_dir>/latest.pth.tar \
        --fname examples/microbiome_jepa/cfgs/layerA_vicreg.yaml \
        --task infants_env
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fire
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

# Reuse the EXACT Susagi metric + baseline estimators/protocols so OUR probe and
# THEIR baseline are scored identically.
from examples.microbiome_jepa.baselines_port import (
    SUSAGI_SEED,
    macro_roc_auc,
    run_baseline,
)
from eb_jepa.architectures import SetTransformerEncoder
from eb_jepa.datasets.microbiome.otu_data import (
    OTUDatasetConfig,
    OTUSampleDataset,
)

SEED = SUSAGI_SEED  # 42, matching Susagi.


# ===========================================================================
# 1. Encode a labeled OTU-sample set with a (frozen) encoder
# ===========================================================================
@torch.no_grad()
def encode_samples(
    encoder: torch.nn.Module,
    dataset: Dataset,
    device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[dict]]:
    """Encode every sample of a ``mode="single"`` OTUSampleDataset to ``z=f(obs)``.

    The dataset yields ``(obs, meta)`` with obs ``{"otu":[1,N_max,F], "mask":[1,N_max]}``
    (T=1). We batch them into ``{"otu":[B,1,N_max,F], "mask":[B,1,N_max]}``, run the
    encoder (-> ``[B, D, T=1, 1, 1]``) and flatten to ``[B, D]`` (matches main.py's
    `CommunitySSL.embed`).

    Returns:
        Z:     ndarray [N, D] float32 community embeddings.
        metas: list[dict] of per-sample meta dicts (carries labels on the real path).
    """
    encoder.eval()
    n = len(dataset)
    Z_chunks: List[np.ndarray] = []
    metas: List[dict] = []

    for start in range(0, n, batch_size):
        idxs = range(start, min(start + batch_size, n))
        otu = torch.stack([dataset[i][0]["otu"] for i in idxs]).to(device)   # [B,1,N_max,F]
        mask = torch.stack([dataset[i][0]["mask"] for i in idxs]).to(device)  # [B,1,N_max]
        state = encoder({"otu": otu, "mask": mask})  # [B, D, 1, 1, 1]
        z = state.flatten(1).float().cpu().numpy()    # [B, D]
        Z_chunks.append(z)
        for i in idxs:
            m = dataset[i][1]
            metas.append(dict(m) if isinstance(m, dict) else {"meta": m})

    Z = np.concatenate(Z_chunks, axis=0) if Z_chunks else np.zeros((0, 0), np.float32)
    return Z, metas


# ===========================================================================
# 2. Linear (and small-MLP) probe with Susagi-matched CV + metrics
# ===========================================================================
def _probe_logreg() -> LogisticRegression:
    """Our linear probe == Susagi's LogReg head (predict_env.py), so a difference
    in score reflects the REPRESENTATION, not the probe."""
    kwargs = dict(solver="lbfgs", penalty="l2", C=10, class_weight=None, max_iter=2000)
    try:
        return LogisticRegression(multi_class="multinomial", **kwargs)
    except TypeError:  # pragma: no cover
        return LogisticRegression(**kwargs)


def _probe_mlp(hidden_units: int = 128) -> MLPClassifier:
    """Small-MLP probe option (same estimator as Susagi's base_lines_mlp _make_mlp)."""
    return MLPClassifier(
        hidden_layer_sizes=(int(hidden_units),),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=SEED,
    )


def _fit_eval_fold(clf, X_tr, y_tr, X_te, y_te, classes_all, n_classes):
    """Fit + score one fold; returns (accuracy, macro_auc). Standardises features
    inside the fold (fit on train only — no leakage), which is standard for a linear
    probe and harmless for the MLP."""
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    # Align proba columns to the global class set (robust to a class missing in train).
    proba_local = clf.predict_proba(X_te)
    proba = np.zeros((X_te.shape[0], n_classes), dtype=float)
    for j_local, c in enumerate(list(clf.classes_)):
        j_global = int(np.where(classes_all == c)[0][0])
        proba[:, j_global] = proba_local[:, j_local]
    auc = macro_roc_auc(y_te, proba, n_classes)
    return acc, auc


def linear_probe(
    Z: np.ndarray,
    y: Sequence,
    groups: Optional[Sequence] = None,
    task: str = "clf",
    probe: str = "linear",
    n_splits: int = 5,
    seed: int = SEED,
    hidden_units: int = 128,
) -> Dict[str, float]:
    """Cross-validated probe of representation Z against labels y.

    Protocol (matches Susagi):
      * grouped CV (``GroupKFold``) when ``groups`` is given (leakage-safe; matches
        scripts/utils.py:eval_masked_ovr), else stratified CV (``StratifiedKFold``
        with shuffle, random_state=seed; matches predict_env.py).
    Metrics (matches Susagi):
      * accuracy + macro one-vs-rest ROC-AUC (binary -> AUC on proba[:, 1]).
    Args:
      task: "clf" (the only supported downstream type here; regression downstreams
            are out of scope for the Susagi tasks). Kept for API symmetry / future use.
      probe: "linear" (LogReg, == Susagi head) or "mlp" (small MLP probe option).
    Returns mean/std accuracy + mean/std macro AUC over folds.
    """
    if task != "clf":
        raise ValueError("linear_probe currently supports task='clf' only "
                         "(Susagi downstream tasks are classification).")
    Z = np.asarray(Z, dtype=np.float64)
    y_enc = LabelEncoder().fit_transform(np.asarray(y))
    classes_all = np.unique(y_enc)
    n_classes = len(classes_all)

    def new_clf():
        return _probe_mlp(hidden_units) if probe == "mlp" else _probe_logreg()

    accs, aucs = [], []
    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        k = max(2, min(n_splits, n_groups))
        splitter = GroupKFold(n_splits=k).split(Z, y_enc, groups=groups)
    else:
        # StratifiedKFold can't exceed the smallest class count.
        _, counts = np.unique(y_enc, return_counts=True)
        k = max(2, min(n_splits, int(counts.min())))
        splitter = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=seed
        ).split(Z, y_enc)

    used = 0
    for tr, te in splitter:
        if len(np.unique(y_enc[tr])) < 2:  # degenerate train fold
            continue
        acc, auc = _fit_eval_fold(
            new_clf(), Z[tr], y_enc[tr], Z[te], y_enc[te], classes_all, n_classes
        )
        accs.append(acc)
        aucs.append(auc)
        used += 1

    accs = np.asarray(accs, dtype=float)
    valid_auc = np.asarray([a for a in aucs if np.isfinite(a)], dtype=float)
    return {
        "probe": probe,
        "cv": "grouped" if groups is not None else "stratified",
        "n_splits": int(used),
        "n_classes": int(n_classes),
        "n_samples": int(Z.shape[0]),
        "feat_dim": int(Z.shape[1]),
        "acc_mean": float(np.mean(accs)) if accs.size else float("nan"),
        "acc_std": float(np.std(accs)) if accs.size else float("nan"),
        "auc_macro_mean": float(np.mean(valid_auc)) if valid_auc.size else float("nan"),
        "auc_macro_std": float(np.std(valid_auc)) if valid_auc.size else float("nan"),
    }


# ===========================================================================
# 3. Sequencing-technology-INVARIANCE probe
# ===========================================================================
def tech_invariance(
    Z: np.ndarray,
    tech_labels: Sequence,
    n_splits: int = 5,
    seed: int = SEED,
    probe: str = "linear",
) -> Dict[str, float]:
    """Decodability of the sequencing TECHNOLOGY from a representation.

    Trains a classifier (default LogReg, same as our linear probe) to predict the
    tech label (HiSeq/MiSeq/454/...) from Z via stratified CV. LOWER accuracy/AUC =
    LESS technical nuisance carried by the representation = BETTER. Also returns the
    majority-class rate as the trivial floor to compare against (acc==majority ->
    tech is NOT linearly decodable, the ideal).

    Accepts ANY ``[N, D]`` rep + labels, so it can score our JEPA rep vs the Susagi
    imposter rep with one call each.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(tech_labels))
    classes, counts = np.unique(y, return_counts=True)
    majority = float(counts.max() / counts.sum())
    out = linear_probe(Z, y, groups=None, task="clf", probe=probe,
                       n_splits=n_splits, seed=seed)
    return {
        "tech_acc_mean": out["acc_mean"],
        "tech_acc_std": out["acc_std"],
        "tech_auc_macro_mean": out["auc_macro_mean"],
        "tech_majority_rate": majority,
        # >0 means tech IS decodable above the trivial floor (worse / more nuisance).
        "tech_acc_above_majority": (
            out["acc_mean"] - majority if np.isfinite(out["acc_mean"]) else float("nan")
        ),
        "n_tech_classes": int(len(classes)),
        "probe": probe,
    }


# ===========================================================================
# 4. Real-data label loading (UNVERIFIED — cluster-only files)
# ===========================================================================
# The Susagi downstream label files are NOT on this Mac; the loaders below mirror the
# documented formats from the Susagi scripts but have NOT been run against real files.
# Each task returns: srs_to_label (dict), and (optionally) a groups key.
def load_real_labels(task: str, data_dir: str) -> Dict[str, dict]:
    """Return ``{sample_key: {"label":..., "group":..., "tech":...}}`` for a task.

    UNVERIFIED. Implemented to the formats documented in the Susagi clone:
      * infants_env: data/infants/meta_withbirth.csv with columns SampleID, Env
        (predict_env.py:load_infants_meta reads SampleID->Env).  group=SampleID.
      * ibs:         data/IBS/final_metadata.csv with run_id,country,ibs; binary label
        from 'Diagnosed by a medical professional' vs 'I do not have this condition',
        excluding 'Self-diagnosed' (predict_ibs.py:load_ibs_metadata). group=country.
    Sequencing-tech token (for tech_invariance) comes from
    data/microbeatlas/sample_terms_mapping_combined_dany_og_biome_tech.txt parsed like
    scripts/utils.py:parse_run_terms (run_id -> [terms]); the tech token is matched
    heuristically against a known vocabulary. This whole function is best-effort and
    flagged UNVERIFIED.
    """
    import csv

    labels: Dict[str, dict] = {}
    task = task.lower()

    if task in ("infants_env", "infants", "env"):
        path = os.path.join(data_dir, "infants", "meta_withbirth.csv")  # UNVERIFIED path
        if not os.path.exists(path):
            raise FileNotFoundError(f"[UNVERIFIED real path] missing infants meta: {path}")
        with open(path, "r", errors="replace") as f:
            reader = csv.DictReader(f)
            # predict_env.py loads SampleID -> Env via load_infants_meta.
            id_col = "SampleID"
            label_col = "Env"
            for row in reader:
                sid = (row.get(id_col) or "").strip()
                env = (row.get(label_col) or "").strip()
                if sid and env:
                    labels[sid] = {"label": env, "group": sid, "tech": None}

    elif task in ("ibs",):
        path = os.path.join(data_dir, "IBS", "final_metadata.csv")  # UNVERIFIED path
        if not os.path.exists(path):
            raise FileNotFoundError(f"[UNVERIFIED real path] missing IBS meta: {path}")
        with open(path, "r", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("run_id") or "").strip()
                country = (row.get("country") or "").strip()
                ibs = (row.get("ibs") or "").strip()
                if not rid or not country or not ibs or "Self-diagnosed" in ibs:
                    continue
                if ibs not in {
                    "I do not have this condition",
                    "Diagnosed by a medical professional (doctor, physician assistant)",
                }:
                    continue
                lab = 1 if "Diagnosed by a medical professional" in ibs else 0
                labels[rid] = {"label": lab, "group": country, "tech": None}
    else:
        raise ValueError(f"unknown real task={task!r} (expected infants_env | ibs)")

    return labels


# Known sequencing-technology vocabulary (lowercase) for the heuristic tech parse.
_TECH_VOCAB = ("hiseq", "miseq", "novaseq", "nextseq", "454", "pyrosequencing",
               "iontorrent", "ion_torrent", "pacbio", "nanopore", "illumina", "sanger")


def parse_tech_from_terms(terms: Sequence[str]) -> Optional[str]:
    """Best-effort sequencing-tech token from a list of free-text terms.

    UNVERIFIED. Susagi's sample_terms file mixes biome + technology terms (filename
    ...og_biome_tech.txt); there is no canonical extractor in the repo, so we match
    against a known vocabulary and return the first hit (else None).
    """
    low = [str(t).strip().lower() for t in terms]
    for t in low:
        if t in _TECH_VOCAB:
            return t
    # substring fallback (e.g. "illumina hiseq 2000")
    for t in low:
        for v in _TECH_VOCAB:
            if v in t:
                return v
    return None


# ===========================================================================
# 5. Build a labeled single-mode dataset (synthetic fallback or real)
# ===========================================================================
def _synthetic_labeled_dataset(
    n_samples: int = 240,
    n_max: int = 64,
    n_classes: int = 3,
    n_tech: int = 2,
    seed: int = SEED,
) -> Tuple[OTUSampleDataset, np.ndarray, np.ndarray, np.ndarray]:
    """A synthetic ``mode='single'`` dataset + fake (label, group, tech) arrays.

    The labels/tech are independent random draws (so a probe scores ~chance) — this
    exercises every code path WITHOUT real data and WITHOUT pretending to be a real
    microbiome result.
    """
    cfg = OTUDatasetConfig(
        synthetic=True, mode="single", n_max=n_max, synth_n_samples=n_samples,
        synth_seed=seed,
    )
    ds = OTUSampleDataset(cfg)
    rng = np.random.default_rng(seed)
    n = len(ds)
    y = rng.integers(0, n_classes, size=n)
    tech = rng.integers(0, n_tech, size=n)
    groups = rng.integers(0, max(2, n // 8), size=n)  # synthetic host groups
    return ds, y, groups, tech


def _build_dataset_and_labels(
    task: str,
    data_dir: Optional[str],
    embeddings_h5: Optional[str],
    n_max: int,
    synth_n_samples: int,
    seed: int,
):
    """Return (dataset[single], y, groups, tech) for `task`.

    Synthetic when no real data_dir/embeddings_h5 resolves; otherwise the UNVERIFIED
    real path (build OTUSampleDataset real + align with load_real_labels by SRS).
    """
    real = OTUSampleDataset._real_available(
        OTUDatasetConfig(data_dir=data_dir, embeddings_h5=embeddings_h5)
    )
    if not real:
        if task != "synthetic":
            print(f"[probe] no real data_dir/embeddings_h5 -> SYNTHETIC fallback "
                  f"(task={task!r} labels are random; metrics ~= chance).")
        return _synthetic_labeled_dataset(
            n_samples=synth_n_samples, n_max=n_max, seed=seed
        )

    # ---- UNVERIFIED real path (cluster only) ----
    print(f"[probe] real data detected at data_dir={data_dir!r} "
          f"embeddings_h5={embeddings_h5!r} — REAL PATH IS UNVERIFIED ON THIS MACHINE.")
    cfg = OTUDatasetConfig(
        data_dir=data_dir, embeddings_h5=embeddings_h5, synthetic=False,
        mode="single", n_max=n_max, synth_n_samples=synth_n_samples,
    )
    ds = OTUSampleDataset(cfg)  # _build_real attaches meta={"srs": srs}
    label_map = load_real_labels(task, data_dir)  # UNVERIFIED
    # Align dataset samples (by SRS) to labels; keep only labeled samples.
    keep_idx, y, groups, tech = [], [], [], []
    for i in range(len(ds)):
        srs = ds[i][1].get("srs")
        rec = label_map.get(srs)
        if rec is None:
            continue
        keep_idx.append(i)
        y.append(rec["label"])
        groups.append(rec.get("group", srs))
        tech.append(rec.get("tech"))
    if not keep_idx:
        raise RuntimeError(
            "[UNVERIFIED real path] no dataset SRS matched the task label file; "
            "check the SRS<->label join."
        )
    ds = torch.utils.data.Subset(ds, keep_idx)
    return ds, np.asarray(y), np.asarray(groups), np.asarray(tech, dtype=object)


# ===========================================================================
# 6. Rebuild encoder from training cfg + load checkpoint
# ===========================================================================
def load_encoder_from_checkpoint(
    checkpoint: Optional[str],
    fname: str,
    device,
    overrides: Optional[dict] = None,
) -> SetTransformerEncoder:
    """Rebuild the SetTransformerEncoder from the training YAML (`fname`) EXACTLY as
    main.py does, then load ``encoder_state_dict`` from `checkpoint`.

    main.py saves both ``model_state_dict`` (the CommunitySSL wrapper) and
    ``encoder_state_dict`` (the bare encoder) — we load the latter so we probe the
    representation, not the projector. If `checkpoint` is None (smoke), a random-init
    encoder is returned (clearly logged).
    """
    from eb_jepa.training_utils import load_config

    cfg = load_config(fname, overrides or None, quiet=True)
    encoder = SetTransformerEncoder(
        token_dim=cfg.model.token_dim,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        pool=cfg.model.pool,
    ).to(device)

    if checkpoint is None:
        print("[probe] no --checkpoint given -> RANDOM-INIT encoder "
              "(metrics are a sanity check, NOT a trained result).")
        return encoder, cfg

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("encoder_state_dict")
    if sd is None:
        # Fallback: strip an "encoder." prefix off the wrapper's model_state_dict.
        msd = ckpt.get("model_state_dict", {})
        sd = {k[len("encoder."):]: v for k, v in msd.items() if k.startswith("encoder.")}
        if not sd:
            raise KeyError(
                f"checkpoint {checkpoint} has neither 'encoder_state_dict' nor an "
                f"'encoder.'-prefixed 'model_state_dict'."
            )
        print("[probe] loaded encoder weights from 'encoder.'-prefixed model_state_dict.")
    info = encoder.load_state_dict(sd, strict=False)
    if getattr(info, "missing_keys", None):
        print(f"[probe] load_state_dict missing keys (first 5): {info.missing_keys[:5]}")
    if getattr(info, "unexpected_keys", None):
        print(f"[probe] load_state_dict unexpected keys (first 5): {info.unexpected_keys[:5]}")
    print(f"[probe] loaded encoder_state_dict from {checkpoint}")
    return encoder, cfg


# ===========================================================================
# 7. Fire entry
# ===========================================================================
def run(
    checkpoint: Optional[str] = None,
    fname: str = "examples/microbiome_jepa/cfgs/layerA_vicreg.yaml",
    task: str = "synthetic",
    probe: str = "linear",
    data_dir: Optional[str] = None,
    embeddings_h5: Optional[str] = None,
    n_max: int = 64,
    synth_n_samples: int = 240,
    device: str = "cpu",
    out_json: Optional[str] = None,
    run_baseline_too: bool = True,
    seed: int = SEED,
) -> dict:
    """Probe a trained encoder on a downstream task + tech-invariance, vs the baseline.

    Args:
      checkpoint: path to a main.py checkpoint (loads ``encoder_state_dict``). If None,
        uses a random-init encoder (sanity only).
      fname: training YAML to rebuild the encoder identically to main.py.
      task: "synthetic" (CPU smoke, random labels) | "infants_env" | "ibs" (real,
        UNVERIFIED, needs data_dir).
      probe: "linear" (LogReg, == Susagi head) | "mlp" (small-MLP probe).
      data_dir / embeddings_h5: real corpus location (cluster only). Absent -> synthetic.
      run_baseline_too: also run the Susagi raw-feature baseline on the SAME labels for
        an apples-to-apples comparison. The baseline X is the masked-mean of the raw
        (z-scored) OTU tokens per sample (a cheap raw-feature vector available without
        the original abundance matrix; flagged as such).
      out_json: optional path to dump the metrics JSON.
    Returns the metrics dict (also printed).
    """
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    encoder, cfg = load_encoder_from_checkpoint(checkpoint, fname, dev)
    # Keep n_max consistent with the encoder's training contract when possible.
    n_max = int(getattr(cfg.data, "n_max", n_max)) if hasattr(cfg, "data") else n_max

    ds, y, groups, tech = _build_dataset_and_labels(
        task=task, data_dir=data_dir, embeddings_h5=embeddings_h5,
        n_max=n_max, synth_n_samples=synth_n_samples, seed=seed,
    )

    Z, _metas = encode_samples(encoder, ds, dev)
    print(f"[probe] encoded Z shape = {Z.shape}; labels n={len(y)}; "
          f"classes={sorted(set(np.asarray(y).tolist()))}")

    # Downstream probe of OUR representation.
    use_groups = groups if (groups is not None and len(np.unique(groups)) >= 2) else None
    jepa_probe = linear_probe(Z, y, groups=use_groups, task="clf", probe=probe, seed=seed)

    # Sequencing-technology-invariance of OUR representation (lower = better).
    tech_metrics = (
        tech_invariance(Z, tech, seed=seed)
        if tech is not None and len(np.unique(np.asarray(tech, dtype=object))) >= 2
        else {"tech_acc_mean": float("nan"), "note": "no/insufficient tech labels"}
    )

    results = {
        "task": task,
        "checkpoint": checkpoint,
        "fname": fname,
        "probe": probe,
        "encoder_dim": int(Z.shape[1]),
        "jepa_probe": jepa_probe,
        "tech_invariance": tech_metrics,
        "real_data_used": bool(
            OTUSampleDataset._real_available(
                OTUDatasetConfig(data_dir=data_dir, embeddings_h5=embeddings_h5)
            )
        ),
    }

    if run_baseline_too:
        # Susagi raw-feature baseline on the SAME labels & (grouped/stratified) split.
        # X_raw = masked-mean of the raw z-scored OTU tokens per sample (a raw-feature
        # surrogate available without the original abundance matrix). On the real path
        # this is the per-sample mean ProkBERT+CLR token, NOT the encoder output.
        X_raw = _raw_feature_matrix(ds)
        base_model = "mlp_grouped" if use_groups is not None else "logreg"
        try:
            baseline = run_baseline(
                X_raw, y, groups=(groups if base_model == "mlp_grouped" else None),
                model=base_model,
            )
        except Exception as e:  # never let the baseline crash the probe report
            baseline = {"error": f"{type(e).__name__}: {e}"}
        results["susagi_baseline"] = baseline
        results["baseline_feature"] = "masked-mean raw z-scored OTU token (raw surrogate)"

    print(json.dumps(results, indent=2, default=str))

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[probe] wrote {out_json}")

    return results


@torch.no_grad()
def _raw_feature_matrix(dataset: Dataset) -> np.ndarray:
    """Per-sample masked-mean of the raw (z-scored) OTU tokens -> [N, F].

    This is the raw-feature surrogate the Susagi baseline runs on (no encoder). For a
    `single`-mode dataset, obs["otu"] is [1, N_max, F] already z-scored with pad slots
    zeroed; the masked mean over real OTUs gives a fixed-length raw descriptor.
    """
    n = len(dataset)
    rows = []
    for i in range(n):
        obs = dataset[i][0]
        otu = obs["otu"][0]            # [N_max, F]
        mask = obs["mask"][0].float()  # [N_max]
        denom = mask.sum().clamp(min=1.0)
        rows.append((otu * mask.unsqueeze(-1)).sum(0).numpy() / denom.item())
    return np.stack(rows, axis=0) if rows else np.zeros((0, 0), np.float32)


if __name__ == "__main__":
    fire.Fire(run)
