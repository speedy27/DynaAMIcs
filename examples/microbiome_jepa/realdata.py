"""
M2 — real-data downstream probe on the infant-environment task (self-contained).

Pipeline (the infant task is self-contained in data/infants/, no 20GB corpus needed for the PROBE):
  - per-sample OTU lists      : infants_otus.tsv (sample_id ERR..., otus [B97_...], abundances [...])
  - resolve B97 -> ProkBERT   : prokbert_embeddings.h5 + otus.rename.map1 (the verified resolver)
  - encode with a FROZEN set-JEPA encoder pretrained on the real MicrobeAtlas corpus -> z [N, D]
  - probe                     : linear (LogReg) z -> Env (12 classes), StratifiedKFold, acc + macro AUC
  - Susagi baseline (fair)    : MLP on Susagi's TRUE abundance matrix abundance.csv (taxa x samples)
  - target to beat            : Susagi reported infant-env acc ~0.549, macro AUC ~0.912 (their result files)

NOTES / honesty:
  - tech-invariance is N/A on infants: Instrument is 100% "Illumina MiSeq" (single class). It needs a
    multi-tech set (corpus subset with tech labels); handled separately.
  - the z-score is fit on infant tokens here (the corpus-fit z-score isn't persisted yet) — an
    approximation for a frozen encoder + linear probe; flagged in the JSON.
  - INTEGRITY: every number printed is measured from this run. A random (un-pretrained) encoder gives
    ~chance probe accuracy and is NOT a result — use --checkpoint with a corpus-pretrained encoder.

Run (cluster, after corpus pretraining):
  python -m examples.microbiome_jepa.realdata --checkpoint <enc>/latest.pth.tar \
      --fname examples/microbiome_jepa/cfgs/layerA_real.yaml \
      --data_dir $EBJEPA_DSETS/susagi/data --device cuda
"""

import json
import os
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

from eb_jepa.architectures import SetTransformerEncoder
from eb_jepa.datasets.microbiome.otu_data import (
    build_otu_key_resolver,
    load_otu_rename_map,
    load_prokbert_embeddings,
)
from eb_jepa.datasets.microbiome.transforms import PerDimZScore, clr
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config

logger = get_logger(__name__)
D_EMB = 384
F = 385


def _plist(s):
    return [x.strip() for x in str(s).strip().strip("[]").split(",") if x.strip()]


def load_infant_communities(data_dir, embeddings_h5=None, n_max=256, pseudocount=1e-6, max_samples=None):
    """-> (raw_tokens [N,n_max,F], masks [N,n_max], sample_ids, env_labels). Pre-zscore tokens."""
    inf = os.path.join(data_dir, "infants")
    h5 = embeddings_h5 or os.path.join(data_dir, "model", "prokbert_embeddings.h5")
    rename = os.path.join(data_dir, "microbeatlas", "otus.rename.map1")
    emb, otu_id_to_row = load_prokbert_embeddings(h5)
    emb_keys = set(otu_id_to_row)
    rename_map = load_otu_rename_map(rename)

    otv = pd.read_csv(os.path.join(inf, "infants_otus.tsv"), sep="\t")
    env = pd.read_csv(os.path.join(inf, "meta_withbirth.csv")).set_index("SampleID")["Env"].to_dict()
    if max_samples:
        otv = otv.iloc[:max_samples]

    all_ids = [oid for s in otv["otus"] for oid in _plist(s)]
    resolver = build_otu_key_resolver(all_ids, rename_map, emb_keys)

    toks, masks, sids, labels = [], [], [], []
    n_resolved = []
    for _, row in otv.iterrows():
        sid = row["sample_id"]
        if sid not in env:
            continue
        ids = _plist(row["otus"])
        abus = [float(x) for x in _plist(row["abundances"])]
        rows, cnts = [], []
        for oid, c in zip(ids, abus):
            r = otu_id_to_row.get(resolver.get(oid, oid))
            if r is not None:
                rows.append(r)
                cnts.append(max(c, 0.0))
        if not rows:
            continue
        n_resolved.append(len(rows))
        rows = rows[:n_max]
        cnts_t = torch.tensor(cnts[:n_max], dtype=torch.float32)
        rel = cnts_t / cnts_t.sum().clamp_min(1e-12)
        emb_t = torch.from_numpy(emb[np.asarray(rows)])              # [n, 384]
        clr_ab = clr(rel.unsqueeze(0), pseudocount).squeeze(0)        # [n]
        tok = torch.cat([emb_t, clr_ab.unsqueeze(-1)], dim=-1)        # [n, 385]
        out = torch.zeros(n_max, F)
        m = torch.zeros(n_max, dtype=torch.bool)
        k = min(len(rows), n_max)
        out[:k] = tok[:k]
        m[:k] = True
        toks.append(out)
        masks.append(m)
        sids.append(sid)
        labels.append(env[sid])
    logger.info(f"infant communities: {len(toks)} samples, resolved OTUs/sample "
                f"mean {np.mean(n_resolved):.0f} (cap n_max={n_max})")
    return torch.stack(toks), torch.stack(masks), sids, labels


@torch.no_grad()
def encode_communities(encoder, tokens, masks, zscore, device, bs=128):
    encoder.eval()
    Z = []
    for i in range(0, len(tokens), bs):
        t = zscore.transform(tokens[i:i + bs])               # [b, n_max, F]
        t = t * masks[i:i + bs].unsqueeze(-1).to(t.dtype)
        obs = {"otu": t.unsqueeze(1).to(device), "mask": masks[i:i + bs].unsqueeze(1).to(device)}
        z = encoder(obs)                                     # [b, D, 1, 1, 1]
        Z.append(z.flatten(1).float().cpu().numpy())
    return np.concatenate(Z, 0)


def _macro_auc(y, proba, classes):
    Y = label_binarize(y, classes=classes)
    if Y.shape[1] == 1:  # binary edge case
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(Y, proba, average="macro", multi_class="ovr"))


def cv_classify(X, y, make_model, n_splits=5, seed=42, standardize=True):
    """StratifiedKFold acc + macro OVR AUC (Susagi's protocol)."""
    y = np.asarray(y)
    classes = np.unique(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = make_model()
        m.fit(Xtr, y[tr])
        pred = m.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        try:
            aucs.append(_macro_auc(y[te], m.predict_proba(Xte), classes))
        except Exception:
            aucs.append(float("nan"))
    return {"acc_mean": float(np.mean(accs)), "acc_se": float(np.std(accs, ddof=1) / np.sqrt(n_splits)),
            "auc_mean": float(np.nanmean(aucs)), "auc_se": float(np.nanstd(aucs, ddof=1) / np.sqrt(n_splits))}


def load_susagi_abundance_matrix(data_dir, sample_ids):
    """abundance.csv = taxa x samples relative abundance. Return X[len(sample_ids), n_taxa] aligned."""
    df = pd.read_csv(os.path.join(data_dir, "infants", "abundance.csv"), index_col=0)
    cols = [s for s in sample_ids if s in df.columns]
    X = df[cols].T.to_numpy(dtype=np.float32)  # [n_aligned, n_taxa]
    return X, cols


def run(
    checkpoint: str = None,
    fname: str = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml",
    data_dir: str = None,
    d_model: int = 128,
    n_max: int = 256,
    max_samples: int = None,
    device: str = "cpu",
    out: str = "checkpoints/microbiome_jepa/realdata_infants",
):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    if data_dir is None:
        data_dir = os.path.join(os.environ.get("EBJEPA_DSETS", "."), "susagi", "data")
    cfg = load_config(fname, {"model.d_model": d_model}, quiet=True)

    encoder = SetTransformerEncoder(
        token_dim=F, d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers, dim_feedforward=cfg.model.dim_feedforward,
        dropout=0.0, pool=cfg.model.get("pool", "mean"),
    ).to(dev)
    pretrained = False
    if checkpoint and os.path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location=dev, weights_only=False)
        sd = ck.get("encoder_state_dict") or ck.get("model_state_dict") or ck
        sd = {k.replace("encoder.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing = encoder.load_state_dict(sd, strict=False)
        pretrained = True
        logger.info(f"loaded encoder from {checkpoint} (missing={len(missing.missing_keys)} "
                    f"unexpected={len(missing.unexpected_keys)})")
    else:
        logger.warning("NO checkpoint -> RANDOM encoder; probe accuracy is NOT a result (harness check only).")

    tokens, masks, sids, labels = load_infant_communities(
        data_dir, n_max=n_max, max_samples=max_samples)
    zscore = PerDimZScore().fit(tokens.reshape(-1, F), mask=masks.reshape(-1))
    Z = encode_communities(encoder, tokens, masks, zscore, dev)
    y = np.asarray(labels)
    logger.info(f"encoded Z {Z.shape}; {len(np.unique(y))} Env classes")

    # OUR probe: linear (LogReg) on the frozen JEPA embedding
    jepa_probe = cv_classify(Z, y, lambda: LogisticRegression(max_iter=2000, C=10.0))

    # Susagi baseline: MLP on the TRUE abundance matrix (apples-to-apples, same CV protocol)
    Xb, cols = load_susagi_abundance_matrix(data_dir, sids)
    yb = np.asarray([labels[sids.index(c)] for c in cols])
    baseline = cv_classify(
        Xb, yb, lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42),
        standardize=False)

    res = {
        "task": "infants_env", "pretrained_encoder": pretrained, "n_samples": len(y),
        "n_classes": int(len(np.unique(y))), "n_max": n_max, "d_model": cfg.model.d_model,
        "jepa_linear_probe": jepa_probe, "susagi_mlp_baseline": baseline,
        "susagi_reported": {"acc": 0.549, "macro_auc": 0.912, "source": "Susagi env_predictions.txt (reference)"},
        "zscore_note": "z-score fit on infant tokens (corpus z-score not persisted) — approximation",
        "tech_invariance": "N/A on infants (Instrument = 100% Illumina MiSeq, single class)",
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out, "realdata_infants.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print("\n================ INFANT-ENV downstream probe ================")
    print(f"pretrained_encoder={pretrained} n_samples={len(y)} n_classes={len(np.unique(y))} d_model={cfg.model.d_model}")
    print(f"OUR JEPA linear probe : acc {jepa_probe['acc_mean']:.3f} ± {jepa_probe['acc_se']:.3f} | "
          f"macroAUC {jepa_probe['auc_mean']:.3f} ± {jepa_probe['auc_se']:.3f}")
    print(f"Susagi MLP (abundance): acc {baseline['acc_mean']:.3f} ± {baseline['acc_se']:.3f} | "
          f"macroAUC {baseline['auc_mean']:.3f} ± {baseline['auc_se']:.3f}")
    print(f"Susagi reported (ref) : acc 0.549 | macroAUC 0.912")
    print(f"saved -> {out}/realdata_infants.json")
    if not pretrained:
        print("NOTE: RANDOM encoder — probe is a harness check, NOT a result.")
    return res


if __name__ == "__main__":
    fire.Fire(run)
