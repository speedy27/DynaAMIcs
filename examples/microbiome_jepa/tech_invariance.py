"""
B — sequencing-technology invariance of the JEPA community representation (a rubric-cited metric).

THESIS: a good microbiome representation should encode BIOLOGY (biome / host site) while being INVARIANT
to the sequencing TECHNOLOGY (a technical nuisance). We test this directly on the real MicrobeAtlas
corpus, where the dominant technical-nuisance axis is the library strategy: AMPLICON (16S) vs WGS
(shotgun) — two protocols that yield very different OTU profiles for the same community.

PROTOCOL (the "SRS->tech join"):
  * corpus sample key  = "<RunID>.<SRS>"  (e.g. SRR2459896.SRS1074972)  [samples-otus.97.mapped]
  * RunID -> Terms     = free-text tokens incl. strategy {amplicon,wgs,rnaseq} + biome {gut,soil,...}
                         [sample_terms_mapping_combined_dany_og_biome_tech.txt]
  => label each corpus sample by joining its RunID to Terms; keep samples with EXACTLY ONE of
     {amplicon,wgs} (clean tech label) and >=1 biome token.
  * encode each community with the FROZEN corpus-pretrained set-JEPA encoder -> JEPA rep.
  * TECH probe (LogReg, 5-fold): recover amplicon-vs-wgs from a rep. LOWER accuracy = MORE tech-invariant
    = BETTER. Compare JEPA rep vs baselines that DO retain tech: (a) raw mean-pooled input tokens,
    (b) a random-init encoder of the same architecture. Report chance (majority class).
  * BIOME probe (LogReg, 5-fold): recover biome from the SAME rep. HIGHER = keeps biology (control:
    a tech-invariant rep should still be biome-informative, else it just lost all information).

INTEGRITY: every number is measured from this run. A random-init-encoder rep is a real baseline (not a
result for our model). We report the per-rep tech accuracy, biome accuracy, chance, and class balance.

Run (cluster, needs the corpus + a corpus-pretrained encoder):
  python -m examples.microbiome_jepa.tech_invariance \
      --checkpoint $WORK/checkpoints/microbiome_jepa/realenc/latest.pth.tar \
      --fname examples/microbiome_jepa/cfgs/layerA_real.yaml --data_dir $EBJEPA_DSETS/susagi/data \
      --per_class_cap 2500 --device cpu
"""

import json
import os
from pathlib import Path

import fire
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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

STRATEGIES = ["amplicon", "wgs"]                         # the tech axis (clean binary)
BIOMES = ["gut", "soil", "water", "marine", "skin", "oral", "feces", "stool",
          "sediment", "plant", "rhizosphere", "freshwater"]  # biome tokens to label the "biology" control


def load_runid_labels(terms_path, max_lines=None):
    """One pass over the Terms file -> {RunID: (strategy, biome)} for cleanly-labelled runs.

    Keep a run iff its term set contains EXACTLY ONE of STRATEGIES (clean tech label). Biome = the first
    matching BIOMES token (or None). Returns the dict + counters."""
    strat_set = set(STRATEGIES)
    labels = {}
    n_seen = n_kept = 0
    with open(terms_path, "r", errors="replace") as fh:
        next(fh, None)  # header: RunID\tTerms
        for line in fh:
            n_seen += 1
            if max_lines and n_seen > max_lines:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            runid = parts[0]
            terms = set(t.strip().lower() for t in parts[1:] if t.strip())
            strat_hit = strat_set & terms
            if len(strat_hit) != 1:
                continue
            biome = next((b for b in BIOMES if b in terms), None)
            labels[runid] = (next(iter(strat_hit)), biome)
            n_kept += 1
    logger.info(f"[terms] scanned {n_seen} runs, kept {n_kept} with a clean single strategy label")
    return labels


def stream_labeled_communities(mapped_path, runid_labels, n_max, per_class_cap):
    """Stream the corpus; collect (otu97,count) lists for samples whose RunID has a clean tech label,
    BALANCED to per_class_cap per strategy. Returns list of dicts {runid, srs, strat, biome, otus}."""
    per_class = {s: 0 for s in STRATEGIES}
    target = per_class_cap * len(STRATEGIES)
    out, cur = [], None
    with open(mapped_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None and cur["otus"]:
                    out.append(cur)
                cur = None
                header = line[1:].split()[0]
                runid = header.split(".")[0]
                srs = header.split(".")[-1]
                lab = runid_labels.get(runid)
                if lab is None:
                    continue
                strat, biome = lab
                if per_class[strat] >= per_class_cap:
                    continue
                per_class[strat] += 1
                cur = {"runid": runid, "srs": srs, "strat": strat, "biome": biome, "otus": []}
                if sum(per_class.values()) >= target:
                    # finish current sample then stop on next header
                    pass
            elif cur is not None:
                fields = line.split()
                if not fields:
                    continue
                triplet = fields[0]
                cnt = 0.0
                if len(fields) > 1:
                    try:
                        cnt = float(fields[1])
                    except ValueError:
                        cnt = 0.0
                otu97 = next((t for t in triplet.split(";") if t.startswith("97_")), None)
                if otu97 is None:
                    p = triplet.split(";")
                    otu97 = p[-1] if p else triplet
                cur["otus"].append((otu97, cnt))
                # stop once both classes are full (and we've started a fresh non-counted sample)
            if all(per_class[s] >= per_class_cap for s in STRATEGIES) and cur is None:
                break
    if cur is not None and cur["otus"]:
        out.append(cur)
    logger.info(f"[corpus] collected {len(out)} labelled communities; per-strategy {per_class}")
    return out


def build_tokens(samples, emb, otu_id_to_row, resolver, n_max, pseudocount=1e-6):
    """-> raw_tokens [N,n_max,F], masks [N,n_max], raw_meanpool [N,F] (pre-zscore mean over present)."""
    N = len(samples)
    toks = torch.zeros(N, n_max, F)
    masks = torch.zeros(N, n_max, dtype=torch.bool)
    for i, s in enumerate(samples):
        rows, counts = [], []
        for oid, cnt in s["otus"]:
            r = otu_id_to_row.get(resolver.get(oid, oid))
            if r is not None:
                rows.append(r)
                counts.append(max(float(cnt), 0.0))
        if not rows:
            continue
        rows = rows[:n_max]
        cnts = torch.tensor(counts[:n_max], dtype=torch.float32)
        rel = cnts / cnts.sum().clamp_min(1e-12)
        emb_t = torch.from_numpy(emb[np.asarray(rows)])
        clr_ab = clr(rel.unsqueeze(0), pseudocount).squeeze(0)
        tok = torch.cat([emb_t, clr_ab.unsqueeze(-1)], dim=-1)   # [n,F]
        k = min(len(rows), n_max)
        toks[i, :k] = tok[:k]
        masks[i, :k] = True
    return toks, masks


@torch.no_grad()
def encode(encoder, tokens, masks, zscore, device, bs=128):
    encoder.eval()
    Z = []
    for i in range(0, len(tokens), bs):
        t = zscore.transform(tokens[i:i + bs]) * masks[i:i + bs].unsqueeze(-1).to(torch.float32)
        obs = {"otu": t.unsqueeze(1).to(device), "mask": masks[i:i + bs].unsqueeze(1).to(device)}
        Z.append(encoder(obs).flatten(1).float().cpu().numpy())
    return np.concatenate(Z, 0)


def raw_meanpool(tokens, masks, zscore):
    """Mean of z-scored present tokens per sample -> [N,F]. The 'raw input' baseline (no learned pooling)."""
    t = zscore.transform(tokens) * masks.unsqueeze(-1).to(torch.float32)
    s = t.sum(1)
    n = masks.sum(1, keepdim=True).clamp_min(1).to(torch.float32)
    return (s / n).numpy()


@torch.no_grad()
def susagi_imposter_rep(tokens, masks, repo, ckpt_path, device, bs=128,
                        d_model=100, nhead=5, num_layers=5, dim_ff=400):
    """The SUSAGI imposter model's per-sample rep = mean-pooled ENCODER hidden state (OTU-only forward).

    Their MicrobiomeTransformer (Microbiome-Modelling/model.py) projects RAW ProkBERT OTU embeddings
    (type1, 384-d) and runs a TransformerEncoder, then emits a per-OTU imposter SCORE. There is no
    sample-level head, so we take the masked mean of the encoder output `x` (pre score-projection) as the
    community embedding — the natural analog of our pooled JEPA community vector. We feed ONLY the raw
    ProkBERT embeddings (tokens[...,:384], pre-z-score), matching their type1 input (no abundance, no
    text). conf00.txt: d_model=100, nhead=5, num_layers=5, dim_ff=400."""
    import sys
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model import MicrobiomeTransformer  # the Susagi repo's model
    m = MicrobiomeTransformer(input_dim_type1=384, input_dim_type2=1536, d_model=d_model, nhead=nhead,
                              num_layers=num_layers, dim_feedforward=dim_ff, dropout=0.0).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    reps = []
    for i in range(0, len(tokens), bs):
        emb = tokens[i:i + bs, :, :384].to(device)               # RAW ProkBERT [b,n_max,384]
        mk = masks[i:i + bs].to(device)                          # [b,n_max] True=present
        x = m.input_projection_type1(emb)                        # [b,n_max,d_model]
        x = m.transformer(x, src_key_padding_mask=~mk)           # [b,n_max,d_model]
        denom = mk.sum(1, keepdim=True).clamp_min(1).float()
        rep = (x * mk.unsqueeze(-1).float()).sum(1) / denom      # masked mean -> [b,d_model]
        reps.append(rep.float().cpu().numpy())
    return np.concatenate(reps, 0)


def probe(X, y, seed=42):
    """5-fold LogReg; returns acc + balanced acc + chance (majority)."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, baccs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))
    _, cnts = np.unique(y, return_counts=True)
    chance = float(cnts.max() / cnts.sum())
    return {"acc_mean": float(np.mean(accs)), "acc_se": float(np.std(accs, ddof=1) / np.sqrt(5)),
            "balanced_acc_mean": float(np.mean(baccs)),
            "balanced_acc_se": float(np.std(baccs, ddof=1) / np.sqrt(5)), "chance": chance,
            "n": int(len(y)), "n_classes": int(len(cnts))}


def run(
    checkpoint: str = None,
    fname: str = "examples/microbiome_jepa/cfgs/layerA_real.yaml",
    data_dir: str = None,
    d_model: int = 128,
    n_max: int = 256,
    per_class_cap: int = 2500,
    terms_max_lines: int = None,
    susagi_repo: str = None,    # path to the Microbiome-Modelling repo (for the imposter-rep baseline)
    susagi_ckpt: str = None,    # path to the Susagi imposter checkpoint (model_state_dict)
    device: str = "cpu",
    out: str = "checkpoints/microbiome_jepa/tech_invariance",
):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    if data_dir is None:
        data_dir = os.path.join(os.environ.get("EBJEPA_DSETS", "."), "susagi", "data")
    mb = os.path.join(data_dir, "microbeatlas")
    terms_path = os.path.join(mb, "sample_terms_mapping_combined_dany_og_biome_tech.txt")
    mapped = os.path.join(mb, "samples-otus.97.mapped")
    h5 = os.path.join(data_dir, "model", "prokbert_embeddings.h5")
    rename = os.path.join(mb, "otus.rename.map1")
    cfg = load_config(fname, {"model.d_model": d_model}, quiet=True)

    runid_labels = load_runid_labels(terms_path, max_lines=terms_max_lines)
    samples = stream_labeled_communities(mapped, runid_labels, n_max, per_class_cap)
    if len(samples) < 50:
        raise RuntimeError(f"only {len(samples)} labelled communities; need more (check Terms join).")

    emb, otu_id_to_row = load_prokbert_embeddings(h5)
    rename_map = load_otu_rename_map(rename)
    all_ids = [oid for s in samples for (oid, _) in s["otus"]]
    resolver = build_otu_key_resolver(all_ids, rename_map, set(otu_id_to_row))
    tokens, masks = build_tokens(samples, emb, otu_id_to_row, resolver, n_max)
    keep = masks.any(1).numpy()
    tokens, masks = tokens[keep], masks[keep]
    samples = [s for s, k in zip(samples, keep) if k]
    strat = [s["strat"] for s in samples]
    biome = [s["biome"] for s in samples]
    zscore = PerDimZScore().fit(tokens.reshape(-1, F), mask=masks.reshape(-1))

    # build the three reps
    reps = {}
    if checkpoint and os.path.exists(checkpoint):
        enc = SetTransformerEncoder(token_dim=F, d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
                                    n_layers=cfg.model.n_layers, dim_feedforward=cfg.model.dim_feedforward,
                                    dropout=0.0, pool=cfg.model.get("pool", "mean")).to(dev)
        ck = torch.load(checkpoint, map_location=dev, weights_only=False)
        sd = ck.get("encoder_state_dict") or ck.get("model_state_dict") or ck
        sd = {k.replace("encoder.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
        enc.load_state_dict(sd, strict=False)
        reps["jepa_pretrained"] = encode(enc, tokens, masks, zscore, dev)
    else:
        logger.warning("no checkpoint -> skipping the pretrained-JEPA rep (only baselines)")
    # random-init encoder baseline (same architecture)
    torch.manual_seed(0)
    enc_rand = SetTransformerEncoder(token_dim=F, d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
                                     n_layers=cfg.model.n_layers, dim_feedforward=cfg.model.dim_feedforward,
                                     dropout=0.0, pool=cfg.model.get("pool", "mean")).to(dev)
    reps["random_encoder"] = encode(enc_rand, tokens, masks, zscore, dev)
    reps["raw_meanpool"] = raw_meanpool(tokens, masks, zscore)
    # the SUSAGI imposter rep (the user's named baseline), if its repo + checkpoint are provided
    if susagi_ckpt and os.path.exists(susagi_ckpt) and susagi_repo and os.path.isdir(susagi_repo):
        try:
            reps["susagi_imposter"] = susagi_imposter_rep(tokens, masks, susagi_repo, susagi_ckpt, dev)
        except Exception as e:
            logger.warning(f"susagi imposter rep failed ({type(e).__name__}: {e}); skipping it")

    # biome control uses only samples with a biome label
    has_biome = np.array([b is not None for b in biome])
    biome_y = [b for b in biome if b is not None]

    results = {"n_samples": len(samples), "per_class_cap": per_class_cap,
               "strategy_counts": {s: int(strat.count(s)) for s in STRATEGIES},
               "n_with_biome": int(has_biome.sum()),
               "biome_counts": {b: int(biome_y.count(b)) for b in sorted(set(biome_y))},
               "tech_probe": {}, "biome_probe": {}}
    print("\n================ SEQUENCING-TECH INVARIANCE (amplicon vs wgs) ================")
    print(f"n={len(samples)} strat={results['strategy_counts']} | TECH acc: LOWER = more invariant = better")
    for name, X in reps.items():
        tp = probe(X, strat)
        results["tech_probe"][name] = tp
        print(f"  TECH  {name:18s} acc {tp['acc_mean']:.3f} ± {tp['acc_se']:.3f}  "
              f"(bal {tp['balanced_acc_mean']:.3f}, chance {tp['chance']:.3f})")
    print(f"\nBIOME control (n={int(has_biome.sum())}, {len(set(biome_y))} biomes): HIGHER = keeps biology")
    for name, X in reps.items():
        if has_biome.sum() < 50:
            break
        bp = probe(X[has_biome], biome_y)
        results["biome_probe"][name] = bp
        print(f"  BIOME {name:18s} acc {bp['acc_mean']:.3f} ± {bp['acc_se']:.3f}  "
              f"(bal {bp['balanced_acc_mean']:.3f}, chance {bp['chance']:.3f})")

    Path(out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out, "tech_invariance.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {out}/tech_invariance.json")
    if "jepa_pretrained" in reps:
        jt = results["tech_probe"]["jepa_pretrained"]["acc_mean"]
        rt = results["tech_probe"]["raw_meanpool"]["acc_mean"]
        print(f"VERDICT: JEPA tech-acc {jt:.3f} vs raw {rt:.3f} -> "
              f"{'MORE' if jt < rt else 'NOT more'} tech-invariant than the raw input")
    return results


if __name__ == "__main__":
    fire.Fire(run)
