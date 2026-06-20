"""
compare.py -- the controlled bench for "DNA-as-image vs DNA-as-language".

Goal: measure whether imaging the DNA (FCGR) preserves the OTU structure a
downstream probe needs, on the SAME task, so we can later put it head-to-head with
the ProkBERT (DNA language model) embedding.

Representations compared (all -> standardized linear probe, train/test split):
  * kmer       FCGR flattened = raw k-mer spectrum            (imaging, NO conv)
  * fcgr_cnn   FCGREncoder (RANDOM, untrained) on FCGR image  (imaging + conv prior)
  * prokbert   ProkBERT embedding, IF a matching cache is given (--prokbert)

Data:
  * default  : synthetic CLADES (each clade = an ancestor + mutated descendants),
               a controlled phylogeny -> the probe predicts the clade.
  * --fasta + --labels : real OTU sequences + per-sequence integer labels.

  python -m examples.microbiome2img.compare                      # synthetic proof
  python -m examples.microbiome2img.compare --clades 12 --per 50
  python -m examples.microbiome2img.compare --fasta otus.fa --labels y.npy
"""

import argparse

import numpy as np
import torch

from examples.microbiome2img.encoder import FCGREncoder
from examples.microbiome2img.fcgr import fcgr_batch

_BASES = np.array(list("ACGT"))


def _random_seq(n, rng):
    return "".join(rng.choice(_BASES, size=n))


def _mutate(seq, rate, rng):
    s = np.array(list(seq))
    m = rng.random(len(s)) < rate
    s[m] = rng.choice(_BASES, size=int(m.sum()))
    return "".join(s)


def _make_clades(n_clades, per_clade, length, divergence, rng, between=0.25):
    """Synthetic phylogeny with a COMMON ROOT: clade ancestors diverge from one
    root by `between` (clade separation), then members diverge by `divergence`
    (within-clade spread). Smaller `between` -> closer clades -> harder probe."""
    root = _random_seq(length, rng)
    seqs, labels = [], []
    for c in range(n_clades):
        ancestor = _mutate(root, between, rng)
        for _ in range(per_clade):
            seqs.append(_mutate(ancestor, divergence, rng))
            labels.append(c)
    return seqs, np.array(labels)


def _read_fasta(path):
    ids, seqs, cur, cid = [], [], [], None
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    ids.append(cid); seqs.append("".join(cur)); cur = []
                cid = line[1:].strip().split()[0]
            else:
                cur.append(line.strip())
    if cur:
        ids.append(cid); seqs.append("".join(cur))
    return ids, seqs


_RANK_COL = {"domain": 1, "phylum": 2, "class": 3, "order": 4, "family": 5, "genus": 6}


def _load_taxonomy(path, rank):
    """otu id -> label string at the chosen rank (skips empty labels)."""
    col = _RANK_COL[rank]
    out = {}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > col and p[col]:
                out[p[0]] = p[col]
    return out


def _build_real(fasta, taxonomy, rank, max_otus, min_per_class, seed):
    """Align FASTA ids with taxonomy at `rank`, drop rare classes, subsample.
    Returns (seqs, y int array, class names)."""
    from collections import Counter
    ids, seqs = _read_fasta(fasta)
    id2seq = dict(zip(ids, seqs))
    lab = _load_taxonomy(taxonomy, rank)
    pairs = [(i, lab[i]) for i in ids if i in lab]
    cnt = Counter(l for _, l in pairs)
    pairs = [(i, l) for i, l in pairs if cnt[l] >= min_per_class]
    rng = np.random.default_rng(seed)
    if max_otus and len(pairs) > max_otus:
        idx = rng.choice(len(pairs), size=max_otus, replace=False)
        pairs = [pairs[j] for j in sorted(idx)]
    # stratified split needs >=2 members per class -> drop singletons post-subsample
    cnt2 = Counter(l for _, l in pairs)
    pairs = [(i, l) for i, l in pairs if cnt2[l] >= 2]
    classes = sorted({l for _, l in pairs})
    cls2int = {c: k for k, c in enumerate(classes)}
    seqs_out = [id2seq[i] for i, _ in pairs]
    y = np.array([cls2int[l] for _, l in pairs])
    return seqs_out, y, classes


def _probe(X, y, seed):
    """Standardized linear probe, stratified split -> (accuracy, macro-F1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=seed)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="FCGR resolution (2^k)")
    ap.add_argument("--clades", type=int, default=10)
    ap.add_argument("--per", type=int, default=40, help="sequences per clade")
    ap.add_argument("--len", type=int, default=500)
    ap.add_argument("--divergence", type=float, default=0.10, help="within-clade spread")
    ap.add_argument("--between", type=float, default=0.20,
                    help="inter-clade divergence from the common root (smaller = harder)")
    ap.add_argument("--dim", type=int, default=128, help="FCGR-CNN embedding dim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--labels", default=None, help=".npy of int labels for --fasta")
    ap.add_argument("--taxonomy", default=None,
                    help="otus_97_taxonomy.tsv -> labels at --rank (real-data mode)")
    ap.add_argument("--rank", default="family",
                    choices=["domain", "phylum", "class", "order", "family", "genus"])
    ap.add_argument("--max-otus", type=int, default=4000, help="subsample for the probe")
    ap.add_argument("--min-per-class", type=int, default=20)
    ap.add_argument("--prokbert", default=None,
                    help=".npy [N, P] ProkBERT embeddings aligned to the sequences")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.fasta and args.taxonomy:
        seqs, y, classes = _build_real(args.fasta, args.taxonomy, args.rank,
                                       args.max_otus, args.min_per_class, args.seed)
        chance = 1.0 / len(classes)
        src = f"real OTUs ({args.rank}: {len(classes)} classes, {len(seqs)} OTUs)"
    elif args.fasta:
        _, seqs = _read_fasta(args.fasta)
        y = np.load(args.labels).astype(int)
        assert len(seqs) == len(y), "fasta and labels length mismatch"
        chance = 1.0 / len(np.unique(y))
        src = f"real OTUs ({args.fasta})"
    else:
        seqs, y = _make_clades(args.clades, args.per, args.len, args.divergence, rng,
                               between=args.between)
        chance = 1.0 / args.clades
        src = f"synthetic clades (n={args.clades}, between={args.between}, within={args.divergence})"

    # FCGR images for every sequence -> [N, S, S]
    imgs = fcgr_batch(seqs, k=args.k)             # probabilities
    N, S = imgs.shape[0], imgs.shape[-1]

    rows = {}
    # 1) k-mer spectrum: the FCGR image flattened (imaging, no conv)
    rows["kmer (FCGR flat)"] = _probe(imgs.reshape(N, -1), y, args.seed)

    # 2) FCGR + CNN random features (imaging + conv inductive bias)
    enc = FCGREncoder(k=args.k, out_dim=args.dim).eval()
    with torch.no_grad():
        Xc = enc(torch.from_numpy(imgs).float()).numpy()
    rows["fcgr_cnn (random)"] = _probe(Xc, y, args.seed)

    # 3) ProkBERT (the other approach), only if a matching cache is provided
    if args.prokbert:
        Xp = np.load(args.prokbert)
        assert len(Xp) == N, "prokbert cache length mismatch"
        rows["prokbert (LM)"] = _probe(Xp, y, args.seed)

    print(f"\n== DNA-as-image vs DNA-as-language -- {src} ==")
    print(f"(linear probe, {N} sequences, {S}x{S} FCGR; chance = {chance:.3f})")
    print(f"{'representation':<22}{'accuracy':>10}{'macro-F1':>10}")
    print("-" * 42)
    for name, (acc, f1) in rows.items():
        print(f"{name:<22}{acc:>10.3f}{f1:>10.3f}")
    if not args.prokbert:
        print("\n(no --prokbert cache: add it to put the LM embedding head-to-head.)")
    print("Read: higher = the representation separates OTU clades better.")


if __name__ == "__main__":
    main()
