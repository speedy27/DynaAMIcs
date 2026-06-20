"""
demo.py -- proof that imaging DNA preserves similarity.

The whole bet of microbiome2img is: "phylogenetically close OTUs produce close
images, so an image encoder can read community structure from DNA pictures." This
script shows that property holds, with NO real data needed (uses controlled
synthetic sequences). When you have the real SSU-rRNA fastas, pass --fasta to run
the exact same check on real OTUs.

  python -m examples.microbiome2img.demo                 # synthetic proof + figure
  python -m examples.microbiome2img.demo --fasta otus.fa # on real OTU sequences

Output: examples/microbiome2img/fcgr_demo.png  + a printed distance table showing
mutated-but-related sequences stay closer in image space than a random outgroup.
"""

import argparse
import os

import numpy as np

from examples.microbiome2img.fcgr import fcgr

_BASES = np.array(list("ACGT"))


def _random_seq(n, rng):
    return "".join(rng.choice(_BASES, size=n))


def _mutate(seq, rate, rng):
    s = np.array(list(seq))
    mask = rng.random(len(s)) < rate
    s[mask] = rng.choice(_BASES, size=int(mask.sum()))
    return "".join(s)


def _read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="image is 2^k x 2^k (6 -> 64x64)")
    ap.add_argument("--len", type=int, default=1000, help="synthetic sequence length")
    ap.add_argument("--fasta", default=None, help="optional: real OTU sequences")
    ap.add_argument("--out", default="examples/microbiome2img/fcgr_demo.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.fasta:
        seqs = _read_fasta(args.fasta)[:4]
        names = [f"otu{i}" for i in range(len(seqs))]
    else:
        ancestor = _random_seq(args.len, rng)
        seqs = [ancestor, _mutate(ancestor, 0.05, rng),
                _mutate(ancestor, 0.15, rng), _random_seq(args.len, rng)]
        names = ["ancestor", "mutant 5%", "mutant 15%", "random outgroup"]

    imgs = [fcgr(s, k=args.k) for s in seqs]

    # pairwise L2 distance between the (probability) images
    flat = np.stack([im.ravel() for im in imgs])
    D = np.sqrt(((flat[:, None] - flat[None]) ** 2).sum(-1))

    print(f"\n== FCGR similarity (k={args.k} -> {1<<args.k}x{1<<args.k} images) ==")
    print("distance to first sequence (closer = more similar):")
    for nm, d in zip(names, D[0]):
        print(f"  {nm:<18} {d:.4f}")
    if not args.fasta:
        ok = D[0, 1] < D[0, 2] < D[0, 3]
        verdict = "YES -- imaging preserves similarity" if ok else "no"
        print(f"\nmonotone (5% < 15% < random)? {verdict}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))
        for a, im, nm in zip(np.atleast_1d(ax), imgs, names):
            a.imshow(np.log1p(im), cmap="magma")
            a.set_title(nm, fontsize=11)
            a.axis("off")
        fig.suptitle(f"FCGR images of DNA (k={args.k}) — close sequences, close pictures")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"figure -> {args.out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
