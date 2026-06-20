"""
fetch_otu_sequences.py -- parse MicrobeAtlas `otus.97.allinfo` into (1) a FASTA of
OTU representative DNA sequences and (2) a taxonomy label table, so microbiome2img
can run FCGR on REAL OTUs with a real taxonomic probe.

Source (~273 MB, the cluster has internet):
  https://microbeatlas.org/downloads/otus/otus.97.allinfo

Row format (tab-separated):
  MAPv3;90_..;96_..;97_<id> <tab> n <tab> ... <tab> <DNA seq> <tab> <accession>
  <tab> d__..;p__..;c__..;o__..;f__..;g__..      (GTDB-style taxonomy)

Usage:
  # 1) download on the cluster
  curl -L -o otus.97.allinfo https://microbeatlas.org/downloads/otus/otus.97.allinfo
  # 2) parse (stdlib only, no deps)
  python3 fetch_otu_sequences.py --allinfo otus.97.allinfo --out-dir .

Outputs:
  otus_97.fasta          >97_<id> / <seq>
  otus_97_taxonomy.tsv   97_<id> \t domain..genus  (label source for the probe)
"""

import argparse
import os

_VALID = set("ACGTN")
_RANKS = ["d__", "p__", "c__", "o__", "f__", "g__"]


def _id97(otu_field):
    for p in otu_field.split(";"):
        if p.startswith("97_"):
            return p
    return otu_field.split(";")[-1]


def _find_seq(parts, min_len):
    """The representative sequence = the longest mostly-ACGT field."""
    best = ""
    for p in parts:
        u = p.strip().upper()
        if len(u) > len(best) and len(u) >= min_len and set(u) <= _VALID:
            best = u
    return best


def _taxonomy(parts):
    for p in parts:
        if p.startswith("d__"):
            return p
    return ""


def _split_tax(tax):
    d = {r: "" for r in _RANKS}
    for tok in tax.split(";"):
        tok = tok.strip()
        for r in _RANKS:
            if tok.startswith(r):
                d[r] = tok[len(r):]
    return [d[r] for r in _RANKS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allinfo", required=True, help="path to otus.97.allinfo")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--max-n-frac", type=float, default=0.02, help="max fraction of Ns")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fa = os.path.join(args.out_dir, "otus_97.fasta")
    tx = os.path.join(args.out_dir, "otus_97_taxonomy.tsv")

    n_in = n_ok = 0
    phyla = {}
    with open(args.allinfo, errors="replace") as f, \
            open(fa, "w") as out_fa, open(tx, "w") as out_tx:
        out_tx.write("otu\tdomain\tphylum\tclass\torder\tfamily\tgenus\n")
        for line in f:
            n_in += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            seq = _find_seq(parts, args.min_len)
            if len(seq) < args.min_len:
                continue
            if seq.count("N") / len(seq) > args.max_n_frac:
                continue
            oid = _id97(parts[0])
            ranks = _split_tax(_taxonomy(parts))
            out_fa.write(f">{oid}\n{seq}\n")
            out_tx.write(oid + "\t" + "\t".join(ranks) + "\n")
            phyla[ranks[1]] = phyla.get(ranks[1], 0) + 1
            n_ok += 1

    print(f"parsed {n_in} rows -> {n_ok} OTUs with usable sequences")
    print(f"FASTA     -> {fa}")
    print(f"taxonomy  -> {tx}")
    top = sorted(phyla.items(), key=lambda kv: -kv[1])[:8]
    print("top phyla:", ", ".join(f"{p or '?'}={c}" for p, c in top))


if __name__ == "__main__":
    main()
