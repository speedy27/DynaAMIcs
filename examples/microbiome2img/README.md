# microbiome2img — reading bacterial DNA as **images** (FCGR + image-JEPA)

A research probe: instead of encoding an OTU's SSU-rRNA sequence with a DNA
language model (**ProkBERT**, our current encoder), render each sequence as a
**Frequency Chaos Game Representation (FCGR) image** and let an image encoder read
it. This reuses the mature `image_jepa` path (`ImpalaEncoder`/`ResNet`, I-JEPA
masking) and turns the microbiome into a genuine *imaged* modality.

## Why microbiome (and not Tahoe)
FCGR images a **sequence of letters** (A/C/G/T). An OTU *is* a DNA sequence → FCGR
applies directly. A Tahoe cell is an **expression vector**, not a sequence — there
is nothing sequence-like to image, so FCGR does not transfer. Microbiome is the
natural fit.

## The idea in one picture
```
 "ACGTTAC..."  ──CGR──▶  trajectory in [0,1]^2  ──bin 2^k──▶  [2^k, 2^k] image
   OTU DNA               (each step: halfway to        pixel = frequency of a
                          the base's corner)            specific k-mer
```
After reading `k` bases the point lands in a unique cell of a `2^k × 2^k` grid that
maps one-to-one to the trailing k-mer, so the image is exactly the OTU's k-mer
spectrum. **k=6 → 64×64**, matching the conv encoders.

## Hypothesis (the judged claim)
> Imaging the DNA (FCGR + CNN) captures OTU/phylogenetic structure **as well as**
> a DNA language model (ProkBERT) — a clean *image-CGR vs embedding* ablation on
> the exact same downstream probes (age clock, T1D).

## What's here
| File | Role |
|---|---|
| `fcgr.py` | the transform: `fcgr(seq, k)` → `[2^k, 2^k]` image (+ `fcgr_batch`) |
| `demo.py` | proof that **close sequences → close images** (synthetic, no data needed) |
| `encoder.py` | `FCGREncoder` (CNN image→embedding) + `FCGRSetEncoder` (the community swap) |
| `compare.py` | the **2-approach bench**: k-mer vs FCGR-CNN (+ ProkBERT slot), linear probe |
| `main.py` + `synth.py` | the **library JEPA** with `FCGRSetEncoder` on a synthetic FCGR cohort (the eb_jepa integration) |
| `cfgs/train.yaml` | config for `main.py` (FCGR / synthetic + JEPA + loss coeffs) |

## Run
```bash
# 1) proof of concept (synthetic) -> distance table + fcgr_demo.png
python -m examples.microbiome2img.demo

# 2) the 2-approach bench on synthetic clades (proxy phylogeny, harder regime)
python -m examples.microbiome2img.compare --clades 15 --between 0.06 --divergence 0.05

# 3) on real OTUs once you have them (sequences + int labels [+ aligned ProkBERT cache])
python -m examples.microbiome2img.compare --fasta otus.fa --labels y.npy --prokbert prokbert.npy
```

## Status & next steps
- ✅ Transform + similarity proof (`demo.py`).
- ✅ **2-approach bench** (`compare.py`): k-mer spectrum vs FCGR-CNN on a controlled
  phylogeny, with a ready slot for the ProkBERT embedding. Runs today on synthetic
  clades; same command on real data via `--fasta/--labels/--prokbert`.
- ⚠️ **Data needed for the *real* head-to-head**: our cache holds only the ProkBERT
  **embeddings**, not the raw ACGT sequences. Fetch the ~16.8k SSU-rRNA representative
  sequences (MicrobeAtlas / the Susagi repo) as a FASTA + export the aligned ProkBERT
  vectors as a `.npy`, then run command (3) above.
- ✅ **Full JEPA integration** (`main.py` + `synth.py`): `FCGRSetEncoder` (now in
  `eb_jepa/architectures.py`, a 5D-contract drop-in for `SetEncoder`) plugged into the
  **library** `eb_jepa.jepa.JEPA` with the SAME `RNNPredictor` + `VC_IDM_Sim_Regularizer`
  + `SquareLossSeq` + bio losses as `examples/microbiome`. Trains end-to-end on a
  synthetic FCGR cohort (`synth.py`), since the cache has no raw sequences.
  Smoke: `python -m examples.microbiome2img.main optim.epochs=2`.
- ⏭️ **Real head-to-head** (image-CGR vs ProkBERT): point `synth.py`'s panel at a real
  OTU FASTA, then compare `skill` / `age_r2` / probe against `examples/microbiome`.
