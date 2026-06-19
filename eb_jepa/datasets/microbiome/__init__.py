"""Microbiome modality for EB-JEPA.

New data sources for the microbiome JEPA world model:
- glv.py        : generalized Lotka-Volterra simulator (synthetic, non-monotonic attractors)
- otu_data.py   : real OTU-set dataset (ProkBERT DNA embeddings + CLR log-abundance)
- transforms.py : CLR + per-dimension z-score normalization, OTU masking/augmentation
- traj.py       : TrajDataset adapters yielding (obs_dict, act, state, reward, extra) slices

Wired into the dispatcher via the `microbiome` branch in eb_jepa/datasets/utils.py.
"""
