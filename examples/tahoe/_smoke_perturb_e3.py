"""
Smoke test for the 2-step E3 regime (no download): the perturbation world-model
running on top of the FROZEN grounded SetTransformer (raw genes -> z), end-to-end.

  1. fabricate a grounded-encoder checkpoint (tahoe_ground.pt shape) over K genes;
  2. fabricate a RAW-GENE perturbation cache (X=[N,K] genes, per-drug fixed shift);
  3. run perturb.run with model.encoder=settransformer -> the SetTransformer encodes
     genes to z, the GRU predictor is trained in z-space, skill stays finite.

  python -m examples.tahoe._smoke_perturb_e3
"""
import os
import tempfile

import numpy as np
import torch

from eb_jepa.architectures import SetTransformer
from examples.tahoe import perturb
from examples.tahoe._smoke_perturb import build_cache


def build_ground_ckpt(path, K, Dz=24, d_model=32, n_latents=6, depth=1, heads=2):
    enc = SetTransformer(n_genes=K, out_d=Dz, d_model=d_model, n_latents=n_latents,
                         depth=depth, heads=heads)
    cfg = {"model": {"d_model": d_model, "n_latents": n_latents, "depth": depth,
                     "heads": heads, "dstc": Dz}}
    torch.save({"target": enc.state_dict(), "online": enc.state_dict(), "cfg": cfg,
                "n_genes": K, "out_d": Dz, "source_dims": {}}, path)


def main():
    tmp = tempfile.mkdtemp()
    K = 32                                            # raw-gene panel (small)
    cache = os.path.join(tmp, "cache_pert_genes.pt")
    build_cache(cache, n_lines=4, n_drugs=5, D=K, per=60)   # D=K -> raw-gene feature space
    gpath = os.path.join(tmp, "tahoe_ground.pt")
    build_ground_ckpt(gpath, K=K)

    os.environ["EBJEPA_CKPTS"] = os.path.join(tmp, "ckpt")
    overrides = [
        f"data.cache_path={cache}", "data.batch_size=128", "data.num_workers=0",
        "optim.epochs=4", "optim.probe_every=2",
        "loss.cos_coeff=1.0", "loss.ot_coeff=0.5", "model.layers=1",
        "model.encoder=settransformer", f"model.ground_ckpt={gpath}",
    ]
    perturb.run("examples/tahoe/cfgs/perturb.yaml", overrides)
    print("\nSMOKE OK (E3: world-model trained on frozen grounded SetTransformer latents)")


if __name__ == "__main__":
    main()
