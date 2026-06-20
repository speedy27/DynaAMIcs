"""
pert_dataset.py - control -> perturbed transitions for the action-conditioned
cell world model (eb_jepa JEPA.unroll).

Each item is a 2-step "trajectory" in frozen MosaicFM embedding space:
  t=0 : a CONTROL cell state (random DMSO cell of the same line, or the cell-line
        centroid if no DMSO controls exist)
  t=1 : the TREATED cell state
with the action = the drug's Morgan fingerprint.

__getitem__ -> dict:
  observations : [D, 2, 1, 1]   (D=2560 ; t0=control, t1=perturbed)
  actions      : [A, 2]         (drug fingerprint, constant over the 2 steps)
  drug, cell_line : int
  pathway      : [M]            module-activity descriptor of the perturbed cell
"""

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset


@dataclass
class PertConfig:
    cache_path: str = "artifacts/tahoe/cache_pert.pt"
    val_fraction: float = 0.2
    seed: int = 0
    split: str = "train"


class PertDataset(Dataset):
    """Control->perturbed transitions in encoder-latent space.

    `encode_fn` (optional) is a FROZEN encoder f_θ: [N, F0] -> [N, Dz]. With it, the
    cache holds RAW GENES (F0 = K panel genes) and we pre-encode them through the
    grounded SetTransformer (the 2-step "E3" regime). Without it, the cache holds
    pretrained MosaicFM embeddings and f_θ is the identity (the "E1" regime).

    The per-cell pathway descriptor P is computed in the ORIGINAL feature space (where
    `modules` lives) BEFORE encoding, so it stays meaningful in both regimes.
    """
    def __init__(self, cfg: PertConfig, stats=None, encode_fn=None):
        self.cfg = cfg
        b = torch.load(cfg.cache_path, weights_only=False)
        Xg = b["X"].float()                             # original feature space [N, F0]
        self.drug = b["drug"]; self.cl = b["cell_line"]; self.is_ctrl = b["is_control"]
        self.fp = b["drug_fp"].float()                  # [n_drugs, A]
        self.drug_names = b["drug_names"]; self.cl_names = b["cl_names"]
        self.modules = b["modules"]; self.n_modules = int(b["n_modules"])

        # pathway descriptor in ORIGINAL space (modules are over the F0 features)
        F0 = Xg.shape[1]
        onehot = torch.zeros(F0, self.n_modules)
        onehot[torch.arange(F0), self.modules] = 1.0
        mod = onehot / onehot.sum(0).clamp_min(1.0)     # [F0, M]
        mu0, sd0 = Xg.mean(0, keepdim=True), Xg.std(0, keepdim=True) + 1e-6
        self.P = ((Xg - mu0) / sd0) @ mod               # [N, M] per-cell program activity

        # state space: encode genes -> z with the frozen encoder, else keep embeddings
        centroid_g = b["centroid"].float()              # [n_lines, F0]
        if encode_fn is not None:
            X = encode_fn(Xg); centroid = encode_fn(centroid_g)   # [*, Dz]
        else:
            X, centroid = Xg, centroid_g
        if stats is None:
            self.mu, self.sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
        else:
            self.mu, self.sd = stats
        self.X = (X - self.mu) / self.sd
        self.centroid = ((centroid - self.mu) / self.sd).float()  # [n_lines, D]
        self.action_dim = self.fp.shape[1]
        self.D = self.X.shape[1]

        # treated cells = items; controls per line for pairing
        treated = (~self.is_ctrl).nonzero(as_tuple=True)[0].numpy()
        rng = np.random.default_rng(cfg.seed); rng.shuffle(treated)
        nval = int(round(len(treated) * cfg.val_fraction))
        self.ids = treated[nval:] if cfg.split == "train" else treated[:nval]
        self.has_ctrl = bool(self.is_ctrl.any())
        if self.has_ctrl:
            self.ctrl_by_line = {}
            cidx = self.is_ctrl.nonzero(as_tuple=True)[0]
            for i in cidx.tolist():
                self.ctrl_by_line.setdefault(int(self.cl[i]), []).append(i)
        self._rng = np.random.default_rng(cfg.seed + 1)

    def stats(self):
        return (self.mu, self.sd)

    def __len__(self):
        return len(self.ids)

    def _control_state(self, line):
        if self.has_ctrl and self.ctrl_by_line.get(line):
            j = self.ctrl_by_line[line][self._rng.integers(len(self.ctrl_by_line[line]))]
            return self.X[j]
        return self.centroid[line]  # pseudo-control fallback

    def __getitem__(self, i):
        j = int(self.ids[i]); line = int(self.cl[j]); d = int(self.drug[j])
        z_ctrl = self._control_state(line); z_pert = self.X[j]
        obs = torch.stack([z_ctrl, z_pert], dim=1).unsqueeze(-1).unsqueeze(-1)  # [D,2,1,1]
        a = self.fp[d].unsqueeze(1).repeat(1, 2)                                # [A,2]
        return {"observations": obs, "actions": a, "drug": d, "cell_line": line,
                "pathway": self.P[j]}                                          # [M]


def make_loaders(cfg: PertConfig, batch_size=512, num_workers=0, encode_fn=None):
    tr = PertDataset(PertConfig(**{**cfg.__dict__, "split": "train"}), encode_fn=encode_fn)
    va = PertDataset(PertConfig(**{**cfg.__dict__, "split": "val"}), stats=tr.stats(),
                     encode_fn=encode_fn)
    return (tr, va,
            DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers))
