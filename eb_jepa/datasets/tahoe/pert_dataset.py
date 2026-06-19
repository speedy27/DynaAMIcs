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
    def __init__(self, cfg: PertConfig, stats=None):
        self.cfg = cfg
        b = torch.load(cfg.cache_path, weights_only=False)
        X = b["X"].float()
        if stats is None:
            self.mu, self.sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
        else:
            self.mu, self.sd = stats
        self.X = (X - self.mu) / self.sd
        self.drug = b["drug"]; self.cl = b["cell_line"]; self.is_ctrl = b["is_control"]
        self.fp = b["drug_fp"].float()                  # [n_drugs, A]
        self.centroid = ((b["centroid"] - self.mu) / self.sd).float()  # [n_lines, D]
        self.drug_names = b["drug_names"]; self.cl_names = b["cl_names"]
        self.modules = b["modules"]; self.n_modules = int(b["n_modules"])
        self.action_dim = self.fp.shape[1]
        self.D = self.X.shape[1]

        onehot = torch.zeros(self.D, self.n_modules)
        onehot[torch.arange(self.D), self.modules] = 1.0
        self._mod = onehot / onehot.sum(0).clamp_min(1.0)  # [D, M]

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
        pathway = z_pert @ self._mod                                            # [M]
        return {"observations": obs, "actions": a, "drug": d, "cell_line": line, "pathway": pathway}


def make_loaders(cfg: PertConfig, batch_size=512, num_workers=0):
    tr = PertDataset(PertConfig(**{**cfg.__dict__, "split": "train"}))
    va = PertDataset(PertConfig(**{**cfg.__dict__, "split": "val"}), stats=tr.stats())
    return (tr, va,
            DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers))
