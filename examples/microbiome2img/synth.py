"""
synth.py -- synthetic FCGR community trajectories for the FCGR-JEPA integration.

We don't (yet) have raw OTU sequences aligned to the real ProkBERT cache, so to
exercise the *exact* eb_jepa world-model pipeline with the FCGR (DNA-as-image)
encoder we build a small, controlled synthetic cohort:

  * a panel of K OTU sequences organized in CLADES (a proxy phylogeny);
  * each OTU rendered ONCE as an FCGR image (its DNA picture);
  * subjects = action-conditioned abundance trajectories: a discrete "diet" action
    boosts one clade per step, so the next community is genuinely predictable from
    (current community, action) -> the world model has real skill to gain and the
    no-change ("identity") baseline is beatable.

It emits the SAME dict contract as the real MicrobiomeDataset, only with FCGR-image
channels instead of ProkBERT-embedding channels, so it slots into the library JEPA
(`eb_jepa.jepa.JEPA` + `FCGRSetEncoder`) completely unchanged.

__getitem__ returns:
  observations : [S*S + 1, T, N, 1]  (channels = flattened FCGR image + log-abundance)
  actions      : [A, T]              (per-step diet one-hot + dt)
  diversity    : [T]                 (Shannon alpha-diversity target)
  phylo        : [T, S*S]            (abundance-weighted mean FCGR image, tree-free)
  age          : scalar              (continuous phenotype, regression probe target)
  label        : scalar              (binary phenotype, classification probe target)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from examples.microbiome2img.fcgr import fcgr_batch

_BASES = np.array(list("ACGT"))


def _random_seq(n, rng):
    return "".join(rng.choice(_BASES, size=n))


def _mutate(seq, rate, rng):
    s = np.array(list(seq))
    m = rng.random(len(s)) < rate
    s[m] = rng.choice(_BASES, size=int(m.sum()))
    return "".join(s)


def _make_clades(n_clades, per_clade, length, divergence, between, rng):
    """Common-root phylogeny: clade ancestors diverge from one root by `between`,
    members diverge from their ancestor by `divergence`. Returns (seqs, clade_ids)."""
    root = _random_seq(length, rng)
    seqs, clade = [], []
    for c in range(n_clades):
        ancestor = _mutate(root, between, rng)
        for _ in range(per_clade):
            seqs.append(_mutate(ancestor, divergence, rng))
            clade.append(c)
    return seqs, np.asarray(clade, dtype=np.int64)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)


@dataclass
class SynthFCGRConfig:
    k: int = 4                  # FCGR resolution -> S = 2**k image, img_ch = S*S channels
    n_clades: int = 6           # number of clades (also number of diet actions)
    otus_per_clade: int = 8     # K = n_clades * otus_per_clade total OTUs
    seq_len: int = 400
    between: float = 0.28       # inter-clade divergence (clade separation)
    divergence: float = 0.06    # within-clade spread
    n_subjects: int = 240
    n_window: int = 6           # T timepoints per trajectory
    n_max: int = 16             # OTU slots kept per community (top-abundance)
    decay: float = 0.85         # abundance log-state persistence
    diet_strength: float = 2.5  # how hard a diet boosts its clade
    noise: float = 0.30         # process noise on the log-abundance state
    abundance_scale: float = 1.0e4
    val_fraction: float = 0.2
    split: str = "train"        # "train" | "val"
    seed: int = 0


class SynthFCGRDataset(Dataset):
    def __init__(self, config: SynthFCGRConfig):
        self.cfg = config
        K = config.n_clades * config.otus_per_clade
        self.K = K
        self.S = 1 << config.k
        self.img_ch = self.S * self.S
        self.action_dim = config.n_clades + 1  # one diet per clade + dt feature

        # --- panel: K OTU sequences -> FCGR images (deterministic from seed) ---
        rng = np.random.default_rng(config.seed)
        seqs, clade = _make_clades(config.n_clades, config.otus_per_clade,
                                   config.seq_len, config.divergence, config.between, rng)
        imgs = fcgr_batch(seqs, k=config.k)  # [K, S, S] probabilities
        self.panel = imgs.reshape(K, -1).astype(np.float32)  # [K, img_ch]
        self.clade = clade

        # diet d boosts clade d in log-abundance space: effect[d, otu] = strength
        self.effect = np.zeros((config.n_clades, K), dtype=np.float32)
        for d in range(config.n_clades):
            self.effect[d, clade == d] = config.diet_strength

        # --- subjects: action-conditioned abundance trajectories ---
        T = config.n_window
        abund = np.zeros((config.n_subjects, T, K), dtype=np.float32)
        diets = np.zeros((config.n_subjects, T), dtype=np.int64)
        pheno = np.zeros(config.n_subjects, dtype=np.float32)
        for s in range(config.n_subjects):
            srng = np.random.default_rng(config.seed * 100003 + s)
            ell = srng.normal(0.0, 1.0, K).astype(np.float32)
            ds = srng.integers(0, config.n_clades, size=T)
            for t in range(T):
                p = _softmax(ell)
                abund[s, t] = p
                d = ds[t]
                ell = config.decay * ell + self.effect[d] + srng.normal(0.0, config.noise, K)
            diets[s] = ds
            # phenotype = mean fraction of clade-0 taxa across the trajectory
            # (a function of the communities, so it is decodable from good latents)
            pheno[s] = abund[s, :, clade == 0].mean()

        self._abund_all = abund
        self._diets_all = diets
        median = float(np.median(pheno))
        labels_all = (pheno > median).astype(np.float32)

        # subject-disjoint split (deterministic; both splits share panel + effects)
        idx = np.arange(config.n_subjects)
        srng = np.random.default_rng(config.seed)
        srng.shuffle(idx)
        n_val = max(1, int(round(config.n_subjects * config.val_fraction)))
        val = set(idx[:n_val].tolist())
        keep = [i for i in range(config.n_subjects)
                if (i in val) == (config.split == "val")]
        self.subjects = keep
        # normalize the regression target on this split's subjects
        self._pheno = pheno
        self._labels = labels_all

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, i):
        cfg = self.cfg
        s = self.subjects[i]
        T, N, K = cfg.n_window, cfg.n_max, self.K
        abund = self._abund_all[s]   # [T, K]
        diets = self._diets_all[s]   # [T]

        obs = torch.zeros(self.img_ch + 1, T, N, 1, dtype=torch.float32)
        div = torch.zeros(T, dtype=torch.float32)
        phylo = torch.zeros(T, self.img_ch, dtype=torch.float32)
        act = torch.zeros(self.action_dim, T, dtype=torch.float32)

        for t in range(T):
            p = abund[t]
            # keep the top-N most abundant OTUs (deterministic dominant taxa)
            top = np.argsort(-p)[:N]
            n = len(top)
            obs[: self.img_ch, t, :n, 0] = torch.from_numpy(self.panel[top].T)  # [img_ch, n]
            logab = np.log1p(cfg.abundance_scale * p[top]).astype(np.float32)
            obs[self.img_ch, t, :n, 0] = torch.from_numpy(logab)
            div[t] = float(-(p * np.log(p + 1e-12)).sum())  # Shannon over full community
            phylo[t] = torch.from_numpy((p[:, None] * self.panel).sum(axis=0))  # weighted mean FCGR
            act[diets[t], t] = 1.0
            act[-1, t] = 1.0 / T  # dt feature (regular sampling)

        return {
            "observations": obs,
            "actions": act,
            "diversity": div,
            "phylo": phylo,
            "age": torch.tensor(float(self._pheno[s]), dtype=torch.float32),
            "label": torch.tensor(float(self._labels[s]), dtype=torch.float32),
        }


def make_loaders(cfg: SynthFCGRConfig, batch_size=64, num_workers=0):
    """Return (train_ds, val_ds, train_loader, val_loader) sharing the same panel."""
    train_ds = SynthFCGRDataset(SynthFCGRConfig(**{**cfg.__dict__, "split": "train"}))
    val_ds = SynthFCGRDataset(SynthFCGRConfig(**{**cfg.__dict__, "split": "val"}))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    return train_ds, val_ds, train_loader, val_loader
