"""
Tahoe-JEPA FAST: ultra-optimized variant of main.py.

Same scientific recipe (two-view SSL, SIGReg/VICReg, pathway prior, linear probe),
but engineered to saturate a GB200 / H100:

  * GPU-resident dataset: X, P loaded ONCE on GPU; no DataLoader, no CPU->GPU
    copy per step, no pickling, no worker IPC.
  * On-device augmentation (dropout + multiplicative noise) directly on the
    GPU batch.
  * bf16 autocast for the encoder/projector forward.
  * TF32 matmul + cudnn.benchmark.
  * Large batch (16k by default) + scaled LR.
  * print(..., flush=True) so logs actually appear under `tee`.

Throughput diagnostic is printed every epoch (cells/s).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import Projector, SetTransformer
from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeDataset
from eb_jepa.losses import BCS, PathwayCoherenceLoss, VICRegLoss
from eb_jepa.schedulers import CosineWithWarmup


class CellEncoder(nn.Module):
    def __init__(self, k, h, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, d),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def embed(enc, X_gpu, bs=8192):
    enc.eval()
    out = []
    for i in range(0, len(X_gpu), bs):
        out.append(enc(X_gpu[i:i + bs]).float().cpu())
    return torch.cat(out).numpy()


def probe(name, Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    keep = ytr >= 0
    if keep.sum() < 10 or len(np.unique(ytr[keep])) < 2:
        return None
    clf = LogisticRegression(max_iter=300).fit(Xtr[keep], ytr[keep])
    m = yva >= 0
    pred = clf.predict(Xva[m])
    return dict(task=name, macro_f1=float(f1_score(yva[m], pred, average="macro")),
                acc=float(accuracy_score(yva[m], pred)))


def _build_encoder(cfg, K, D, device):
    kind = cfg.model.get("encoder", "mlp")
    if kind == "settransformer":
        enc = SetTransformer(
            n_genes=K, out_d=D, d_model=cfg.model.get("d_model", 384),
            n_latents=cfg.model.get("n_latents", 32),
            depth=cfg.model.get("depth", 4), heads=cfg.model.get("heads", 6),
        )
    else:
        enc = CellEncoder(K, cfg.model.henc, D)
    return enc.to(device), kind


def run(fname, overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    assert torch.cuda.is_available(), "main_fast requires CUDA"
    device = torch.device("cuda")
    torch.manual_seed(cfg.meta.seed); np.random.seed(cfg.meta.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dcfg = TahoeConfig(cache_path=cfg.data.cache_path, drop_frac=cfg.data.drop_frac,
                       noise_std=cfg.data.noise_std, val_fraction=cfg.data.val_fraction,
                       seed=cfg.meta.seed)
    tr = TahoeDataset(TahoeConfig(**{**dcfg.__dict__, "split": "train"}))
    va = TahoeDataset(TahoeConfig(**{**dcfg.__dict__, "split": "val"}), stats=tr.stats())
    K, D = tr.K, cfg.model.dstc
    BS = int(cfg.data.batch_size)

    # GPU-resident slices
    Xtr_gpu = tr.X[tr.ids].to(device, non_blocking=True)
    Xva_gpu = va.X[va.ids].to(device, non_blocking=True)
    Ptr_gpu = tr.P[tr.ids].to(device, non_blocking=True)
    drop = float(cfg.data.drop_frac)
    noise = float(cfg.data.noise_std)
    N = Xtr_gpu.shape[0]

    enc, enc_kind = _build_encoder(cfg, K, D, device)
    if enc_kind == "settransformer":
        gs = cfg.data.get("gene_sources", "")
        if gs and os.path.exists(gs):
            tables = torch.load(gs, weights_only=False)
            for name, tbl in tables.items():
                enc.register_gene_source(name, tbl)
            print(f"  gene-init sources: {list(tables)}", flush=True)

    proj = Projector(f"{D}-{cfg.model.proj}-{cfg.model.proj}").to(device)
    if cfg.loss.reg == "sigreg":
        reg = BCS(num_slices=cfg.loss.num_slices, lmbd=cfg.loss.lmbd).to(device)
    else:
        reg = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff).to(device)
    pathway = PathwayCoherenceLoss().to(device)
    pw = float(cfg.loss.pathway_coeff)

    n_params = sum(p.numel() for p in list(enc.parameters()) + list(proj.parameters()))
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"== tahoe-jepa FAST | dev={device} | N_tr={N} N_va={len(Xva_gpu)} | "
          f"K={K} D={D} BS={BS} reg={cfg.loss.reg} enc={enc_kind} | "
          f"params={n_params/1e6:.2f}M | gpu_mem(data)={mem_gb:.2f}GB ==", flush=True)

    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    steps_per_epoch = max(1, N // BS)
    sched = CosineWithWarmup(opt, total_steps=steps_per_epoch * cfg.optim.epochs,
                             warmup_ratio=0.1, min_lr=cfg.optim.lr * 0.01)

    use_amp = bool(cfg.optim.get("bf16", True))
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for ep in range(1, cfg.optim.epochs + 1):
        enc.train(); proj.train()
        t0 = time.time()
        perm = torch.randperm(N, device=device)
        agg = {"reg": 0.0, "path": 0.0, "n": 0}
        for s in range(steps_per_epoch):
            idx = perm[s * BS:(s + 1) * BS]
            x = Xtr_gpu[idx]
            m1 = (torch.rand_like(x) > drop).to(x.dtype)
            m2 = (torch.rand_like(x) > drop).to(x.dtype)
            n1 = 1.0 + noise * torch.randn_like(x)
            n2 = 1.0 + noise * torch.randn_like(x)
            v1, v2 = x * m1 * n1, x * m2 * n2
            P = Ptr_gpu[idx]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                z1, z2 = enc(v1), enc(v2)
                rl = reg(proj(z1), proj(z2))["loss"]
                pl = pathway(z1, P)
                total = rl + pw * pl
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            agg["reg"] += rl.item() * BS; agg["path"] += pl.item() * BS; agg["n"] += BS
        dt = time.time() - t0
        n = max(1, agg["n"])
        msg = (f"[ep {ep:03d}] reg={agg['reg']/n:.4f} pathway={agg['path']/n:.4f} | "
               f"{dt:.1f}s ({n/dt:.0f} cells/s)")
        if ep % cfg.optim.probe_every == 0 or ep == cfg.optim.epochs:
            Etr = embed(enc, Xtr_gpu); Eva = embed(enc, Xva_gpu)
            for tname, arr in [("drug", "drug"), ("moa", "moa"), ("cell_line", "cell_line")]:
                ytr = getattr(tr, arr)[tr.ids].numpy(); yva = getattr(va, arr)[va.ids].numpy()
                if tname == "moa":
                    u = tr.moa_names.index("unclear") if "unclear" in tr.moa_names else -999
                    ytr = np.where(ytr == u, -1, ytr); yva = np.where(yva == u, -1, yva)
                r = probe(tname, Etr, ytr, Eva, yva)
                if r: msg += f" | {tname}: F1={r['macro_f1']:.3f}"
        print(msg, flush=True)

    print("\n== final linear-probe comparison (macro-F1) ==", flush=True)
    Xtr_raw = tr.X[tr.ids].numpy(); Xva_raw = va.X[va.ids].numpy()
    Etr = embed(enc, Xtr_gpu); Eva = embed(enc, Xva_gpu)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(D, 50)).fit(Xtr_raw)
    Ptr_raw, Pva_raw = pca.transform(Xtr_raw), pca.transform(Xva_raw)
    rand_enc, _ = _build_encoder(cfg, K, D, device)
    Etr_rnd = embed(rand_enc, Xtr_gpu); Eva_rnd = embed(rand_enc, Xva_gpu)
    final_metrics = {}
    for tname, arr in [("cell_line", "cell_line"), ("drug", "drug"), ("moa", "moa")]:
        ytr = getattr(tr, arr)[tr.ids].numpy(); yva = getattr(va, arr)[va.ids].numpy()
        if tname == "moa":
            u = tr.moa_names.index("unclear") if "unclear" in tr.moa_names else -999
            ytr = np.where(ytr == u, -1, ytr); yva = np.where(yva == u, -1, yva)
        final_metrics[tname] = {}
        for feat, Xt, Xv in [("raw", Xtr_raw, Xva_raw), ("pca50", Ptr_raw, Pva_raw),
                             ("random-enc", Etr_rnd, Eva_rnd), ("JEPA(ours)", Etr, Eva)]:
            r = probe(tname, Xt, ytr, Xv, yva)
            if r:
                print(f"  {tname:10s} {feat:11s} F1={r['macro_f1']:.3f} acc={r['acc']:.3f}",
                      flush=True)
                final_metrics[tname][feat] = {"macro_f1": r["macro_f1"], "acc": r["acc"]}

    ckpt = os.environ.get("EBJEPA_CKPTS", "checkpoints/tahoe_fast")
    os.makedirs(ckpt, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "cfg": OmegaConf.to_container(cfg)},
               os.path.join(ckpt, "tahoe_jepa.pt"))
    summary = {"encoder": enc_kind, "seed": int(cfg.meta.seed), "genes": int(K),
               "reg": str(cfg.loss.reg), "epochs": int(cfg.optim.epochs),
               "batch_size": BS, "bf16": use_amp, "metrics": final_metrics}
    with open(os.path.join(ckpt, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"metrics -> {os.path.join(ckpt, 'metrics.json')}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/train_fast.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
