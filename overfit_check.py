"""Can PepAffinity-GNN overfit a small set?

This is a sanity check, not a result. A model that CANNOT drive training loss
toward zero on a handful of examples has a wiring bug -- a detached gradient, a
frozen module, a label that never reaches the loss. A model that CAN overfit is
merely not broken; it says nothing about generalisation.

Three arms, so the check also tells us the new modules are actually connected:
  M0    backbone only, the released architecture
  M1    + peptide feature adapter
  M1+M2 + source offset and within-group ranking loss

Reported per arm: train MSE, train Spearman, and -- for M1 -- how far the
adapter has moved from its zero initialisation. If M1's delta stays at zero the
adapter is receiving no gradient and the arm is secretly M0.
"""
import argparse, glob, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from model_def import GraphNetwork, load_pretrained
from pepaffinity import PepAffinityNet, PairwiseRankingLoss, pepaffinity_loss
from pepaffinity_data import build_complex, EsmEmbedder


def load_set(n, pdbqt_dir, index_csv, cache):
    idx = pd.read_csv(index_csv)
    lab = dict(zip(idx.cid, idx.pAff)) if "cid" in idx else {}
    emb = EsmEmbedder(cache_dir=cache)
    items, files = [], sorted(glob.glob(os.path.join(pdbqt_dir, "*.pdbqt")))
    for f in files:
        cid = Path(f).stem
        if cid not in lab:
            continue
        # one synthetic group per 8 complexes, so the ranking loss has pairs;
        # the real corpus supplies true (paper x target x assay) groups
        out = build_complex(f, emb, y=lab[cid], source_id=len(items) // 8)
        if out is None:
            continue
        items.append(out)
        if len(items) >= n:
            break
    return items


def run(items, arm, epochs, lr, device, seed=0):
    torch.manual_seed(seed)
    backbone = load_pretrained(GraphNetwork(dropout=0.0))   # no dropout: we WANT to overfit
    n_src = max(int(it[0].source_id) for it in items) + 1
    net = PepAffinityNet(backbone, n_sources=n_src,
                         use_m1=arm in ("M1", "M1+M2"),
                         use_m2=arm == "M1+M2").to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rank = PairwiseRankingLoss(max_pairs_per_group=50)
    y = torch.tensor([float(it[0].y) for it in items], device=device)
    gid = torch.tensor([int(it[0].source_id) for it in items], device=device)
    sid = gid if arm == "M1+M2" else None

    batched = []
    for inter, ir, ip in items:
        trio = []
        for d in (inter, ir, ip):
            d = d.clone()
            d.batch = torch.zeros(d.x.shape[0], dtype=torch.long)
            trio.append(d.to(device))
        batched.append(trio)

    hist = []
    net.train()
    for ep in range(epochs):
        opt.zero_grad()
        preds = torch.cat([net(a, b, c,
                               sid[i:i+1] if sid is not None else None).view(1)
                           for i, (a, b, c) in enumerate(batched)])
        parts = pepaffinity_loss(preds, y, gid if arm == "M1+M2" else None,
                                 net.source, w_reg=1.0,
                                 w_rank=2.0 if arm == "M1+M2" else 0.0,
                                 w_offset=0.01,
                                 rank_fn=rank if arm == "M1+M2" else None)
        parts["total"].backward()
        opt.step()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            with torch.no_grad():
                rho = spearmanr(y.cpu().numpy(), preds.detach().cpu().numpy()).statistic
            hist.append((ep, float(parts["reg"]), float(parts["rank"]), float(rho)))
    delta = (float(net.pep_encoder.mlp[-1].weight.abs().max())
             if net.pep_encoder is not None else float("nan"))
    return hist, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pdbqt", default="/mnt/sda/tmp/ace_proaff")
    ap.add_argument("--index", default="/mnt/sda/tmp/ace_clean/index.csv")
    ap.add_argument("--cache", default="/mnt/sda/tmp/esm_cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    t = time.time()
    items = load_set(a.n, a.pdbqt, a.index, a.cache)
    ys = np.array([float(it[0].y) for it in items])
    print(f"loaded {len(items)} complexes in {time.time()-t:.0f}s  "
          f"pAff {ys.min():.2f}-{ys.max():.2f} sd {ys.std():.2f}  device {a.device}\n")

    for arm in ["M0", "M1", "M1+M2"]:
        t = time.time()
        hist, delta = run(items, arm, a.epochs, a.lr, a.device)
        print(f"--- {arm} ---  ({time.time()-t:.0f}s)")
        print(f"   {'epoch':>6s} {'train MSE':>10s} {'rank':>7s} {'train rho':>10s}")
        for ep, mse, rk, rho in hist:
            print(f"   {ep:>6d} {mse:>10.4f} {rk:>7.3f} {rho:>+10.3f}")
        if not np.isnan(delta):
            print(f"   M1 adapter max|w| moved from 0.0 to {delta:.4f}"
                  f"{'  <-- STILL ZERO, adapter is dead' if delta == 0 else ''}")
        print()


if __name__ == "__main__":
    main()
