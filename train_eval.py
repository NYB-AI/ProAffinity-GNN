"""Fine-tune ProAffinity-GNN on a prepared graph set and evaluate held-out folds.

Arms
  M0     backbone only, fine-tuned from the released checkpoint
  M1     + peptide feature adapter
  M1+M2  + source offset (train only) and within-group pairwise ranking loss

Splits are always GROUPED, never random:
  cluster  peptide sequence clusters  -> cold-peptide  (the default; ACE is one
                                        target, so this is the meaningful split)
  source   publication                -> leave-publication-out

Reported per arm: pooled held-out Spearman over all folds, plus the mean and sd
of the per-fold values. Seeds are repeated because a single seed on this data is
noise -- measured seed sd is ~0.05 on ACE CV, wider than most effects we chase.

Zero-shot ProAffinity on ACE-513 scored rho = +0.135 / r = +0.125; a fine-tuned
arm that does not clear that has not earned its training time.
"""
import argparse, glob, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch_geometric.data import Batch

from model_def import GraphNetwork, load_pretrained
from pepaffinity import PepAffinityNet, PairwiseRankingLoss, pepaffinity_loss


def load_graphs(graph_dir, index_csv, cluster_csv=None):
    idx = pd.read_csv(index_csv)
    if cluster_csv and Path(cluster_csv).exists():
        cl = pd.read_csv(cluster_csv)[["peptide", "seq_cluster"]].drop_duplicates("peptide")
        idx = idx.merge(cl, on="peptide", how="left")
    if "seq_cluster" not in idx or idx.seq_cluster.isna().all():
        idx["seq_cluster"] = np.arange(len(idx))
    idx["seq_cluster"] = idx.seq_cluster.fillna(
        pd.Series(np.arange(len(idx)) + 10 ** 7, index=idx.index)).astype(int)

    items = []
    for r in idx.itertuples():
        p = Path(graph_dir) / f"{r.cid}.pt"
        if not p.exists():
            continue
        items.append(dict(cid=r.cid, path=p, y=float(r.pAff),
                          source=int(getattr(r, "source_id", 0)),
                          cluster=int(r.seq_cluster)))
    return items


def within_group(df):
    """Mean Spearman inside each held-out group -- the task metric.

    Pooled cross-target Spearman is misleading here: it rewards getting each
    target's overall LEVEL right, which is worth nothing when screening peptides
    against one target. Measured on panelB, fine-tuning lifted pooled rho from
    -0.033 to +0.338 while within-group rho went from +0.130 to +0.123, i.e. the
    entire apparent gain was target calibration.
    """
    rs = []
    for _, g in df.groupby("group"):
        if len(g) < 4 or g.y.nunique() < 3:
            continue
        r = spearmanr(g.y, g.pred).statistic
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def make_folds(items, key, k=5, seed=0, within_group=False):
    """Grouped k-fold: every member of a group lands in the same fold.

    within_group=True instead splits INSIDE each group, so every group appears in
    both training and test. That is the per-target setting -- the regime where
    ECFP-16 beats the length baseline (+0.230 vs +0.199) while losing to it when
    the target is unseen (+0.167). It is also the paper's "within-target"
    partition. Scores are still reported per group, never pooled across targets.
    """
    rng = np.random.default_rng(seed)
    if within_group:
        fold = np.empty(len(items), dtype=int)
        by = {}
        for i, it in enumerate(items):
            by.setdefault(it[key], []).append(i)
        for _, idx in by.items():
            fold[np.array(idx)] = rng.permutation(len(idx)) % k
        return fold
    groups = sorted({it[key] for it in items})
    assign = {g: i for g, i in zip(rng.permutation(groups), np.arange(len(groups)) % k)}
    return np.array([assign[it[key]] for it in items])


def _load_trio(it, device):
    """Batch.from_data_list builds the `batch` vector AttentiveFP's readout
    needs, so the stored single-complex graphs do not carry one."""
    inter, ir, ip = torch.load(it["path"], map_location="cpu", weights_only=False)
    return [d.to(device) for d in (inter, ir, ip)]


def run_fold(train_items, test_items, arm, epochs, lr, batch, device, seed,
             rank_weight=2.0):
    torch.manual_seed(seed)
    n_src = max([it["source"] for it in train_items + test_items]) + 1
    net = PepAffinityNet(load_pretrained(GraphNetwork(dropout=0.1)),
                         n_sources=n_src,
                         use_m1=arm in ("M1", "M1+M2"),
                         use_m2=arm == "M1+M2").to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    rank = PairwiseRankingLoss(max_pairs_per_group=50)

    cache = {}
    def trio(it):
        if it["cid"] not in cache:
            cache[it["cid"]] = _load_trio(it, device)
        return cache[it["cid"]]

    order = np.arange(len(train_items))
    rng = np.random.default_rng(seed)
    net.train()
    for ep in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            chunk = [train_items[i] for i in order[s:s + batch]]
            if len(chunk) < 2:
                continue
            opt.zero_grad()
            # One collated forward pass per minibatch. Looping complex-by-complex
            # leaves the GPU ~30% utilised: the cost is Python and kernel-launch
            # overhead, not arithmetic. Batch.from_data_list concatenates the
            # node-level fields (x, pep_mask, pep_feats) and builds the `batch`
            # vector AttentiveFP's readout needs.
            trios = [trio(it) for it in chunk]
            ba, bb, bc = (Batch.from_data_list([t[k] for t in trios]) for k in range(3))
            ys = [it["y"] for it in chunk]
            gids = [it["source"] for it in chunk]
            sid = (torch.tensor(gids, device=device) if arm == "M1+M2" else None)
            preds = net(ba, bb, bc, sid).view(-1)
            y = torch.tensor(ys, device=device, dtype=torch.float)
            g = torch.tensor(gids, device=device)
            parts = pepaffinity_loss(preds, y, g if arm == "M1+M2" else None,
                                     net.source, w_reg=1.0,
                                     w_rank=rank_weight if arm == "M1+M2" else 0.0,
                                     w_offset=0.01,
                                     rank_fn=rank if arm == "M1+M2" else None)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(test_items), batch):
            chunk = test_items[s:s + batch]
            trios = [trio(it) for it in chunk]
            ba, bb, bc = (Batch.from_data_list([t[k] for t in trios]) for k in range(3))
            p = net(ba, bb, bc, None).view(-1).cpu()
            out += [(it["cid"], it["y"], float(v), it["source"]) for it, v in zip(chunk, p)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="/mnt/sda/tmp/graphs_ace")
    ap.add_argument("--index", default="/mnt/sda/tmp/ace_clean/index_src.csv")
    ap.add_argument("--clusters", default="/mnt/sda/home/tuanhai/peptides_exp/baselines/corpus/corpus.csv")
    ap.add_argument("--arms", default="M0,M1")
    ap.add_argument("--split", default="cluster",
                    choices=["cluster", "source", "within_group"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rank_weight", type=float, default=2.0,
                    help="M2 ranking loss weight relative to regression")
    ap.add_argument("--out", default="/mnt/sda/home/tuanhai/peptides_exp/baselines/scores/pepaff_ace513.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    items = load_graphs(a.graphs, a.index, a.clusters)
    key = "cluster" if a.split == "cluster" else "source"   # within_group uses source too
    ys = np.array([it["y"] for it in items])
    print(f"{len(items)} graphs | split={a.split} "
          f"({len({it[key] for it in items})} groups) | "
          f"pAff {ys.min():.2f}-{ys.max():.2f} sd {ys.std():.2f} | {a.device}\n")

    rows = []
    for arm in a.arms.split(","):
        per_seed = []
        for seed in range(a.seeds):
            folds = make_folds(items, key, a.folds, seed,
                               within_group=(a.split == "within_group"))
            preds = []
            t0 = time.time()
            for f in range(a.folds):
                tr = [it for it, g in zip(items, folds) if g != f]
                te = [it for it, g in zip(items, folds) if g == f]
                if len(te) < 3 or len(tr) < 20:
                    continue
                preds += run_fold(tr, te, arm, a.epochs, a.lr, a.batch, a.device, seed,
                                  a.rank_weight)
            df = pd.DataFrame(preds, columns=["cid", "y", "pred", "group"])
            pooled = spearmanr(df.y, df.pred).statistic
            wg = within_group(df)
            per_seed.append((wg, pooled))
            rows += [dict(arm=arm, seed=seed, cid=c, y=y, pred=p, group=g)
                     for c, y, p, g in preds]
            print(f"  {arm:6s} seed {seed}  within-group rho = {wg:+.3f}   "
                  f"(pooled {pooled:+.3f})   {time.time()-t0:.0f}s", flush=True)
        w = [x[0] for x in per_seed]; pl = [x[1] for x in per_seed]
        print(f"  {arm:6s} MEAN within-group = {np.mean(w):+.3f} sd {np.std(w):.3f}   "
              f"| pooled {np.mean(pl):+.3f} sd {np.std(pl):.3f}\n", flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
