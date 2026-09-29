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


def load_graphs(graph_dirs, index_csv, cluster_csv=None, conf_csv=None):
    """graph_dirs is one directory or several -- one per diffusion pose of the
    SAME complexes. With several, a complex is kept only if every pose built, so
    all pose arms score the identical set and the comparison is paired."""
    idx = pd.read_csv(index_csv)
    if cluster_csv and Path(cluster_csv).exists():
        cl = pd.read_csv(cluster_csv)[["peptide", "seq_cluster"]].drop_duplicates("peptide")
        idx = idx.merge(cl, on="peptide", how="left")
    if "seq_cluster" not in idx or idx.seq_cluster.isna().all():
        idx["seq_cluster"] = np.arange(len(idx))
    idx["seq_cluster"] = idx.seq_cluster.fillna(
        pd.Series(np.arange(len(idx)) + 10 ** 7, index=idx.index)).astype(int)

    dirs = [graph_dirs] if isinstance(graph_dirs, (str, Path)) else list(graph_dirs)
    best = {}
    if conf_csv and Path(conf_csv).exists():
        c = pd.read_csv(conf_csv)
        best = dict(zip(c.cid.astype(str), c.best_pose.astype(int)))

    items = []
    for r in idx.itertuples():
        paths = [Path(d) / f"{r.cid}.pt" for d in dirs]
        if not all(p.exists() for p in paths):
            continue
        items.append(dict(cid=r.cid, paths=paths, path=paths[0], y=float(r.pAff),
                          source=int(getattr(r, "source_id", 0)),
                          cluster=int(r.seq_cluster),
                          best_pose=min(best.get(str(r.cid), 0), len(paths) - 1)))
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


def _load_trio(it, device, pose=0):
    """Batch.from_data_list builds the `batch` vector AttentiveFP's readout
    needs, so the stored single-complex graphs do not carry one."""
    inter, ir, ip = torch.load(it["paths"][pose], map_location="cpu", weights_only=False)
    return [d.to(device) for d in (inter, ir, ip)]


def pose_policy(mode, it, n_poses, rng):
    """Which pose to TRAIN on, and which to AVERAGE over at test.

    Boltz's samples of one complex place the peptide a median 10.3 A apart on
    panelB while the pocket superposes to 0.13 A, so "the structure" is really a
    distribution and `p0` -- what every earlier result used -- is one draw from it.
      p0/p1/p2   three independent draws; their spread IS the draw sensitivity
      bestconf   the pose Boltz scores highest (confidence vs disagreement is
                 only r = -0.22, so this is a weak selector and expected to be weak)
      ensemble   train on p0, average predictions over all poses at test
      augment    a fresh random pose every time a complex is seen in training,
                 then average at test -- the only arm that tells the model the
                 pose is uncertain
    """
    if mode in ("p0", "p1", "p2"):
        k = min(int(mode[1]), n_poses - 1)
        return k, [k]
    if mode == "bestconf":
        k = it["best_pose"]
        return k, [k]
    if mode == "ensemble":
        return 0, list(range(n_poses))
    if mode == "augment":
        return int(rng.integers(n_poses)), list(range(n_poses))
    raise ValueError(mode)


def run_fold(train_items, test_items, arm, epochs, lr, batch, device, seed,
             rank_weight=2.0, fps=None, pose_mode="p0"):
    torch.manual_seed(seed)
    n_src = max([it["source"] for it in train_items + test_items]) + 1
    use_fp = arm.endswith("+FP") and fps is not None
    fp_dim = next(iter(fps.values())).numel() if use_fp else 0
    net = PepAffinityNet(load_pretrained(GraphNetwork(dropout=0.1)),
                         n_sources=n_src,
                         use_m1="M1" in arm,
                         use_m2="M2" in arm,
                         use_m3="M3" in arm,
                         fp_dim=fp_dim).to(device)
    def fpv(chunk):
        if not use_fp: return None
        return torch.stack([fps[it["cid"]] for it in chunk]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    rank = PairwiseRankingLoss(max_pairs_per_group=50)

    n_poses = len(train_items[0]["paths"])
    prng = np.random.default_rng(seed + 991)
    cache = {}
    def trio(it, pose):
        k = (it["cid"], pose)
        if k not in cache:
            cache[k] = _load_trio(it, device, pose)
        return cache[k]

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
            trios = [trio(it, pose_policy(pose_mode, it, n_poses, prng)[0])
                     for it in chunk]
            ba, bb, bc = (Batch.from_data_list([t[k] for t in trios]) for k in range(3))
            ys = [it["y"] for it in chunk]
            gids = [it["source"] for it in chunk]
            sid = (torch.tensor(gids, device=device) if "M2" in arm else None)
            preds = net(ba, bb, bc, sid, fpv(chunk)).view(-1)
            y = torch.tensor(ys, device=device, dtype=torch.float)
            g = torch.tensor(gids, device=device)
            parts = pepaffinity_loss(preds, y, g if "M2" in arm else None,
                                     net.source, w_reg=1.0,
                                     w_rank=rank_weight if "M2" in arm else 0.0,
                                     w_offset=0.01,
                                     rank_fn=rank if "M2" in arm else None)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(test_items), batch):
            chunk = test_items[s:s + batch]
            # A complex may be scored on several poses; average the PREDICTIONS
            # rather than the structures, since averaging coordinates 10 A apart
            # produces a geometry that is not a pose of anything.
            acc = np.zeros(len(chunk)); cnt = np.zeros(len(chunk))
            npose = max(len(pose_policy(pose_mode, it, n_poses, prng)[1]) for it in chunk)
            for pi in range(npose):
                sel = [(j, it) for j, it in enumerate(chunk)
                       if pi < len(pose_policy(pose_mode, it, n_poses, prng)[1])]
                if not sel:
                    continue
                trios = [trio(it, pose_policy(pose_mode, it, n_poses, prng)[1][pi])
                         for _, it in sel]
                ba, bb, bc = (Batch.from_data_list([t[k] for t in trios]) for k in range(3))
                p = net(ba, bb, bc, None, fpv([it for _, it in sel])).view(-1).cpu().numpy()
                for (j, _), v in zip(sel, p):
                    acc[j] += float(v); cnt[j] += 1
            out += [(it["cid"], it["y"], acc[j] / cnt[j], it["source"])
                    for j, it in enumerate(chunk)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="/mnt/sda/tmp/graphs_ace",
                    help="one graph dir, or several comma-separated -- one per pose")
    ap.add_argument("--pose_modes", default="p0",
                    help="p0,p1,p2,bestconf,ensemble,augment (see pose_policy)")
    ap.add_argument("--conf", default="", help="csv with cid,best_pose for bestconf")
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
    ap.add_argument("--fingerprints", default="",
                    help="path to a {cid: tensor} .pt of ECFP counts; enables the M1+FP arm")
    ap.add_argument("--rank_weight", type=float, default=2.0,
                    help="M2 ranking loss weight relative to regression")
    ap.add_argument("--out", default="/mnt/sda/home/tuanhai/peptides_exp/baselines/scores/pepaff_ace513.csv")
    ap.add_argument("--folds_csv", default="",
                    help="csv with cid,fold giving an EXPLICIT fold assignment, "
                         "overriding --split. Needed because grouped k-fold over "
                         "31 MHC publications is misleading: two of them hold 65% "
                         "of the rows and the inverse-Simpson effective count is "
                         "4.3, so a random grouping still trains on a dominant "
                         "publication in every fold. This lets the dominant "
                         "publications be held out by name.")
    ap.add_argument("--parts", default="",
                    help="directory for per-(arm,seed) partial results; default "
                         "'<out>.parts'. A seed already present there is reloaded "
                         "instead of retrained, so a killed run resumes at seed "
                         "granularity. Pass 'none' to disable.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    FPS = torch.load(a.fingerprints) if a.fingerprints else None
    dirs = [d for d in a.graphs.split(",") if d]
    items = load_graphs(dirs, a.index, a.clusters, a.conf)
    if FPS is not None:
        items = [it for it in items if it["cid"] in FPS]
        print(f"fingerprints loaded: {len(FPS)} entries, "
              f"{next(iter(FPS.values())).numel()}-d", flush=True)
    key = "cluster" if a.split == "cluster" else "source"   # within_group uses source too
    FIXED_FOLDS = None
    if a.folds_csv:
        fa = pd.read_csv(a.folds_csv)
        m = dict(zip(fa.cid.astype(str), fa.fold.astype(int)))
        miss = [it["cid"] for it in items if str(it["cid"]) not in m]
        if miss:
            raise SystemExit(f"--folds_csv is missing {len(miss)} of {len(items)} cids, "
                             f"e.g. {miss[:3]} -- refusing to silently drop them")
        FIXED_FOLDS = np.array([m[str(it["cid"])] for it in items])
        a.folds = int(FIXED_FOLDS.max()) + 1
        print(f"explicit folds from {a.folds_csv}: {a.folds} folds, sizes "
              f"{np.bincount(FIXED_FOLDS).tolist()}  (seeds vary only the model init)")
    ys = np.array([it["y"] for it in items])
    print(f"{len(dirs)} pose dir(s) | modes={a.pose_modes}")
    print(f"{len(items)} graphs | split={a.split} "
          f"({len({it[key] for it in items})} groups) | "
          f"pAff {ys.min():.2f}-{ys.max():.2f} sd {ys.std():.2f} | {a.device}\n")

    # Resume support. Predictions used to exist only in memory until the single
    # to_csv at the very end, so killing a run -- for GPU headroom, an OOM, a
    # reboot -- threw away every prediction it had made, in one case over three
    # hours of them. One file per (arm, seed) costs nothing and caps the loss at
    # the seed currently in flight.
    parts = None
    if a.parts != "none":
        parts = Path(a.parts) if a.parts else Path(str(a.out) + ".parts")
        parts.mkdir(parents=True, exist_ok=True)

    def part_path(label, seed):
        return parts / f"{label.replace('/', '__')}_seed{seed}.csv"

    rows = []
    for arm, pmode in [(x, y) for x in a.arms.split(",") for y in a.pose_modes.split(",")]:
        label = arm if pmode == "p0" and len(dirs) == 1 else f"{arm}/{pmode}"
        per_seed = []
        for seed in range(a.seeds):
            pp = part_path(label, seed) if parts is not None else None
            if pp is not None and pp.exists():
                done = pd.read_csv(pp)
                wg = within_group(done)
                pooled = spearmanr(done.y, done.pred).statistic
                per_seed.append((wg, pooled))
                rows += done.to_dict(orient="records")
                print(f"  {label:14s} seed {seed}  within-group rho = {wg:+.3f}   "
                      f"(pooled {pooled:+.3f})   [resumed]", flush=True)
                continue
            folds = (FIXED_FOLDS if FIXED_FOLDS is not None else
                     make_folds(items, key, a.folds, seed,
                                within_group=(a.split == "within_group")))
            preds = []
            t0 = time.time()
            for f in range(a.folds):
                tr = [it for it, g in zip(items, folds) if g != f]
                te = [it for it, g in zip(items, folds) if g == f]
                if len(te) < 3 or len(tr) < 20:
                    continue
                preds += run_fold(tr, te, arm, a.epochs, a.lr, a.batch, a.device, seed,
                                  a.rank_weight, FPS, pmode)
            df = pd.DataFrame(preds, columns=["cid", "y", "pred", "group"])
            pooled = spearmanr(df.y, df.pred).statistic
            wg = within_group(df)
            per_seed.append((wg, pooled))
            new = [dict(arm=label, seed=seed, cid=c, y=y, pred=p, group=g)
                   for c, y, p, g in preds]
            rows += new
            if pp is not None:
                pd.DataFrame(new).to_csv(pp, index=False)
            print(f"  {label:14s} seed {seed}  within-group rho = {wg:+.3f}   "
                  f"(pooled {pooled:+.3f})   {time.time()-t0:.0f}s", flush=True)
        w = [x[0] for x in per_seed]; pl = [x[1] for x in per_seed]
        print(f"  {label:14s} MEAN within-group = {np.mean(w):+.3f} sd {np.std(w):.3f}   "
              f"| pooled {np.mean(pl):+.3f} sd {np.std(pl):.3f}\n", flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
