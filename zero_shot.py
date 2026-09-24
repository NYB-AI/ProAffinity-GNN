"""Score prepared graphs with the released checkpoint, no training.

The baseline every fine-tuned arm has to beat. On ACE-513 the fine-tuned arms
came in BELOW this, which is only visible if it is measured on the same graphs
rather than assumed.
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, pearsonr
from torch_geometric.data import Batch
from model_def import GraphNetwork, load_pretrained

ap = argparse.ArgumentParser()
ap.add_argument("--graphs", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

idx = pd.read_csv(a.index)
items = [r for r in idx.itertuples() if (Path(a.graphs) / f"{r.cid}.pt").exists()]
print(f"{len(items)} graphs", flush=True)

net = load_pretrained(GraphNetwork(dropout=0.0)).to(a.device).eval()
rows, t0 = [], time.time()
with torch.no_grad():
    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        trios = [torch.load(Path(a.graphs) / f"{r.cid}.pt", map_location="cpu",
                            weights_only=False) for r in chunk]
        bs = [Batch.from_data_list([t[k] for t in trios]).to(a.device) for k in range(3)]
        p = net(*bs).view(-1).cpu().numpy()
        rows += [dict(cid=r.cid, y=float(r.pAff), pred=float(v),
                      group=getattr(r, "source_id", 0)) for r, v in zip(chunk, p)]
        if (s // a.batch) % 10 == 0:
            print(f"  {s+len(chunk)}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

d = pd.DataFrame(rows)
d.to_csv(a.out, index=False)
rho = spearmanr(d.y, d.pred).statistic
print(f"\nPOOLED  rho = {rho:+.3f}   r = {pearsonr(d.y, d.pred)[0]:+.3f}   n = {len(d)}")
wg = [spearmanr(g.y, g.pred).statistic for _, g in d.groupby("group")
      if g.y.nunique() >= 3 and len(g) >= 4]
wg = [v for v in wg if np.isfinite(v)]
print(f"WITHIN-GROUP  mean rho = {np.mean(wg):+.3f}  median {np.median(wg):+.3f}  ({len(wg)} groups)")
print(f"wrote {a.out}")
