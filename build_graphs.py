"""Parallel graph construction from already-folded pdbqt complexes.

Graph building is CPU-bound and embarrassingly parallel; ESM embeddings are
cached on disk so the receptor sequence is embedded once for a whole set.
Writes one .pt per complex plus a manifest, so a run can be resumed.
"""
import argparse, glob, os, time, traceback
from pathlib import Path
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

def _one(args):
    f, out_dir, cache, label, source_id = args
    cid = Path(f).stem
    dst = Path(out_dir) / f"{cid}.pt"
    if dst.exists():
        return cid, "cached", None
    try:
        from pepaffinity_data import build_complex, EsmEmbedder
        global _EMB
        try:
            _EMB
        except NameError:
            _EMB = EsmEmbedder(cache_dir=cache, device="cpu")
        out = build_complex(f, _EMB, y=label, source_id=source_id)
        if out is None:
            return cid, "unbuildable", None
        torch.save(tuple(d.cpu() for d in out), dst)
        return cid, "ok", None
    except Exception as e:
        return cid, "error", f"{type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbqt", required=True)
    ap.add_argument("--index", required=True, help="csv with cid + pAff (+ optional source_id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/mnt/sda/tmp/esm_cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(a.index)
    lab = dict(zip(idx.cid, idx.pAff))
    src = dict(zip(idx.cid, idx.source_id)) if "source_id" in idx.columns else {}
    files = sorted(glob.glob(os.path.join(a.pdbqt, "*.pdbqt")))
    files = [f for f in files if Path(f).stem in lab]
    if a.limit:
        files = files[:a.limit]
    jobs = [(f, a.out, a.cache, lab[Path(f).stem], int(src.get(Path(f).stem, 0)))
            for f in files]
    print(f"{len(jobs)} complexes -> {a.out}  ({a.workers} workers)")

    # Warm the embedding cache in ONE process first. Receptors repeat across a
    # set (all 548 ACE complexes share one), so the unique sequence count is far
    # below the complex count, and the workers then never load the 2.6 GB model.
    from pepaffinity_data import read_pdbqt, EsmEmbedder
    todo = [f for f in files if not (Path(a.out) / f"{Path(f).stem}.pt").exists()]
    if todo:
        seqs = set()
        for f in todo:
            for ch in read_pdbqt(f).values():
                seqs.add("".join(r["type"] for r in ch))
        emb = EsmEmbedder(cache_dir=a.cache)
        miss = [s for s in seqs
                if not (Path(a.cache) / f"{__import__('hashlib').sha1(s.encode()).hexdigest()}.pt").exists()]
        print(f"  {len(seqs)} unique sequences, {len(miss)} to embed")
        t = time.time()
        emb.warm(miss)
        print(f"  embeddings warmed in {time.time()-t:.0f}s")

    t0, counts, errs = time.time(), {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            cid, status, err = fu.result()
            counts[status] = counts.get(status, 0) + 1
            if err:
                errs.append((cid, err))
            if i % 100 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el:.0f}s  ({el/i:.2f}s each)  {counts}")
    print(f"\ndone in {time.time()-t0:.0f}s: {counts}")
    for cid, e in errs[:5]:
        print(f"  ERROR {cid}: {e}")
    pd.DataFrame([{"cid": Path(f).stem} for f in files]).to_csv(
        Path(a.out) / "manifest.csv", index=False)

if __name__ == "__main__":
    main()
