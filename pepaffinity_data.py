"""Graph construction for PepAffinity-GNN.

Reimplements the repo's pdb2graph + graph_construct pipeline as an importable
library rather than a script that runs on import, and attaches the three fields
M1/M2 need and the original pipeline does not produce:

    pep_mask   [N] bool  -- which nodes are peptide residues
    pep_feats  [N, 16]   -- M1 features, zero-filled for receptor nodes
    source_id  scalar    -- publication index, for M2's per-source offset

Graph definition follows the INFERENCE script, which is what the released
checkpoint behaves like (see the M0 fix): interface edges at 6 A between any
heavy atoms, intra-chain edges at 3.5 A, and a 280-d edge vector of
28 atom-pair types x 10 distance bins. Bin width is cutoff/10, so the bins mean
different distances in the inter and intra graphs -- that is the upstream
design, preserved deliberately.

Known limitation, measured and left in place: bin 10 is a catch-all for every
atom pair beyond 0.9*cutoff, and it collects ~90% of the mass, which makes the
edge vector largely a residue-size count. Changing it would break compatibility
with the released weights, so it belongs in a separate experiment, not here.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from pepaffinity import peptide_residue_features, PEPTIDE_FEATURE_DIM

ATOM_PAIR = ['A_A','A_C','A_OA','A_N','A_NA','A_SA','A_HD',
             'C_C','C_OA','C_N','C_NA','C_SA','C_HD',
             'OA_OA','OA_N','OA_NA','OA_SA','OA_HD',
             'N_N','N_NA','N_SA','N_HD',
             'NA_NA','NA_SA','NA_HD','SA_SA','SA_HD','HD_HD']
PAIR_IDX = {p: i for i, p in enumerate(ATOM_PAIR)}
N_BINS = 10
EDGE_DIM = len(ATOM_PAIR) * N_BINS          # 280
INTER_CUTOFF = 6.0
INTRA_CUTOFF = 3.5

THREE_TO_ONE = {
    'GLY':'G','ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','TYR':'Y',
    'ASP':'D','ASN':'N','GLU':'E','LYS':'K','GLN':'Q','MET':'M','SER':'S','THR':'T',
    'CYS':'C','PRO':'P','HIS':'H','ARG':'R','UNK':'X',
}


# ------------------------------------------------------------------ parsing

def read_pdbqt(path):
    """-> {chain: [ {num, type, atoms:[(pdbqt_type, x, y, z)]} ]}, in file order.

    Residues without a CA are dropped, matching upstream: they are incomplete
    and would give a meaningless centre.
    """
    chains = defaultdict(list)
    cur = None
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        ch, num, rname = line[21], line[22:27].strip(), line[17:21].strip()
        atom = (line[77:79].strip(), float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if cur is None or cur["chain"] != ch or cur["num"] != num:
            cur = {"chain": ch, "num": num, "type": THREE_TO_ONE.get(rname, "X"),
                   "atoms": [], "names": []}
            chains[ch].append(cur)
        cur["atoms"].append(atom)
        cur["names"].append(line[12:16].strip())
    out = {}
    for ch, res in chains.items():
        out[ch] = [r for r in res if "CA" in r["names"]]
    return out


def _coords(res):
    return np.array([[a[1], a[2], a[3]] for a in res["atoms"]], dtype=float)


def _edge_vector(ra, rb, cutoff):
    """28 x 10 atom-pair-type x distance-bin histogram, flattened."""
    enc = np.zeros(EDGE_DIM, dtype=np.float32)
    width = cutoff / N_BINS
    for ta, xa, ya, za in ra["atoms"]:
        for tb, xb, yb, zb in rb["atoms"]:
            d = math.dist((xa, ya, za), (xb, yb, zb))
            b = min(N_BINS, max(1, math.ceil(d / width)))
            key = f"{ta}_{tb}"
            if key not in PAIR_IDX:
                key = f"{tb}_{ta}"
            if key not in PAIR_IDX:          # metal, modified residue, ...
                continue                     # upstream raises here and drops the complex
            enc[(b - 1) * len(ATOM_PAIR) + PAIR_IDX[key]] += 1.0
    return enc


def _pairs_within(listA, listB, cutoff, same=False):
    """Residue pairs with any heavy-atom distance <= cutoff."""
    ca = [_coords(r) for r in listA]
    cb = ca if same else [_coords(r) for r in listB]
    out = []
    for i, A in enumerate(ca):
        start = i + 1 if same else 0
        for j in range(start, len(cb)):
            d = np.sqrt(((A[:, None, :] - cb[j][None, :, :]) ** 2).sum(-1)).min()
            if d <= cutoff:
                out.append((i, j))
    return out


# ------------------------------------------------------------------ embeddings

class EsmEmbedder:
    """ESM-2 650M residue embeddings, cached per sequence. Loading the model is
    the expensive part, so one instance is reused across a whole set."""

    def __init__(self, name="facebook/esm2_t33_650M_UR50D", device=None, cache_dir=None):
        # The 650M model is loaded LAZILY. Graph building parallelises across
        # processes, and every worker eagerly loading 2.6 GB of weights costs
        # more than the geometry it is there to compute. With the disk cache
        # warmed first, workers never touch the model at all.
        self.name = name
        self.tok = None
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_model(self):
        if self.model is not None:
            return
        from transformers import AutoTokenizer, EsmModel
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = EsmModel.from_pretrained(self.name).eval().to(self.device)

    def warm(self, seqs):
        """Embed and cache a set of sequences up front, so parallel workers can
        run without ever loading the model."""
        for s in dict.fromkeys(seqs):
            self(s)

    @torch.no_grad()
    def __call__(self, seq: str) -> torch.Tensor:
        if seq in self.cache:
            return self.cache[seq]
        # hashlib, NOT hash(): PYTHONHASHSEED is randomised per process, so
        # hash() would give every worker a different filename for the same
        # sequence and the cache would never hit across processes.
        import hashlib
        fp = (self.cache_dir / f"{hashlib.sha1(seq.encode()).hexdigest()}.pt"
              if self.cache_dir else None)
        if fp is not None and fp.exists():
            emb = torch.load(fp, map_location="cpu")
        else:
            self._ensure_model()
            enc = self.tok(seq, return_tensors="pt").to(self.device)
            emb = self.model(**enc).last_hidden_state.squeeze(0)[1:-1].cpu()
            if fp is not None:
                torch.save(emb, fp)
        self.cache[seq] = emb
        return emb


# ------------------------------------------------------------------ build

def build_complex(pdbqt, embedder, y=None, source_id=0,
                  rec_chain="A", pep_chain="B"):
    """-> (inter, intra_rec, intra_pep) Data objects, or None if unbuildable."""
    chains = read_pdbqt(pdbqt)
    if rec_chain not in chains or pep_chain not in chains:
        return None
    rec, pep = chains[rec_chain], chains[pep_chain]
    if not rec or not pep:
        return None

    seq_rec = "".join(r["type"] for r in rec)
    seq_pep = "".join(r["type"] for r in pep)
    x = torch.cat([embedder(seq_rec), embedder(seq_pep)], 0).float()
    n_rec, n_pep = len(rec), len(pep)
    if x.shape[0] != n_rec + n_pep:
        return None

    # M1 fields: peptide nodes come after the receptor, matching node order
    pep_mask = torch.zeros(n_rec + n_pep, dtype=torch.bool)
    pep_mask[n_rec:] = True
    pep_feats = torch.zeros(n_rec + n_pep, PEPTIDE_FEATURE_DIM)
    pf = peptide_residue_features(seq_pep)
    if pf.shape[0] == n_pep:
        pep_feats[n_rec:] = pf

    def pack(pairs, listA, listB, offA, offB, cutoff, n_nodes, undirected=True):
        if not pairs:
            return (torch.empty(2, 0, dtype=torch.long), torch.empty(0, EDGE_DIM))
        ei, ea = [], []
        for i, j in pairs:
            ei.append((offA + i, offB + j))
            ea.append(_edge_vector(listA[i], listB[j], cutoff))
        ei = torch.tensor(ei, dtype=torch.long).t()
        ea = torch.tensor(np.array(ea), dtype=torch.float)
        if undirected:                       # upstream duplicates both directions
            ei = torch.cat([ei, ei.flip(0)], 1)
            ea = torch.cat([ea, ea], 0)
        return ei, ea

    inter_pairs = _pairs_within(rec, pep, INTER_CUTOFF)
    ei, ea = pack(inter_pairs, rec, pep, 0, n_rec, INTER_CUTOFF, n_rec + n_pep)
    inter = Data(x=x, edge_index=ei, edge_attr=ea)
    inter.pep_mask = pep_mask
    inter.pep_feats = pep_feats
    inter.source_id = torch.tensor([int(source_id)], dtype=torch.long)
    if y is not None:
        inter.y = torch.tensor([float(y)], dtype=torch.float)

    rp = _pairs_within(rec, rec, INTRA_CUTOFF, same=True)
    ei1, ea1 = pack(rp, rec, rec, 0, 0, INTRA_CUTOFF, n_rec)
    intra_rec = Data(x=embedder(seq_rec).float(), edge_index=ei1, edge_attr=ea1)

    pp = _pairs_within(pep, pep, INTRA_CUTOFF, same=True)
    ei2, ea2 = pack(pp, pep, pep, 0, 0, INTRA_CUTOFF, n_pep)
    intra_pep = Data(x=embedder(seq_pep).float(), edge_index=ei2, edge_attr=ea2)

    return inter, intra_rec, intra_pep
