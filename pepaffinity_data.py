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


# Metal ions are lost by read_pdbqt: they sit in residues with no CA, and those are
# dropped to match upstream. For a metalloprotease like ACE the catalytic Zn2+ is the
# interaction peptide inhibitors are built around, so it has to be recovered
# separately rather than through the residue list.
METAL_TYPES = {"ZN", "MG", "MN", "FE", "CU", "NI", "CO", "CD", "HG"}


def read_metals(path):
    """-> [(type, x, y, z)] for metal ions anywhere in the file, ATOM or HETATM."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        t = line[77:79].strip().upper()
        if t in METAL_TYPES:
            out.append((t, float(line[30:38]), float(line[38:46]), float(line[46:54])))
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


# ---------------------------------------------------------------- M3 edge features
# The existing 280-d edge vector is a 28 atom-pair-type x 10 distance-bin histogram,
# so contact multiplicity and coarse distance are already represented. What it does
# NOT carry is residue-level interaction chemistry (a salt bridge is not deducible
# from PDBQT atom types alone), continuous distance, which parts of each residue
# touch, or relative orientation. Module M3 of the plan asks for exactly those.
EDGE_M3_DIM = 26
_RBF_MU = np.linspace(2.0, 6.0, 8, dtype=np.float32)
_RBF_SIG = 0.6
_BB = {"N", "CA", "C", "O", "OXT"}
_POS, _NEG = set("KR"), set("DE")
_ARO, _HYD = set("FWY"), set("AVLIMFWCY")
_DONOR_T, _ACCEPT_T = {"HD", "HS"}, {"OA", "NA", "SA", "OS", "NS"}


def _bb_mask(r):
    """Backbone atoms, by NAME. The stored PDBQT `type` cannot distinguish a
    backbone carbon from a side-chain one -- both are "C" -- so this has to use
    read_pdbqt's parallel `names` list."""
    return np.array([n in _BB for n in r["names"]], dtype=bool)


def _side_centroid(r):
    """Centroid of the side chain, falling back to the whole residue for GLY."""
    bb = _bb_mask(r)
    A = np.asarray([(x, y, z) for _, x, y, z in r["atoms"]], dtype=np.float32)
    if len(A) == 0:
        return np.zeros(3, dtype=np.float32)
    sc = A[~bb] if (~bb).any() else A
    return sc.mean(0)


def _ca(r):
    for n, (t, x, y, z) in zip(r["names"], r["atoms"]):
        if n == "CA":
            return np.array([x, y, z], dtype=np.float32)
    return np.asarray([(x, y, z) for _, x, y, z in r["atoms"]], dtype=np.float32).mean(0)


def _edge_m3_vector(ra, rb, metals=()):
    """[26] interaction descriptor for one receptor-peptide residue pair."""
    A = np.asarray([(x, y, z) for _, x, y, z in ra["atoms"]], dtype=np.float32)
    B = np.asarray([(x, y, z) for _, x, y, z in rb["atoms"]], dtype=np.float32)
    f = np.zeros(EDGE_M3_DIM, dtype=np.float32)
    if len(A) == 0 or len(B) == 0:
        return f
    ta = [t for t, *_ in ra["atoms"]]
    tb = [t for t, *_ in rb["atoms"]]
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    dmin = float(D.min())

    f[0] = dmin / 6.0
    # continuous distance, which the binned histogram does not provide
    f[1:9] = np.exp(-((dmin - _RBF_MU) ** 2) / (2 * _RBF_SIG ** 2))
    f[9]  = min((D <= 4.0).sum(), 20) / 20.0
    f[10] = min((D <= 4.5).sum(), 20) / 20.0
    f[11] = min((D <= 6.0).sum(), 40) / 40.0

    # which part of each residue makes the contact
    abb, bbb = _bb_mask(ra), _bb_mask(rb)
    close = D <= 4.5
    def cnt(m1, m2):
        return min(int(close[np.ix_(m1, m2)].sum()), 10) / 10.0 if m1.any() and m2.any() else 0.0
    f[12] = cnt(abb, bbb)          # backbone-backbone
    f[13] = cnt(abb, ~bbb)         # receptor backbone - peptide side chain
    f[14] = cnt(~abb, bbb)         # receptor side chain - peptide backbone
    f[15] = cnt(~abb, ~bbb)        # side chain - side chain

    # rule-based interaction chemistry, which atom types alone cannot express
    ca_, cb_ = ra["type"], rb["type"]
    f[16] = 1.0 if ((ca_ in _POS and cb_ in _NEG) or (ca_ in _NEG and cb_ in _POS)) and dmin <= 4.5 else 0.0
    f[17] = 1.0 if ca_ in _HYD and cb_ in _HYD and dmin <= 4.5 else 0.0
    f[18] = 1.0 if ca_ in _ARO and cb_ in _ARO and dmin <= 5.5 else 0.0
    f[19] = 1.0 if ((ca_ in _POS and cb_ in _ARO) or (ca_ in _ARO and cb_ in _POS)) and dmin <= 6.0 else 0.0

    # H-bond-like, on HEAVY atoms. The obvious implementation -- a polar hydrogen
    # against an acceptor -- cannot work on these files: Open Babel writes the added
    # hydrogens as residue groups with no CA, and read_pdbqt drops any residue
    # lacking one, so 413 of 2195 atoms never reach the graph and the peptide chain
    # keeps none at all. Nitrogen/oxygen pairs within 3.5 A is the standard
    # heavy-atom criterion and needs no hydrogens.
    pol = ("N", "NA", "NS", "O", "OA", "OS")
    na_ = np.array([t in pol for t in ta]); nb_ = np.array([t in pol for t in tb])
    hb = int((D[np.ix_(na_, nb_)] <= 3.5).sum()) if na_.any() and nb_.any() else 0
    f[20] = min(hb, 6) / 6.0

    # relative orientation: does each side chain point toward the partner?
    sa, sb = _side_centroid(ra), _side_centroid(rb)
    ua, ub = sa - _ca(ra), sb - _ca(rb)          # residue -> side chain direction
    va, vb = sb - sa, sa - sb                    # toward the partner
    for idx, (u, v) in ((21, (ua, va)), (22, (ub, vb))):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-6 and nv > 1e-6:
            f[idx] = float(np.dot(u / nu, v / nv))
    f[23] = float(np.clip(dmin - 3.0, 0.0, 3.0)) / 3.0

    # metal coordination: a shared ion is a bridge, not two independent contacts
    if len(metals):
        M = np.asarray([(x, y, z) for _, x, y, z in metals], dtype=np.float32)
        da_m = np.sqrt(((A[:, None, :] - M[None, :, :]) ** 2).sum(-1)).min(0)
        db_m = np.sqrt(((B[:, None, :] - M[None, :, :]) ** 2).sum(-1)).min(0)
        near = float(min(da_m.min(), db_m.min()))   # closest approach of either residue
        f[24] = float(np.clip(1.0 - near / 6.0, 0.0, 1.0))
        f[25] = 1.0 if float((np.maximum(da_m, db_m)).min()) <= 3.0 else 0.0
    return f


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
    metals = read_metals(pdbqt)
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
    # M3: an interaction descriptor per interface edge, in the same order as
    # edge_attr (including the duplicated reverse direction pack() appends)
    if inter_pairs:
        m3 = np.stack([_edge_m3_vector(rec[i], pep[j], metals) for i, j in inter_pairs])
        m3 = np.concatenate([m3, m3], 0)
    else:
        m3 = np.zeros((0, EDGE_M3_DIM), dtype=np.float32)
    inter.edge_m3 = torch.tensor(m3, dtype=torch.float)
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
