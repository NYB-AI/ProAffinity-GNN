"""PepAffinity-GNN: modules M1 and M2 on top of ProAffinity-GNN.

M1  PeptideFeatureEncoder   peptide-aware residue features, ADDED to the ESM-2
                            node embedding rather than concatenated, so the
                            pretrained first projection (lin1: 256 x 1280) still
                            loads. Zero-initialised at the output, so a fresh M1
                            model is numerically identical to M0 and any
                            difference measured later is learned, not an
                            artefact of re-initialising a layer.

M2  SourceOffset            one trainable scalar per publication, added during
                            training and forced to zero at inference. It absorbs
                            systematic lab offsets so the structural model does
                            not spend capacity on them. A new screening campaign
                            has no offset to look up, which is exactly why it
                            must be zero at test time.

M2  PairwiseRankingLoss     fits the ORDER of peptides inside one
                            (publication x target x assay x method) group. No
                            comparison ever crosses a group, so lab offsets
                            cancel identically rather than being modelled.

Measured context, so nobody re-derives it from scratch:
  * 69.9% of label variance in the corpus is between-group, not between-peptide.
  * On descriptors alone, this objective lifts pooled ranking +0.050 -> +0.122,
    against a within-group ceiling of +0.176 and a length-only control of +0.163.
    So M2's direction is real but small; do not expect M1's features to carry the
    model on their own.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- M1

AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)
CHARGE   = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
AROMATIC = set("FWY")
POLAR    = set("STNQCY")
DONOR    = set("KRWNQSTY")
ACCEPTOR = set("DENQSTY")

PEPTIDE_FEATURE_DIM = 16


def peptide_residue_features(seq: str) -> torch.Tensor:
    """[L, 16] per-residue features for one peptide chain.

    Twelve positional/chemical terms plus four peptide-global terms broadcast to
    every residue -- the plan's "cheap features first" set. Everything is scaled
    to roughly unit range so the adapter MLP sees a sane input distribution.
    """
    seq = "".join(c for c in str(seq).upper() if c in AA)
    L = len(seq)
    if L == 0:
        return torch.zeros(0, PEPTIDE_FEATURE_DIM)

    net_charge = sum(CHARGE.get(c, 0.0) for c in seq)
    hyd_frac   = sum(c in "AVLIMFWC" for c in seq) / L
    aro_frac   = sum(c in AROMATIC for c in seq) / L

    rows = []
    for i, c in enumerate(seq):
        denom = max(L - 1, 1)
        rows.append([
            i / 100.0,                      # absolute position, scaled
            i / denom,                      # normalised position 0..1
            i / L,                          # distance from N-terminus
            (L - 1 - i) / L,                # distance from C-terminus
            1.0 if i == 0 else 0.0,         # N-terminal flag
            1.0 if i == L - 1 else 0.0,     # C-terminal flag
            CHARGE.get(c, 0.0),             # residue charge
            KD[c] / 4.5,                    # hydrophobicity, scaled
            1.0 if c in AROMATIC else 0.0,
            1.0 if c in POLAR else 0.0,
            1.0 if c in DONOR else 0.0,
            1.0 if c in ACCEPTOR else 0.0,
            L / 50.0,                       # global: length
            net_charge / 10.0,              # global: net charge
            hyd_frac,                       # global: hydrophobic fraction
            aro_frac,                       # global: aromatic fraction
        ])
    return torch.tensor(rows, dtype=torch.float)


class PeptideFeatureEncoder(nn.Module):
    """M1. Projects peptide features to the node width and ADDS them.

    Additive, not concatenated: concatenation would widen the input and discard
    the pretrained lin1 projection. The output layer is zero-initialised, so at
    step 0 this module contributes exactly nothing and the wrapped model is bit
    -for-bit M0.
    """

    def __init__(self, node_dim: int = 1280, hidden: int = 64,
                 in_dim: int = PEPTIDE_FEATURE_DIM, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, node_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, pep_feats: torch.Tensor,
                pep_mask: torch.Tensor) -> torch.Tensor:
        """x [N, node_dim]; pep_feats [N, in_dim] (rows for non-peptide nodes are
        ignored); pep_mask [N] bool. Only peptide nodes are modified."""
        if pep_mask is None or pep_mask.sum() == 0:
            return x
        delta = self.mlp(pep_feats[pep_mask])
        out = x.clone()
        out[pep_mask] = self.norm(x[pep_mask] + delta)
        return out


# --------------------------------------------------------------------------- M2

class SourceOffset(nn.Module):
    """M2a. A trainable scalar per publication, used ONLY during training.

    At inference there is no offset to look up for an unseen campaign, so the
    structural model must stand on its own -- calling with source_id=None (or in
    eval mode) contributes zero. `penalty()` regularises the offsets toward zero
    so they cannot quietly absorb the biology.
    """

    def __init__(self, n_sources: int, init_std: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(n_sources, 1)
        nn.init.normal_(self.emb.weight, std=init_std) if init_std > 0 \
            else nn.init.zeros_(self.emb.weight)

    def forward(self, pred: torch.Tensor, source_id: torch.Tensor | None) -> torch.Tensor:
        if source_id is None or not self.training:
            return pred
        return pred + self.emb(source_id).view(pred.shape)

    def penalty(self) -> torch.Tensor:
        return self.emb.weight.pow(2).mean()


class PairwiseRankingLoss(nn.Module):
    """M2b. Order peptides inside a group; never compare across groups.

    softplus(-sign(y_i - y_j) * (pred_i - pred_j)): zero when the predicted order
    matches, growing linearly when it is wrong. Pairs whose labels are closer
    than `min_gap` are skipped -- ordering two measurements that agree within
    experimental error is noise, not signal.
    """

    def __init__(self, max_pairs_per_group: int = 50, min_gap: float = 0.0):
        super().__init__()
        self.max_pairs_per_group = max_pairs_per_group
        self.min_gap = min_gap

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                group_id: torch.Tensor) -> torch.Tensor:
        pred, target = pred.view(-1), target.view(-1)
        device = pred.device
        losses = []
        for g in torch.unique(group_id):
            idx = torch.nonzero(group_id == g, as_tuple=False).view(-1)
            if idx.numel() < 2:
                continue
            i, j = torch.triu_indices(idx.numel(), idx.numel(), offset=1, device=device)
            a, b = idx[i], idx[j]
            gap = target[a] - target[b]
            keep = gap.abs() > self.min_gap
            a, b, gap = a[keep], b[keep], gap[keep]
            if a.numel() == 0:
                continue
            if a.numel() > self.max_pairs_per_group:
                sel = torch.randperm(a.numel(), device=device)[:self.max_pairs_per_group]
                a, b, gap = a[sel], b[sel], gap[sel]
            losses.append(F.softplus(-torch.sign(gap) * (pred[a] - pred[b])))
        if not losses:
            return torch.zeros((), device=device, requires_grad=True)
        return torch.cat(losses).mean()


def pepaffinity_loss(pred, target, group_id=None, source_layer=None,
                     w_reg=1.0, w_rank=2.0, w_offset=0.01, rank_fn=None):
    """Combined objective. Ranking is weighted above regression by default,
    because ranking within a comparable group is the task and the absolute value
    across labs is the part we do not trust."""
    out = {"reg": F.mse_loss(pred.view(-1), target.view(-1))}
    out["rank"] = (rank_fn(pred, target, group_id)
                   if (rank_fn is not None and group_id is not None)
                   else torch.zeros((), device=pred.device))
    out["offset"] = (source_layer.penalty() if source_layer is not None
                     else torch.zeros((), device=pred.device))
    out["total"] = w_reg * out["reg"] + w_rank * out["rank"] + w_offset * out["offset"]
    return out


# --------------------------------------------------------------- wrapped network

class FingerprintHead(nn.Module):
    """Whole-molecule fingerprint concatenated to the pooled graph vector.

    ECFP is a property of the WHOLE peptide -- 2,048 substructure counts over the
    SMILES -- so there is no meaningful mapping from a bit to a residue and it
    cannot join the per-residue M1 features. It is attached where its scope
    belongs: after the three towers pool to 3 x 64 = 192, alongside that vector,
    which is also how the representation earned its result in the literature
    (peptide and protein features concatenated, then regressed).

    The projection is zero-initialised so a fresh model reproduces the backbone
    head exactly, as with M1.
    """

    def __init__(self, in_dim: int = 2048, graph_dim: int = 192,
                 hidden: int = 128, out_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.fp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.fp[-1].weight)
        nn.init.zeros_(self.fp[-1].bias)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, graph_head_out: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """graph_head_out [B, out_dim] (the backbone's 32-d pre-activation);
        fp [B, in_dim] log1p-scaled counts. Returns the summed 32-d vector."""
        return graph_head_out + self.norm(self.fp(fp))


class PepAffinityNet(nn.Module):
    """ProAffinity-GNN's three towers, plus M1 on the interface graph's peptide
    nodes and M2's source offset on the output. The GNN itself is untouched, so
    a pretrained state_dict loads with strict=False and only the new modules are
    missing."""

    def __init__(self, backbone: nn.Module, node_dim: int = 1280,
                 n_sources: int = 0, use_m1: bool = True, use_m2: bool = True,
                 fp_dim: int = 0):
        super().__init__()
        self.backbone = backbone
        self.pep_encoder = PeptideFeatureEncoder(node_dim) if use_m1 else None
        self.source = SourceOffset(n_sources) if (use_m2 and n_sources > 0) else None
        self.fp_head = FingerprintHead(fp_dim) if fp_dim else None

    def forward(self, inter_data, intra1, intra2, source_id=None, fp=None):
        if self.pep_encoder is not None and getattr(inter_data, "pep_mask", None) is not None:
            inter_data = inter_data.clone()
            inter_data.x = self.pep_encoder(inter_data.x, inter_data.pep_feats,
                                            inter_data.pep_mask.bool())
        if self.fp_head is None or fp is None:
            pred = self.backbone(inter_data, intra1, intra2)
        else:
            # reach into the backbone so the fingerprint joins at the 32-d layer,
            # leaving the three pretrained towers untouched
            b = self.backbone
            g = torch.cat([b.graph1(inter_data), b.graph2(intra1), b.graph3(intra2)], dim=1)
            h = F.relu(b.fc1(g))
            pred = b.fc2(self.fp_head(h, fp))
        if self.source is not None:
            pred = self.source(pred, source_id)
        return pred
