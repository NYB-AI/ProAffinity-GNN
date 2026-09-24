"""Tests for M1/M2. Each one checks a property the modules are supposed to have,
not that they merely run."""
import torch
import pytest

from pepaffinity import (
    AA, PEPTIDE_FEATURE_DIM, peptide_residue_features,
    PeptideFeatureEncoder, SourceOffset, PairwiseRankingLoss, pepaffinity_loss,
)


# ------------------------------------------------------------------ M1 features

def test_feature_shape_and_finiteness():
    f = peptide_residue_features("VTVNPYKWLP")
    assert f.shape == (10, PEPTIDE_FEATURE_DIM)
    assert torch.isfinite(f).all()


def test_terminal_flags_mark_exactly_the_ends():
    f = peptide_residue_features("ACDEF")
    assert f[0, 4] == 1.0 and f[1:, 4].sum() == 0.0      # N-terminal
    assert f[-1, 5] == 1.0 and f[:-1, 5].sum() == 0.0    # C-terminal


def test_position_is_directional():
    """N->C direction must be visible: reversing the sequence must change the
    features. A peptide is not a bag of residues."""
    a = peptide_residue_features("ACDEF")
    b = peptide_residue_features("FEDCA")
    assert not torch.allclose(a, b)


def test_global_features_are_constant_within_a_peptide():
    f = peptide_residue_features("KKEEAA")
    for col in (12, 13, 14, 15):           # length, net charge, hyd frac, aro frac
        assert f[:, col].std() < 1e-6


def test_net_charge_sign():
    assert peptide_residue_features("KKK")[0, 13] > 0
    assert peptide_residue_features("DDD")[0, 13] < 0


def test_empty_and_nonstandard_sequences():
    assert peptide_residue_features("").shape == (0, PEPTIDE_FEATURE_DIM)
    # X and other non-standard codes are dropped, not crashed on
    assert peptide_residue_features("XXAXX").shape == (1, PEPTIDE_FEATURE_DIM)


# ------------------------------------------------------------------ M1 adapter

def test_m1_is_a_noop_at_init():
    """The whole point of zero-initialising the output layer: a fresh M1 model
    must be numerically identical to M0, so any later difference is learned."""
    enc = PeptideFeatureEncoder(node_dim=32)
    x = torch.randn(6, 32)
    feats = torch.randn(6, PEPTIDE_FEATURE_DIM)
    mask = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool)
    out = enc(x, feats, mask)
    assert torch.allclose(out[~mask], x[~mask])           # untouched nodes
    # peptide rows pass through LayerNorm(x + 0); check the delta itself is zero
    assert torch.allclose(enc.mlp(feats[mask]), torch.zeros(2, 32))


def test_m1_only_touches_peptide_nodes():
    enc = PeptideFeatureEncoder(node_dim=8)
    torch.nn.init.normal_(enc.mlp[-1].weight, std=0.5)    # wake it up
    x = torch.randn(5, 8)
    mask = torch.tensor([1, 0, 0, 1, 0], dtype=torch.bool)
    out = enc(x, torch.randn(5, PEPTIDE_FEATURE_DIM), mask)
    assert torch.allclose(out[~mask], x[~mask])
    assert not torch.allclose(out[mask], x[mask])


def test_m1_preserves_width_so_pretrained_lin1_still_loads():
    enc = PeptideFeatureEncoder(node_dim=1280)
    x = torch.randn(3, 1280)
    mask = torch.ones(3, dtype=torch.bool)
    assert enc(x, torch.randn(3, PEPTIDE_FEATURE_DIM), mask).shape == (3, 1280)


def test_m1_gradients_reach_the_adapter():
    """Zero-initialising the OUTPUT layer means that at step 0 only that layer
    receives gradient -- the chain rule multiplies the earlier layers' gradient
    by the (zero) output weights. The adapter is not frozen: once the output
    layer moves off zero, the hidden layer starts learning too. Both halves of
    that are checked here, because "no gradient at init" could otherwise be
    mistaken for a dead module."""
    enc = PeptideFeatureEncoder(node_dim=4)
    x = torch.randn(3, 4)
    feats = torch.randn(3, PEPTIDE_FEATURE_DIM)
    mask = torch.ones(3, dtype=torch.bool)

    # NOT .sum(): the adapter's output goes through LayerNorm, which makes every
    # row zero-mean, so sum(LayerNorm(z)) = n*beta and is INDEPENDENT of z. A
    # .sum() loss therefore has exactly zero gradient to the adapter by
    # construction, and an earlier version of this test only passed on
    # floating-point round-off -- it failed 4 runs in 6. Use a loss that
    # actually depends on the output.
    def loss(e):
        return (e(x, feats, mask) ** 2).sum()

    loss(enc).backward()
    assert enc.mlp[-1].weight.grad.abs().sum() > 0          # output layer learns immediately
    assert enc.mlp[0].weight.grad.abs().sum() == 0          # hidden layer waits one step

    opt = torch.optim.SGD(enc.parameters(), lr=0.1)
    opt.step(); opt.zero_grad()
    loss(enc).backward()
    assert enc.mlp[0].weight.grad.abs().sum() > 0           # and then it learns


# ------------------------------------------------------------------ M2 offset

def test_offset_applies_in_training_and_vanishes_at_inference():
    """A new screening campaign has no offset to look up, so eval() must drop it."""
    so = SourceOffset(4)
    torch.nn.init.normal_(so.emb.weight, std=1.0)
    pred = torch.zeros(3, 1)
    sid = torch.tensor([0, 1, 2])
    so.train()
    assert not torch.allclose(so(pred, sid), pred)
    so.eval()
    assert torch.allclose(so(pred, sid), pred)


def test_offset_is_zero_when_no_source_given():
    so = SourceOffset(4)
    torch.nn.init.normal_(so.emb.weight, std=1.0)
    so.train()
    pred = torch.randn(3, 1)
    assert torch.allclose(so(pred, None), pred)


def test_offset_penalty_pulls_toward_zero():
    so = SourceOffset(3)
    assert so.penalty().item() == pytest.approx(0.0)
    torch.nn.init.constant_(so.emb.weight, 2.0)
    assert so.penalty().item() == pytest.approx(4.0)


# ------------------------------------------------------------------ M2 ranking

def test_ranking_loss_rewards_correct_order():
    fn = PairwiseRankingLoss()
    y = torch.tensor([1.0, 2.0, 3.0])
    g = torch.tensor([0, 0, 0])
    good = fn(torch.tensor([1.0, 2.0, 3.0]) * 5, y, g)
    bad = fn(torch.tensor([3.0, 2.0, 1.0]) * 5, y, g)
    assert good < bad


def test_ranking_loss_never_compares_across_groups():
    """Two groups whose ORDER is internally right but whose levels are opposite:
    the loss must be near zero, because cross-group pairs are never formed."""
    fn = PairwiseRankingLoss()
    y = torch.tensor([1.0, 2.0, 11.0, 12.0])
    g = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([10.0, 20.0, -10.0, -5.0])   # right within, inverted between
    assert fn(pred, y, g).item() < 0.1


def test_ranking_loss_is_invariant_to_a_per_group_offset():
    """Adding a constant to one group's predictions must not change the loss --
    that invariance is exactly what makes lab offsets harmless."""
    fn = PairwiseRankingLoss()
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    g = torch.tensor([0, 0, 1, 1])
    p = torch.tensor([0.5, 1.5, 0.5, 1.5])
    shifted = p + torch.tensor([7.0, 7.0, 0.0, 0.0])
    assert fn(p, y, g).item() == pytest.approx(fn(shifted, y, g).item(), abs=1e-6)


def test_ranking_loss_zero_when_no_pairs_exist():
    fn = PairwiseRankingLoss()
    out = fn(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor([0, 1]))
    assert out.item() == 0.0
    assert out.requires_grad


def test_min_gap_skips_pairs_inside_experimental_error():
    fn = PairwiseRankingLoss(min_gap=1.0)
    y = torch.tensor([1.0, 1.2])          # closer than the gap
    assert fn(torch.tensor([5.0, -5.0]), y, torch.tensor([0, 0])).item() == 0.0


def test_ranking_loss_caps_pairs_per_group():
    fn = PairwiseRankingLoss(max_pairs_per_group=5)
    n = 40
    pred = torch.randn(n, requires_grad=True)
    out = fn(pred, torch.arange(n, dtype=torch.float), torch.zeros(n, dtype=torch.long))
    assert torch.isfinite(out) and out.requires_grad
    out.backward()
    # capped at 5 pairs, so at most 10 of the 40 peptides can receive gradient
    assert 0 < int((pred.grad != 0).sum()) <= 10


def test_ranking_loss_backpropagates():
    fn = PairwiseRankingLoss()
    p = torch.randn(4, requires_grad=True)
    fn(p, torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0, 0, 0, 0])).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


# ------------------------------------------------------------------ combined

def test_combined_loss_components_and_weighting():
    rank = PairwiseRankingLoss()
    so = SourceOffset(2)
    torch.nn.init.constant_(so.emb.weight, 1.0)
    pred = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.0, 2.0, 3.0])
    g = torch.tensor([0, 0, 0])
    out = pepaffinity_loss(pred, tgt, g, so, w_reg=1.0, w_rank=2.0,
                           w_offset=0.01, rank_fn=rank)
    assert set(out) == {"reg", "rank", "offset", "total"}
    assert out["reg"].item() == pytest.approx(0.0)
    assert out["offset"].item() == pytest.approx(1.0)
    expected = out["reg"] + 2.0 * out["rank"] + 0.01 * out["offset"]
    assert out["total"].item() == pytest.approx(expected.item(), abs=1e-6)


def test_combined_loss_without_groups_or_source():
    out = pepaffinity_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert out["rank"].item() == 0.0 and out["offset"].item() == 0.0
    assert out["total"].item() == pytest.approx(0.5)
