"""The ProAffinity-GNN architecture, importable without side effects.

ProAffinity_GNN.py defines these classes but also loads a dataset and starts
training at module scope, so it cannot be imported. The definitions here are
byte-equivalent to that file's; keeping them separate lets the model be reused
for evaluation, fine-tuning and tests.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import AttentiveFP


class AttentiveFPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim,
                 num_layers, num_timesteps, dropout):
        super().__init__()
        self.model = AttentiveFP(in_channels, hidden_channels, out_channels,
                                 edge_dim, num_layers, num_timesteps, dropout)

    def forward(self, data):
        return self.model(data.x, data.edge_index, data.edge_attr, data.batch)


class GraphNetwork(torch.nn.Module):
    """Three independent AttentiveFP towers -- interface, receptor-internal,
    peptide-internal -- concatenated (3 x 64 = 192) into a two-layer head."""

    def __init__(self, in_channels=1280, hidden_channels=256, out_channels=64,
                 edge_dim=280, num_layers=3, num_timesteps=2, dropout=0.5,
                 linear_out1=32, linear_out2=1):
        super().__init__()
        self.graph1 = AttentiveFPModel(in_channels, hidden_channels, out_channels,
                                       edge_dim, num_layers, num_timesteps, dropout)
        self.graph2 = AttentiveFPModel(in_channels, hidden_channels, out_channels,
                                       edge_dim, num_layers, num_timesteps, dropout)
        self.graph3 = AttentiveFPModel(in_channels, hidden_channels, out_channels,
                                       edge_dim, num_layers, num_timesteps, dropout)
        self.fc1 = torch.nn.Linear(out_channels * 3, linear_out1)
        self.fc2 = torch.nn.Linear(linear_out1, linear_out2)

    def forward(self, inter_data, intra_data1, intra_data2):
        x = torch.cat([self.graph1(inter_data),
                       self.graph2(intra_data1),
                       self.graph3(intra_data2)], dim=1)
        return self.fc2(F.relu(self.fc1(x)))


def load_pretrained(model, path="model/model.pkl", strict=True):
    """Load the released checkpoint across the PyG GATConv API change.

    The checkpoint was trained when GATConv kept separate `lin_src` / `lin_dst`
    projections. PyG >= 2.5 uses a single `lin` when in_channels is an int. The
    two were TIED in the original (verified: lin_src == lin_dst for all 9
    convolutions), so collapsing them is lossless, not an approximation.
    """
    import torch
    sd = torch.load(path, map_location="cpu")
    out, dropped = {}, []
    for k, v in sd.items():
        if k.endswith("lin_dst.weight"):
            src = sd.get(k.replace("lin_dst", "lin_src"))
            if src is not None and not torch.equal(src, v):
                raise ValueError(
                    f"{k}: lin_src and lin_dst differ, so they cannot be collapsed "
                    "into a single `lin`. Pin an older torch_geometric instead.")
            dropped.append(k)
            continue
        out[k.replace("lin_src.weight", "lin.weight")] = v
    missing, unexpected = model.load_state_dict(out, strict=False)
    if strict and (missing or unexpected):
        raise ValueError(f"checkpoint mismatch: missing={list(missing)} "
                         f"unexpected={list(unexpected)}")
    return model
