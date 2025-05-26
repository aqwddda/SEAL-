import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScoreGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
        )
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):

        h = x

        for i in range(len(self.convs) - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = self.drops[i](h)

        # 最后一层没有加bn和dropout
        out = self.convs[-1](h, edge_index)

        return out


class DotProductPredictor(nn.Module):
    def forward(self, out, edge_label_index):
        src = edge_label_index[0]
        dst = edge_label_index[1]
        return (out[src] * out[dst]).sum(dim=-1)


class HadamardMLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, out, edge_label_index):
        src = edge_label_index[0]
        dst = edge_label_index[1]
        # Element-wise product (Hadamard)
        dot = out[src] * out[dst]
        return self.mlp(dot).view(-1)


class ConcatMLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, out, edge_label_index):
        src = edge_label_index[0]
        dst = edge_label_index[1]
        # Concatenate embeddings
        h = torch.cat([out[src], out[dst]], dim=-1)
        return self.mlp(h).view(-1)
