import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
class StandardNodeGCN(nn.Module):
    """Standard GCN baseline with energy-based OOD scoring."""

    def __init__(self, in_dim=1433, topo_hidden=64, num_classes=7):
        super().__init__()
        self.gcn = GCNConv(in_dim, topo_hidden)
        self.classifier = nn.Linear(topo_hidden, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        return self.classifier(h)

    def get_energy_score(self, x, edge_index, T=1.0):
        logits = self.forward(x, edge_index)
        # Higher energy indicates a more anomalous node.
        energy = -T * torch.logsumexp(logits / T, dim=-1)
        return energy