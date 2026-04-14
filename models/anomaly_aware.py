import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class NodeAnomalyAwareModel(nn.Module):
    """Node-level anomaly-aware model with joint classification and alignment."""
    def __init__(self, in_dim=1433, topo_hidden=64, align_dim=32, num_classes=7):
        super().__init__()
        self.gcn = GCNConv(in_dim, topo_hidden)
        self.proj_topo = nn.Linear(topo_hidden, align_dim)
        self.proj_sem = nn.Linear(in_dim, align_dim)
        # The classification head anchors representation learning.
        self.classifier = nn.Linear(align_dim, num_classes)

    def forward(self, x, edge_index):
        h_topo = self.gcn(x, edge_index).relu()
        z_topo = self.proj_topo(h_topo)
        z_sem = self.proj_sem(x)

        # Use aligned topology features for node classification.
        logits = self.classifier(z_topo)

        # Use the representation gap as the anomaly score.
        anomaly_scores = torch.norm(z_topo - z_sem, p=2, dim=-1)

        return logits, anomaly_scores, z_topo, z_sem