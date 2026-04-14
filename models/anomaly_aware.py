import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class NodeAnomalyAwareModel(nn.Module):
    """带联合分类与对齐机制的节点级异常感知模型。"""
    def __init__(self, in_dim=1433, topo_hidden=64, align_dim=32, num_classes=7):
        super().__init__()
        self.gcn = GCNConv(in_dim, topo_hidden)
        self.proj_topo = nn.Linear(topo_hidden, align_dim)
        self.proj_sem = nn.Linear(in_dim, align_dim)
        # 分类头用于为表示学习提供监督锚点。
        self.classifier = nn.Linear(align_dim, num_classes)

    def forward(self, x, edge_index):
        h_topo = self.gcn(x, edge_index).relu()
        z_topo = self.proj_topo(h_topo)
        z_sem = self.proj_sem(x)

        # 使用对齐后的拓扑特征进行节点分类。
        logits = self.classifier(z_topo)

        # 使用拓扑表示与语义表示之间的差异作为异常分数。
        anomaly_scores = torch.norm(z_topo - z_sem, p=2, dim=-1)

        return logits, anomaly_scores, z_topo, z_sem