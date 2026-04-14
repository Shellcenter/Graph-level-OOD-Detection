import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
# 🛑 删除了所有不需要的冗余 import，彻底消灭报错源头！

class NodeAnomalyAwareModel(nn.Module):
    """基于拓扑-语义解耦的节点异常感知器 (联合优化防塌陷)"""
    def __init__(self, in_dim=1433, topo_hidden=64, align_dim=32, num_classes=7):
        super().__init__()
        self.gcn = GCNConv(in_dim, topo_hidden)
        self.proj_topo = nn.Linear(topo_hidden, align_dim)
        self.proj_sem = nn.Linear(in_dim, align_dim)
        # 必须加一个分类器，作为“锚点”防止特征塌陷
        self.classifier = nn.Linear(align_dim, num_classes)

    def forward(self, x, edge_index):
        h_topo = self.gcn(x, edge_index).relu()
        z_topo = self.proj_topo(h_topo)
        z_sem = self.proj_sem(x)

        # 分类逻辑：强迫 z_topo 学到有用的结构+语义融合信息
        logits = self.classifier(z_topo)

        # 异常得分 = 空间物理距离
        anomaly_scores = torch.norm(z_topo - z_sem, p=2, dim=-1)

        return logits, anomaly_scores, z_topo, z_sem