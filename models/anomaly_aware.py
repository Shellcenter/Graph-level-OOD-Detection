import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DualStreamOODDetector(nn.Module):
    def __init__(self, topo_in_dim, sem_in_dim, hidden_dim, z_dim):
        super(DualStreamOODDetector, self).__init__()

        # 拓扑流: GNN Backbone
        self.conv1 = GCNConv(topo_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 语义流: 假设外部已提供 LLM 特征，直接映射

        # 跨空间投影头 (Phase 1.5)
        self.mlp_topo = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

        self.mlp_sem = nn.Sequential(
            nn.Linear(sem_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x_topo, edge_index, h_sem):
        # 提取拓扑特征 (受网络结构平滑影响)
        h_topo = F.relu(self.conv1(x_topo, edge_index))
        h_topo = self.conv2(h_topo, edge_index)

        # 映射至统一隐藏空间并进行 L2 归一化
        z_topo = F.normalize(self.mlp_topo(h_topo), p=2, dim=1)
        z_sem = F.normalize(self.mlp_sem(h_sem), p=2, dim=1)

        return z_topo, z_sem

    def compute_anomaly_score(self, z_topo, z_sem):
        # 计算结构与语义的对齐残差 (||ΔZ||_2)
        residual = z_topo - z_sem
        return torch.norm(residual, p=2, dim=1)