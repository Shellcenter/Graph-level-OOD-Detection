import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DualStreamNodeDetector(nn.Module):
    """
    基于对齐残差的节点级 OOD 探测模型
    """

    def __init__(self, topo_in_dim, sem_in_dim, hidden_dim, z_dim):
        super(DualStreamNodeDetector, self).__init__()

        # 拓扑流: GNN Backbone (Train Process 1 学习目标)
        self.conv1 = GCNConv(topo_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 跨空间投影头: 映射至统一隐藏空间 (Train Process 2 学习目标)
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
        # 提取拓扑特征
        h_topo = F.relu(self.conv1(x_topo, edge_index))
        h_topo = self.conv2(h_topo, edge_index)

        # 投影与 L2 归一化
        z_topo = F.normalize(self.mlp_topo(h_topo), p=2, dim=1)
        z_sem = F.normalize(self.mlp_sem(h_sem), p=2, dim=1)

        return z_topo, z_sem

    def get_anomaly_score(self, z_topo, z_sem):
        # 计算结构与语义的残差得分
        return torch.norm(z_topo - z_sem, p=2, dim=1)