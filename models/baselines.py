import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
class StandardNodeGCN(nn.Module):
    """炮灰基线：标准的GCN + Energy Score (复刻 LLMGuard 逻辑)"""

    def __init__(self, in_dim=1433, topo_hidden=64, num_classes=7):
        super().__init__()
        self.gcn = GCNConv(in_dim, topo_hidden)
        self.classifier = nn.Linear(topo_hidden, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        return self.classifier(h)

    def get_energy_score(self, x, edge_index, T=1.0):
        logits = self.forward(x, edge_index)
        # Energy Score 计算公式: -T * logsumexp(logits / T)
        # 能量越高 (越趋近于0或正数)，说明越异常
        energy = -T * torch.logsumexp(logits / T, dim=-1)
        return energy