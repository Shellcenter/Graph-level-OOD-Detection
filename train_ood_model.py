import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax


# 图级异常感知读出模型。

class AnomalyAwareModel(nn.Module):
    def __init__(self, sem_dim=384, topo_hidden=64, align_dim=32):
        super().__init__()

        # 拓扑编码器。
        self.gcn = GCNConv(sem_dim, topo_hidden)

        # 模态对齐层。
        self.proj_topo = nn.Linear(topo_hidden, align_dim)
        self.proj_sem = nn.Linear(sem_dim, align_dim)

        # 差异评分器，输入维度为 32 + 32 + 32 = 96。
        self.scorer = nn.Sequential(
            nn.Linear(align_dim * 3, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

        # 图级分类器。
        self.classifier = nn.Linear(topo_hidden, 1)

    def forward(self, x_sem, edge_index, batch_index):
        """
        参数：
            x_sem: 节点语义特征，形状为 [N, 384]。
            edge_index: 图的 COO 格式连边信息。
            batch_index: 每个节点所属图的批索引。
        """
        # 第一步：提取包含拓扑信息的节点表示。
        h_topo = self.gcn(x_sem, edge_index).relu()

        # 第二步：对齐拓扑表示与语义表示。
        z_topo = self.proj_topo(h_topo)
        z_sem = self.proj_sem(x_sem)

        # 第三步：在节点级别计算表示差异分数。
        diff = torch.abs(z_topo - z_sem)
        concat_feat = torch.cat([z_topo, z_sem, diff], dim=-1)
        s_i = self.scorer(concat_feat)

        # 第四步：结合节点注意力进行图级聚合。
        alpha = softmax(s_i, batch_index)
        weighted_h = h_topo * alpha
        z_graph = global_add_pool(weighted_h, batch_index)

        # 输出图级分类原始分数。
        logits = self.classifier(z_graph)
        return logits.squeeze(), alpha, z_topo, z_sem


# 最小前向传播连通性测试。

if __name__ == "__main__":
    print("Initializing Anomaly-Aware Readout model...")
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32)

    # 构造一张小型合成图用于快速前向测试。
    dummy_x = torch.randn(5, 384)
    dummy_edge_index = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4],
                                     [1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.long)
    dummy_batch = torch.zeros(5, dtype=torch.long)

    # 前向传播测试。
    logits, alphas, _, _ = model(dummy_x, dummy_edge_index, dummy_batch)

    print("\nForward pass completed successfully.")
    print(f"Graph-level logit: {logits.item():.4f}")
    print(f"Attention weights (alphas):\n{alphas.detach().numpy()}")
    print("\nThe model is ready for integration with the generated graph dataset.")


# 用于对比的标准 GCN 基线模型。

class StandardGCN(nn.Module):
    def __init__(self, sem_dim=384, topo_hidden=64):
        super().__init__()
        self.gcn = GCNConv(sem_dim, topo_hidden)
        # 不包含对齐项与注意力加权的简单分类器。
        self.classifier = nn.Linear(topo_hidden, 1)

    def forward(self, x_sem, edge_index, batch_index):
        # 第一步：提取节点特征。
        h = self.gcn(x_sem, edge_index).relu()
        # 第二步：全局池化得到图表示。
        z_graph = global_add_pool(h, batch_index)
        # 第三步：进行图级分类。
        logits = self.classifier(z_graph)
        return logits.squeeze()