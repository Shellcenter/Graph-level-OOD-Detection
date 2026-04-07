import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax


# =====================================================================
# 核心架构：异常感知读出网络 (Anomaly-Aware Readout Network)
# =====================================================================
class AnomalyAwareModel(nn.Module):
    def __init__(self, sem_dim=384, topo_hidden=64, align_dim=32):
        super().__init__()

        # 1. 拓扑特征提取骨干 (GNN Backbone)
        self.gcn = GCNConv(sem_dim, topo_hidden)

        # 2. 模态对齐层 (Alignment MLPs)
        self.proj_topo = nn.Linear(topo_hidden, align_dim)
        self.proj_sem = nn.Linear(sem_dim, align_dim)

        # 3. 差异评分器 (Mismatch Scorer)
        # 输入维度: topo对齐(32) + sem对齐(32) + 绝对残差(32) = 96
        self.scorer = nn.Sequential(
            nn.Linear(align_dim * 3, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)  # 输出每个节点的原始异常得分
        )

        # 4. 图级 OOD 分类器 (Graph-level Classifier)
        self.classifier = nn.Linear(topo_hidden, 1)

    def forward(self, x_sem, edge_index, batch_index):
        """
        x_sem: 节点的语义特征 [N, 384] (由大模型+SentenceTransformer提取)
        edge_index: 图的拓扑连边
        batch_index: 标明每个节点属于哪一张图
        """
        # 第一步：提取拓扑物理特征
        h_topo = self.gcn(x_sem, edge_index).relu()  # [N, 64]

        # 第二步：空间对齐
        z_topo = self.proj_topo(h_topo)  # [N, 32]
        z_sem = self.proj_sem(x_sem)  # [N, 32]

        # 第三步：差异计算与评分 (核心创新点)
        diff = torch.abs(z_topo - z_sem)
        concat_feat = torch.cat([z_topo, z_sem, diff], dim=-1)  # [N, 96]
        s_i = self.scorer(concat_feat)  # [N, 1]

        # 第四步：异常信号放大与全局池化
        alpha = softmax(s_i, batch_index)  # 图内 Softmax 归一化 (放大器)
        weighted_h = h_topo * alpha  # 异常节点特征被强制放大
        z_graph = global_add_pool(weighted_h, batch_index)  # 加权聚合成图向量 [Batch_size, 64]

        # 输出分类 logits
        logits = self.classifier(z_graph)
        return logits.squeeze(), alpha, z_topo, z_sem  # 🚀 吐出中间特征用于辅助约束


# =====================================================================
# 探针测试：连通性验证
# =====================================================================
if __name__ == "__main__":
    print(">>> 正在初始化 Anomaly-Aware Readout 模型...")
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32)

    # 伪造一张 5 节点的图进行连通性测试 (模拟刚才跑出的数据)
    dummy_x = torch.randn(5, 384)  # 5个节点的文本向量
    dummy_edge_index = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4],
                                     [1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.long)
    dummy_batch = torch.zeros(5, dtype=torch.long)  # 这 5 个节点都属于第 0 张图

    # 前向传播测试
    logits, alphas = model(dummy_x, dummy_edge_index, dummy_batch)

    print("\n✅ 模型前向传播成功！")
    print(f"图级预测输出 (Logits): {logits.item():.4f}")
    print(f"注意力权重分布 (Alphas): \n{alphas.detach().numpy()}")
    print("\n准备好将 generate_graph_ood.py 的真实数据接入进来了吗？")