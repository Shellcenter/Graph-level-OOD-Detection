import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def generate_node_level_ood_data(dataset_name='Cora', ood_ratio=0.1):
    # 加载基准图数据
    dataset = Planetoid(root='./data', name=dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    num_nodes = data.num_nodes

    # 初始化双流输入特征
    # 实际工程中，h_sem 应替换为由 LLM (如 SentenceBERT) 提取的真实文本向量
    data.x_topo = data.x.clone()
    data.h_sem = data.x.clone()

    # 生成 OOD 节点索引
    num_ood = int(num_nodes * ood_ratio)
    ood_indices = torch.randperm(num_nodes)[:num_ood]

    # 设置 OOD 标签 (0: ID, 1: OOD)
    data.y_ood = torch.zeros(num_nodes, dtype=torch.long)
    data.y_ood[ood_indices] = 1

    # 模拟真实 OOD 的“结构-语义背离”特性：
    # 保持图拓扑不变，仅打乱其语义特征 h_sem
    shuffle_indices = ood_indices[torch.randperm(num_ood)]
    data.h_sem[ood_indices] = data.h_sem[shuffle_indices]

    # 划分训练集 (仅使用部分正常 ID 节点进行对齐法则学习)
    id_indices = torch.where(data.y_ood == 0)[0]
    num_train = int(len(id_indices) * 0.6)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[id_indices[:num_train]] = True
    data.test_mask = torch.ones(num_nodes, dtype=torch.bool)

    print(f"Dataset loaded. Total nodes: {num_nodes}, Edges: {data.edge_index.size(1)}")
    print(f"Injected {num_ood} OOD nodes ({ood_ratio * 100}%).")

    return data