import torch
import torch.nn as nn
import torch.optim as optim

# 从你之前写的两个文件里导入核心组件！
# (注意：确保 generate_graph_ood.py 和 train_ood_model.py 在同一个文件夹下)
from generate_graph_ood import create_graph_skeleton, generate_node_semantics, build_pyg_data, API_KEY
from train_ood_model import AnomalyAwareModel
from google import genai
from sentence_transformers import SentenceTransformer


def train_model():
    print(">>> 1. 初始化引擎与环境...")
    # 探测是否有 GPU，统一设备空间
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[硬件探针] 当前正在使用的计算设备: {device}")
    client = genai.Client(api_key=API_KEY)
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 实例化我们的模型
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32)

    # 定义优化器 (Adam) 和 损失函数 (BCEWithLogitsLoss 包含 Sigmoid)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    print("\n>>> 2. 准备极简训练集 (1张ID图, 1张OOD图) 作为概念验证...")
    # 为了防止 API 超出限额，我们先只用极小的数据证明模型能收敛 (Overfitting test)
    edge_index, node_roles = create_graph_skeleton(num_nodes=5)

    # 生成 1 张正常图
    id_texts = generate_node_semantics(client, node_roles, is_ood=False)
    id_data = build_pyg_data(edge_index, id_texts, text_encoder, y_label=0.0)  # 注意标签是 0.0 Float

    # 生成 1 张异常图
    ood_texts = generate_node_semantics(client, node_roles, is_ood=True)
    ood_data = build_pyg_data(edge_index, ood_texts, text_encoder, y_label=1.0)  # 注意标签是 1.0 Float

    dataset = [id_data, ood_data]

    print("\n>>> 3. 开始训练 (Training Loop) ...")
    epochs = 30  # 跑 30 轮

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # 遍历我们的数据集
        for data in dataset:
            optimizer.zero_grad()  # 梯度清零

            # 因为是逐张图训练，所有节点的 batch 归属都是 0
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long)

            # 前向传播
            logits, alphas = model(data.x, data.edge_index, batch_index)

            # 计算 Loss (预测值 vs 真实标签)
            loss = criterion(logits, data.y.float())

            # 反向传播与权重更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 每 5 轮打印一次监控信息
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {total_loss / len(dataset):.4f}")

    print("\n>>> 4. 训练结束，学术验收！")
    # 让我们看看训练后的模型，面对 OOD 图时，会不会把注意力集中在异常节点上
    model.eval()
    with torch.no_grad():
        batch_index = torch.zeros(ood_data.x.size(0), dtype=torch.long)
        logits, alphas = model(ood_data.x, ood_data.edge_index, batch_index)

        print("\n[验收结果] 面对 Hard OOD 图：")
        print(f"- 模型预测 Logits: {logits.item():.4f} (越大越代表是OOD)")
        print("- 模型分配的注意力权重 (Alphas):")
        for i, alpha in enumerate(alphas.squeeze().numpy()):
            role = "中心正常" if i == 0 else "边缘变异"
            print(f"  Node {i} ({role}): {alpha:.4f}")


if __name__ == "__main__":
    train_model()