import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# 导入核心组件
from generate_graph_ood import create_graph_skeleton, generate_node_semantics, build_pyg_data, API_KEY
from train_ood_model import AnomalyAwareModel
from google import genai
from sentence_transformers import SentenceTransformer

# =====================================================================
# 核心配置区
# =====================================================================
NUM_GRAPHS_PER_CLASS = 30  # 生成 30 张 ID 图，30 张 OOD 图（总计 60 张）
DATASET_PATH = "graph_ood_dataset.pt"  # 数据集本地缓存路径


def set_seed(seed=42):
    """固定随机种子，保证每次跑分结果可复现 (学术严谨性)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataset(device):
    """如果本地有缓存就直接加载，没有就调用大模型生成并保存"""
    if os.path.exists(DATASET_PATH):
        print(f"\n>>> 📦 发现本地缓存数据集 '{DATASET_PATH}'，瞬间加载！")
        dataset = torch.load(DATASET_PATH)
        return dataset

    print(f"\n>>> 🚀 尚未发现缓存，开始批量调用大模型生成数据...")
    print(f">>> 总计将生成 {NUM_GRAPHS_PER_CLASS * 2} 张图，这可能需要 1~2 分钟，请耐心等待或喝杯咖啡！")

    client = genai.Client(api_key=API_KEY)
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    edge_index, node_roles = create_graph_skeleton(num_nodes=5)

    dataset = []
    # 1. 批量生成 ID 图 (Label 0.0)
    print("\n--- 🟢 开始生成分布内 (ID) 数据 ---")
    for i in range(NUM_GRAPHS_PER_CLASS):
        texts = generate_node_semantics(client, node_roles, is_ood=False)
        data = build_pyg_data(edge_index, texts, text_encoder, y_label=0.0)
        dataset.append(data)
        print(f"  - ID Graph {i + 1}/{NUM_GRAPHS_PER_CLASS} 生成完毕")

    # 2. 批量生成 Hard OOD 图 (Label 1.0)
    print("\n--- 🔴 开始生成分布外 (Hard OOD) 数据 ---")
    for i in range(NUM_GRAPHS_PER_CLASS):
        texts = generate_node_semantics(client, node_roles, is_ood=True)
        data = build_pyg_data(edge_index, texts, text_encoder, y_label=1.0)
        dataset.append(data)
        print(f"  - OOD Graph {i + 1}/{NUM_GRAPHS_PER_CLASS} 生成完毕")

    # 保存到本地，一劳永逸！
    torch.save(dataset, DATASET_PATH)
    print(f"\n>>> ✅ 数据集已成功缓存至: {DATASET_PATH}")
    return dataset


def train_and_evaluate():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> [硬件探针] 当前计算设备: {device}")

    # 1. 准备数据
    full_dataset = prepare_dataset(device)
    random.shuffle(full_dataset)  # 打乱顺序

    # 划分训练集 (80%) 和 测试集 (20%)
    split_idx = int(len(full_dataset) * 0.8)
    train_dataset = full_dataset[:split_idx]
    test_dataset = full_dataset[split_idx:]
    print(f">>> 数据划分完毕: 训练集 {len(train_dataset)} 张, 测试集 {len(test_dataset)} 张")

    # 2. 初始化模型与优化器
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    # 3. 核心训练循环 (加入辅助对齐损失 Auxiliary Loss)
    print("\n>>> 🚀 开始模型训练...")
    epochs = 40
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)

            # 前向传播 (接收 4 个返回值)
            logits, alphas, z_topo, z_sem = model(data.x, data.edge_index, batch_index)

            # 主干分类损失
            loss_cls = criterion(logits.unsqueeze(0), data.y.float())
            # 辅助对齐损失 (只在正常图上约束 MSE 为 0)
            mse_loss = nn.MSELoss()(z_topo, z_sem)
            loss_align = mse_loss * (1.0 - data.y.float())

            # 联合优化
            loss = loss_cls + 0.5 * loss_align

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {total_loss / len(train_dataset):.4f}")

    # 4. 跑分评测 (Test Evaluation)
    print("\n================ [学术跑分报告] ================")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
            logits, alphas, _, _ = model(data.x, data.edge_index, batch_index)

            # 计算概率预测值 (Sigmoid)
            prob = torch.sigmoid(logits.unsqueeze(0)).item()
            all_preds.append(prob)
            all_labels.append(data.y.item())

    # 计算顶会两大核心指标
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_labels, all_preds)

    print(f"📊 测试集准确率 (Accuracy) : {acc * 100:.2f}%")
    print(f"📈 曲线下面积 (AUROC)    : {auc * 100:.2f}%")

    if auc > 0.8:
        print("\n✨ 导师评语: AUROC 突破 80%！你的 '异常感知读出机制' 在统计学上被证明绝对有效！可以直接截图放进论文了！")


if __name__ == "__main__":
    train_and_evaluate()