import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from env_config import configure_proxy, get_api_key
from generate_graph_ood import create_graph_skeleton, generate_node_semantics, build_pyg_data
from train_ood_model import AnomalyAwareModel
from google import genai
from sentence_transformers import SentenceTransformer


# 实验配置。

NUM_GRAPHS_PER_CLASS = 30  # 总样本数为 2 * NUM_GRAPHS_PER_CLASS。
DATASET_PATH = "graph_ood_dataset.pt"


def set_seed(seed=42):
    """设置随机种子，便于结果复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataset(device):

    if os.path.exists(DATASET_PATH):
        print(f"\nFound cached dataset at '{DATASET_PATH}'. Loading from disk...")
        dataset = torch.load(DATASET_PATH)
        return dataset

    print("\nNo cached dataset found. Generating samples with the language model...")
    print(f"Total graphs to generate: {NUM_GRAPHS_PER_CLASS * 2}. This may take 1-2 minutes.")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    client = genai.Client(api_key=get_api_key())
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    edge_index, node_roles = create_graph_skeleton(num_nodes=5)

    dataset = []
    # 生成分布内图样本。
    print("\n--- Generating in-distribution (ID) graphs ---")
    for i in range(NUM_GRAPHS_PER_CLASS):
        texts = generate_node_semantics(client, node_roles, is_ood=False)
        data = build_pyg_data(edge_index, texts, text_encoder, y_label=0.0)
        dataset.append(data)
        print(f"  - ID graph {i + 1}/{NUM_GRAPHS_PER_CLASS} generated")

    # 生成分布外图样本。
    print("\n--- Generating out-of-distribution (OOD) graphs ---")
    for i in range(NUM_GRAPHS_PER_CLASS):
        texts = generate_node_semantics(client, node_roles, is_ood=True)
        data = build_pyg_data(edge_index, texts, text_encoder, y_label=1.0)
        dataset.append(data)
        print(f"  - OOD graph {i + 1}/{NUM_GRAPHS_PER_CLASS} generated")

    # 将数据集缓存到本地，便于后续重复实验。
    torch.save(dataset, DATASET_PATH)
    print(f"\nDataset cached at: {DATASET_PATH}")
    return dataset


def train_and_evaluate():
    set_seed(42)
    proxy_port = configure_proxy()
    if proxy_port:
        print(f"Proxy configured on port: {proxy_port}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}")

    # 准备数据集。
    full_dataset = prepare_dataset(device)
    random.shuffle(full_dataset)

    # 划分训练集和测试集。
    split_idx = int(len(full_dataset) * 0.8)
    train_dataset = full_dataset[:split_idx]
    test_dataset = full_dataset[split_idx:]
    print(f"Dataset split completed: {len(train_dataset)} training graphs, {len(test_dataset)} test graphs")

    # 初始化模型、优化器和损失函数。
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    # 使用分类损失和对齐损失进行训练。
    print("\nStarting model training...")
    epochs = 40
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)

            # 前向传播。
            logits, alphas, z_topo, z_sem = model(data.x, data.edge_index, batch_index)

            # 主分类损失。
            loss_cls = criterion(logits.unsqueeze(0), data.y.float())
            # 仅在 ID 图样本上施加对齐损失。
            mse_loss = nn.MSELoss()(z_topo, z_sem)
            loss_align = mse_loss * (1.0 - data.y.float())

            # 联合优化。
            loss = loss_cls + 0.5 * loss_align

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {total_loss / len(train_dataset):.4f}")

    # 在测试集上评估模型。
    print("\nEvaluation results")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
            logits, alphas, _, _ = model(data.x, data.edge_index, batch_index)

            # 将原始分类分数转换为概率。
            prob = torch.sigmoid(logits.unsqueeze(0)).item()
            all_preds.append(prob)
            all_labels.append(data.y.item())

    # 计算汇总指标。
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUROC: {auc * 100:.2f}%")

    if auc > 0.8:
        print("\nResult summary: AUROC exceeded 80% on the current split.")


if __name__ == "__main__":
    train_and_evaluate()