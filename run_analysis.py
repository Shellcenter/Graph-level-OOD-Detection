import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

from train_ood_model import AnomalyAwareModel, StandardGCN

# 强制使用英文字体，避免画图时中文字体乱码
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

DATASET_PATH = "graph_ood_dataset.pt"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(model_name, model, train_dataset, test_dataset, device, is_anomaly_aware=False):
    """通用的模型训练与评估函数"""
    print(f"\n================ [ Training {model_name} ] ================")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 40

    for epoch in range(epochs):
        model.train()
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)

            if is_anomaly_aware:
                logits, alphas, z_topo, z_sem = model(data.x, data.edge_index, batch_index)
                loss_cls = criterion(logits.unsqueeze(0), data.y.float())
                mse_loss = nn.MSELoss()(z_topo, z_sem)
                loss_align = mse_loss * (1.0 - data.y.float())
                loss = loss_cls + 0.5 * loss_align
            else:
                # 基线模型只有 Logits 输出
                logits = model(data.x, data.edge_index, batch_index)
                loss = criterion(logits.unsqueeze(0), data.y.float())

            loss.backward()
            optimizer.step()

    # 跑分
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
            if is_anomaly_aware:
                logits, _, _, _ = model(data.x, data.edge_index, batch_index)
            else:
                logits = model(data.x, data.edge_index, batch_index)

            prob = torch.sigmoid(logits.unsqueeze(0)).item()
            all_preds.append(prob)
            all_labels.append(data.y.item())

    auc = roc_auc_score(all_labels, all_preds)
    print(f"[{model_name}] Final test AUROC: {auc * 100:.2f}%")
    return model


def plot_attention_heatmap(model, dataset, device):
    """高逼格论文插图生成器：画出注意力的降维打击"""
    print("\n>>> Rendering attention heatmap...")
    model.eval()

    # 找出一张 Hard OOD 图 (y == 1)
    ood_data = next(data for data in dataset if data.y.item() == 1.0)
    ood_data = ood_data.to(device)
    batch_index = torch.zeros(ood_data.x.size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        _, alphas, _, _ = model(ood_data.x, ood_data.edge_index, batch_index)
        alphas = alphas.squeeze().cpu().numpy()

    # 构建 NetworkX 拓扑图
    G = nx.star_graph(4)  # 5个节点的星型图

    # 画图设置
    plt.figure(figsize=(8, 6))
    plt.title("Anomaly-Aware Readout: Attention Heatmap on Hard OOD Graph\n(Node 1 is the camouflage anomaly)",
              fontsize=14, pad=15)

    # 计算布局 (中心点在中间，其余一圈)
    pos = nx.spring_layout(G, seed=42)

    # 根据 Alpha 权重决定节点颜色 (越红代表注意力越高)
    cmap = plt.cm.Reds

    # 画边
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5, edge_color="gray")

    # 画节点
    nodes = nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=alphas, cmap=cmap, vmin=0, vmax=1.0,
                                   edgecolors='black', linewidths=2)

    # 写上 Node ID
    nx.draw_networkx_labels(G, pos, font_size=16, font_family="sans-serif", font_color="white")

    # 添加颜色条 (Colorbar) 告诉审稿人红色代表高 Attention
    cbar = plt.colorbar(nodes)
    cbar.set_label('Attention Weight (Alpha)', fontsize=12)

    # 保存图片
    plt.savefig("attention_heatmap.png", dpi=300, bbox_inches='tight')
    print("Figure saved to 'attention_heatmap.png'.")
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载你之前缓存的完美数据集 (1秒极速加载)
    full_dataset = torch.load(DATASET_PATH)
    random.shuffle(full_dataset)
    split_idx = int(len(full_dataset) * 0.8)
    train_dataset = full_dataset[:split_idx]
    test_dataset = full_dataset[split_idx:]

    # 1. 训练基线模型 (炮灰)
    model_base = StandardGCN().to(device)
    evaluate_model("Standard GCN (Baseline)", model_base, train_dataset, test_dataset, device, is_anomaly_aware=False)

    # 2. 训练你的创新模型 (主角)
    model_ours = AnomalyAwareModel().to(device)
    model_ours = evaluate_model("Anomaly-Aware Model (Ours)", model_ours, train_dataset, test_dataset, device,
                                is_anomaly_aware=True)

    # 3. 终极绘图
    plot_attention_heatmap(model_ours, test_dataset, device)