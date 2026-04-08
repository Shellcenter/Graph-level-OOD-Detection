import sys
import os

# 🚀 导师的强行引路代码：绝对防弹的路径注册
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# 从我们刚建好的专属武器库和工具箱中导入！
from models.anomaly_aware import NodeAnomalyAwareModel
from utils.metrics import compute_metrics

# 代理设置 (确保能顺畅下载 Cora)
PROXY_PORT = "9674"
os.environ['http_proxy'] = f'http://127.0.0.1:{PROXY_PORT}'
os.environ['https_proxy'] = f'http://127.0.0.1:{PROXY_PORT}'


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    # 1. 严肃的参数解析 (Argparse)
    parser = argparse.ArgumentParser(description="Graph OOD Detection Benchmark")
    parser.add_argument("--dataset", type=str, default="Cora", help="数据集名称")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--ood_ratio", type=float, default=0.2, help="OOD变异比例")
    args = parser.parse_args()

    # 2. 专业的日志输出系统
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout  # 👈 核心修复：强行把日志塞进白色的普通管道！
    )
    logging.info(f"🚀 正在启动实验 | 数据集: {args.dataset} | 学习率: {args.lr} | 轮数: {args.epochs}")

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"💻 硬件探针激活 | 当前计算设备: {device}")

    # 3. 数据加载与高危变异注入
    logging.info("📂 正在拉取真实基准数据集...")
    dataset = Planetoid(root='./data', name=args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    logging.info("💉 正在向测试集注入隐蔽 OOD 变异节点...")
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    ood_num = int(len(test_idx) * args.ood_ratio)
    ood_idx = test_idx[:ood_num]
    id_test_idx = test_idx[ood_num:]

    # 篡改变异逻辑
    tampered_x = data.x.clone()
    shuffled_indices = torch.randperm(ood_idx.size(0))
    tampered_x[ood_idx] = data.x[ood_idx[shuffled_indices]]
    data.x_tampered = tampered_x

    # 4. 初始化主角模型
    logging.info("🏗️ 正在初始化 Anomaly-Aware 核心架构...")
    our_model = NodeAnomalyAwareModel(in_dim=dataset.num_features, num_classes=dataset.num_classes).to(device)
    our_opt = optim.Adam(our_model.parameters(), lr=args.lr)

    # 5. 训练循环 (自监督联合优化)
    logging.info("🔥 开始全图无监督对齐训练...")
    for epoch in range(1, args.epochs + 1):
        our_model.train()
        our_opt.zero_grad()
        logits, scores, z_topo, z_sem = our_model(data.x, data.edge_index)

        loss_cls = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss_align = nn.MSELoss()(z_topo, z_sem)  # 全图物理空间对齐

        loss = loss_cls + 1.0 * loss_align
        loss.backward()
        our_opt.step()

        if epoch % 20 == 0 or epoch == 1:
            logging.info(f"   Epoch [{epoch:03d}/{args.epochs}] | Total Loss: {loss.item():.4f}")

    # 6. 评估与跑分
    logging.info("🧪 正在测试集上执行降维打击验证...")
    our_model.eval()
    with torch.no_grad():
        _, our_scores, _, _ = our_model(data.x_tampered, data.edge_index)
        id_scores_our = our_scores[id_test_idx].cpu().numpy()
        ood_scores_our = our_scores[ood_idx].cpu().numpy()

    labels_our = np.concatenate([np.zeros(len(id_scores_our)), np.ones(len(ood_scores_our))])
    scores_our = np.concatenate([id_scores_our, ood_scores_our])

    # 调用 utils/metrics.py 中的专业计分器
    our_auc, our_fpr95 = compute_metrics(labels_our, scores_our)

    logging.info("=" * 50)
    logging.info(f"🏆 主角最终成绩 -> AUROC: {our_auc * 100:.2f}% | FPR95: {our_fpr95 * 100:.2f}%")
    logging.info("=" * 50)
    logging.info("✅ 实验结束，所有进程已安全退出。")


if __name__ == "__main__":
    main()