import torch
from models.anomaly_aware import DualStreamOODDetector
from generate_graph_ood import generate_node_level_ood_data
from train_ood_model import train_epoch, test_epoch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Runtime device: {device}")

    # 1. 准备数据
    data = generate_node_level_ood_data('Cora', ood_ratio=0.15).to(device)

    # 2. 初始化模型与优化器
    topo_in_dim = data.x_topo.size(1)
    sem_in_dim = data.h_sem.size(1)
    hidden_dim = 128
    z_dim = 64

    model = DualStreamOODDetector(topo_in_dim, sem_in_dim, hidden_dim, z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 3. 执行训练与验证
    epochs = 200
    best_auc = 0.0
    best_ap = 0.0

    print("Starting training process...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer)

        if epoch % 10 == 0:
            auc, ap = test_epoch(model, data)
            if auc > best_auc:
                best_auc = auc
                best_ap = ap
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")

    print("Optimization Finished!")
    print(f"Best Test Performance - AUC: {best_auc:.4f}, AP: {best_ap:.4f}")


if __name__ == '__main__':
    main()