import torch
from models.node_detector import DualStreamNodeDetector
from generate_graph_ood import generate_node_level_data
from train_node_model import train_node_epoch, test_node_epoch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载节点级实验数据
    data = generate_node_level_data(dataset='Cora', ood_ratio=0.15).to(device)

    # 初始化模型
    model = DualStreamNodeDetector(
        topo_in_dim=data.x_topo.size(1),
        sem_in_dim=data.h_sem.size(1),
        hidden_dim=128,
        z_dim=64
    ).to(device)

    # ---------------------------------------------------------
    # Phase 0: Train Process 1 (Pre-training & Freezing)
    # ---------------------------------------------------------
    print("Step 1: Freezing GNN Backbone for Topology Invariance...")
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # Phase 1.5: Train Process 2 (Alignment Training)
    # ---------------------------------------------------------
    print("Step 2: Training MLP Projection Heads for Alignment...")
    # 只优化 MLP 层
    optimizer = torch.optim.Adam([
        {'params': model.mlp_topo.parameters()},
        {'params': model.mlp_sem.parameters()}
    ], lr=0.005, weight_decay=5e-4)

    best_auc = 0.0
    for epoch in range(1, 201):
        loss = train_node_epoch(model, data, optimizer)

        if epoch % 10 == 0:
            auc, ap = test_node_epoch(model, data)
            if auc > best_auc:
                best_auc = auc
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

    print("Optimization Finished!")
    print(f"Best Node-level AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()