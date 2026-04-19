import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    z_topo, z_sem = model(data.x_topo, data.edge_index, data.h_sem)

    # 仅约束 ID 节点进行空间对齐
    loss_align = F.mse_loss(z_topo[data.train_mask], z_sem[data.train_mask])

    loss_align.backward()
    optimizer.step()

    return loss_align.item()


def test_epoch(model, data):
    model.eval()
    with torch.no_grad():
        z_topo, z_sem = model(data.x_topo, data.edge_index, data.h_sem)

        # 获取残差异常得分
        scores = model.compute_anomaly_score(z_topo, z_sem)

        test_scores = scores[data.test_mask].cpu().numpy()
        test_labels = data.y_ood[data.test_mask].cpu().numpy()

        # 评估指标计算
        auc = roc_auc_score(test_labels, test_scores)
        ap = average_precision_score(test_labels, test_scores)

    return auc, ap