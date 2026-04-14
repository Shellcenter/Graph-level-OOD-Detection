import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, roc_curve
from env_config import configure_proxy

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Model definitions.

class StandardNodeGCN(nn.Module):
    """GCN baseline with energy-based OOD scoring."""

    def __init__(self, in_dim=1433, topo_hidden=64, num_classes=7):
        super().__init__()
        self.gcn = GCNConv(in_dim, topo_hidden)
        self.classifier = nn.Linear(topo_hidden, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        return self.classifier(h)

    def get_energy_score(self, x, edge_index, T=1.0):
        logits = self.forward(x, edge_index)
        # Higher energy indicates a more anomalous node.
        energy = -T * torch.logsumexp(logits / T, dim=-1)
        return energy


class NodeAnomalyAwareModel(nn.Module):
    """Node-level anomaly-aware model with topology-semantic alignment."""

    def __init__(self, in_dim=1433, topo_hidden=64, align_dim=32, num_classes=7):
        super().__init__()
        self.gcn = GCNConv(in_dim, topo_hidden)
        self.proj_topo = nn.Linear(topo_hidden, align_dim)
        self.proj_sem = nn.Linear(in_dim, align_dim)

        self.classifier = nn.Linear(align_dim, num_classes)

    def forward(self, x, edge_index):
        h_topo = self.gcn(x, edge_index).relu()
        z_topo = self.proj_topo(h_topo)
        z_sem = self.proj_sem(x)

        # Use aligned topology features for node classification.
        logits = self.classifier(z_topo)

        # Use the representation gap as the anomaly score.
        anomaly_scores = torch.norm(z_topo - z_sem, p=2, dim=-1)

        return logits, anomaly_scores, z_topo, z_sem


# Evaluation metrics.

def compute_metrics(labels, scores):
    """Compute AUROC and FPR95 for node-level OOD detection."""
    auc = roc_auc_score(labels, scores)

    # FPR95 is the false-positive rate at 95% true-positive rate.
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx[0]] if len(idx) > 0 else 1.0

    return auc, fpr95


# Benchmark pipeline: load data, inject perturbations, train, and evaluate.

def run_benchmark():
    set_seed(42)
    proxy_port = configure_proxy()
    if proxy_port:
        print(f"Proxy configured on port: {proxy_port}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")

    # Load the Cora benchmark dataset.
    print("\nLoading benchmark dataset: Cora...")
    dataset = Planetoid(root='./data', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    print(
        f"Cora statistics | Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, "
        f"Features: {data.num_features}"
    )

    # Reserve a subset of test nodes as OOD examples.
    print("\nInjecting OOD perturbations into the test split...")
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    # Use the first 200 test nodes as OOD examples.
    ood_idx = test_idx[:200]
    id_test_idx = test_idx[200:]

    # Shuffle selected node features to simulate semantic inconsistency.
    tampered_x = data.x.clone()
    shuffled_indices = torch.randperm(ood_idx.size(0))
    tampered_x[ood_idx] = data.x[ood_idx[shuffled_indices]]
    data.x_tampered = tampered_x


    # Train the baseline model.

    print("\nTraining baseline model: Standard GCN (Energy)")
    base_model = StandardNodeGCN(in_dim=dataset.num_features).to(device)
    base_opt = optim.Adam(base_model.parameters(), lr=0.01, weight_decay=5e-4)

    # The baseline is trained with node classification supervision.
    for epoch in range(100):
        base_model.train()
        base_opt.zero_grad()
        logits = base_model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        base_opt.step()

    base_model.eval()
    with torch.no_grad():
        # Evaluate on perturbed node features.
        energy_scores = base_model.get_energy_score(data.x_tampered, data.edge_index)
        # Split scores into ID and OOD subsets.
        id_scores_base = energy_scores[id_test_idx].cpu().numpy()
        ood_scores_base = energy_scores[ood_idx].cpu().numpy()

    labels_base = np.concatenate([np.zeros(len(id_scores_base)), np.ones(len(ood_scores_base))])
    scores_base = np.concatenate([id_scores_base, ood_scores_base])
    base_auc, base_fpr95 = compute_metrics(labels_base, scores_base)
    print(f"Baseline results | AUROC: {base_auc * 100:.2f}% | FPR95: {base_fpr95 * 100:.2f}%")


    # Train the anomaly-aware model.

    print("\nTraining model: Anomaly-Aware Model")
    # Use the dataset class count for the classifier head.
    our_model = NodeAnomalyAwareModel(in_dim=dataset.num_features, num_classes=dataset.num_classes).to(device)
    our_opt = optim.Adam(our_model.parameters(), lr=0.01)

    for epoch in range(100):
        our_model.train()
        our_opt.zero_grad()
        logits, scores, z_topo, z_sem = our_model(data.x, data.edge_index)

        # Classification loss is computed on labeled training nodes.
        loss_cls = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        # Apply alignment regularization over the full graph.
        loss_align = nn.MSELoss()(z_topo, z_sem)

        # Joint optimization.
        loss = loss_cls + 1.0 * loss_align
        loss.backward()
        our_opt.step()

    our_model.eval()
    with torch.no_grad():
        _, our_scores, _, _ = our_model(data.x_tampered, data.edge_index)
        id_scores_our = our_scores[id_test_idx].cpu().numpy()
        ood_scores_our = our_scores[ood_idx].cpu().numpy()

    labels_our = np.concatenate([np.zeros(len(id_scores_our)), np.ones(len(ood_scores_our))])
    scores_our = np.concatenate([id_scores_our, ood_scores_our])
    our_auc, our_fpr95 = compute_metrics(labels_our, scores_our)
    print(f"Anomaly-Aware results | AUROC: {our_auc * 100:.2f}% | FPR95: {our_fpr95 * 100:.2f}%")

    print("\nBenchmark run completed.")


if __name__ == "__main__":
    run_benchmark()