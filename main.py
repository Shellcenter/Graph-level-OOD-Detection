import sys
import os
import time

# Ensure local imports work when running this file directly.
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
from env_config import configure_proxy
from models.anomaly_aware import NodeAnomalyAwareModel
from utils.metrics import compute_metrics


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def setup_logger(args):
    """Configure console and file logging for an experiment run."""

    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a timestamped log file for each run.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"exp_{args.dataset}_ood{args.ood_ratio}_{timestamp}.log")

    # Reset the root logger to avoid duplicate handlers.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler.
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler.
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return log_file

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Graph OOD Detection Benchmark")
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--ood_ratio", type=float, default=0.2, help="OOD corruption ratio in the test split")
    args = parser.parse_args()

    # Initialize logging.
    log_file_path = setup_logger(args)
    logging.info(f"Experiment logging initialized. Log file: {log_file_path}")
    proxy_port = configure_proxy()
    if proxy_port:
        logging.info(f"Proxy configured on port: {proxy_port}")
    logging.info(
        f"Starting experiment | Dataset: {args.dataset} | Learning rate: {args.lr} | Epochs: {args.epochs}"
    )

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Computation device: {device}")

    # Load the dataset and build a perturbed test split.
    logging.info("Loading benchmark dataset...")
    dataset = Planetoid(root='./data', name=args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    logging.info("Applying OOD perturbations to the test split...")
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    ood_num = int(len(test_idx) * args.ood_ratio)
    ood_idx = test_idx[:ood_num]
    id_test_idx = test_idx[ood_num:]

    # Shuffle selected test-node features to simulate OOD perturbations.
    tampered_x = data.x.clone()
    shuffled_indices = torch.randperm(ood_idx.size(0))
    tampered_x[ood_idx] = data.x[ood_idx[shuffled_indices]]
    data.x_tampered = tampered_x

    # Initialize the anomaly-aware model.
    logging.info("Initializing the Anomaly-Aware model...")
    our_model = NodeAnomalyAwareModel(in_dim=dataset.num_features, num_classes=dataset.num_classes).to(device)
    our_opt = optim.Adam(our_model.parameters(), lr=args.lr)

    # Joint training with classification and alignment losses.
    logging.info("Starting joint training with full-graph alignment...")
    for epoch in range(1, args.epochs + 1):
        our_model.train()
        our_opt.zero_grad()
        logits, scores, z_topo, z_sem = our_model(data.x, data.edge_index)

        loss_cls = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss_align = nn.MSELoss()(z_topo, z_sem)  # Full-graph alignment loss.

        loss = loss_cls + 1.0 * loss_align
        loss.backward()
        our_opt.step()

        if epoch % 20 == 0 or epoch == 1:
            logging.info(f"   Epoch [{epoch:03d}/{args.epochs}] | Total Loss: {loss.item():.4f}")

    # Evaluate on the perturbed test split.
    logging.info("Evaluating on the tampered test split...")
    our_model.eval()
    with torch.no_grad():
        _, our_scores, _, _ = our_model(data.x_tampered, data.edge_index)
        id_scores_our = our_scores[id_test_idx].cpu().numpy()
        ood_scores_our = our_scores[ood_idx].cpu().numpy()

    labels_our = np.concatenate([np.zeros(len(id_scores_our)), np.ones(len(ood_scores_our))])
    scores_our = np.concatenate([id_scores_our, ood_scores_our])

    # Compute AUROC and FPR95.
    our_auc, our_fpr95 = compute_metrics(labels_our, scores_our)

    logging.info("=" * 50)
    logging.info(f"Final results | AUROC: {our_auc * 100:.2f}% | FPR95: {our_fpr95 * 100:.2f}%")
    logging.info("=" * 50)
    logging.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()