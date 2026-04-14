import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

from train_ood_model import AnomalyAwareModel, StandardGCN

# Use an English font to avoid rendering issues in figures.
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

DATASET_PATH = "graph_ood_dataset.pt"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(model_name, model, train_dataset, test_dataset, device, is_anomaly_aware=False):
    """Train and evaluate a model on the generated graph dataset."""
    print(f"\nTraining {model_name}...")
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
                # The baseline returns logits only.
                logits = model(data.x, data.edge_index, batch_index)
                loss = criterion(logits.unsqueeze(0), data.y.float())

            loss.backward()
            optimizer.step()

    # Evaluate the trained model.
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

    print("\nRendering attention heatmap...")
    model.eval()

    # Select one OOD graph for visualization.
    ood_data = next(data for data in dataset if data.y.item() == 1.0)
    ood_data = ood_data.to(device)
    batch_index = torch.zeros(ood_data.x.size(0), dtype=torch.long).to(device)

    with torch.no_grad():
        _, alphas, _, _ = model(ood_data.x, ood_data.edge_index, batch_index)
        alphas = alphas.squeeze().cpu().numpy()

    # Reconstruct the graph topology for visualization.
    G = nx.star_graph(4)

    # Figure setup.
    plt.figure(figsize=(8, 6))
    plt.title("Anomaly-Aware Readout: Attention Heatmap on an OOD Graph\n(Node 1 is the designated anomalous node)",
              fontsize=14, pad=15)

    # Compute node layout.
    pos = nx.spring_layout(G, seed=42)

    # Node color reflects attention weight.
    cmap = plt.cm.Reds

    # Draw edges.
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5, edge_color="gray")

    # Draw nodes.
    nodes = nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=alphas, cmap=cmap, vmin=0, vmax=1.0,
                                   edgecolors='black', linewidths=2)

    # Draw labels.
    nx.draw_networkx_labels(G, pos, font_size=16, font_family="sans-serif", font_color="white")

    # Add color bar.
    cbar = plt.colorbar(nodes)
    cbar.set_label('Attention Weight (Alpha)', fontsize=12)

    # Save the figure.
    plt.savefig("attention_heatmap.png", dpi=300, bbox_inches='tight')
    print("Figure saved to 'attention_heatmap.png'.")
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the cached dataset.
    full_dataset = torch.load(DATASET_PATH)
    random.shuffle(full_dataset)
    split_idx = int(len(full_dataset) * 0.8)
    train_dataset = full_dataset[:split_idx]
    test_dataset = full_dataset[split_idx:]

    # Train the baseline model.
    model_base = StandardGCN().to(device)
    evaluate_model("Standard GCN (Baseline)", model_base, train_dataset, test_dataset, device, is_anomaly_aware=False)

    # Train the anomaly-aware model.
    model_ours = AnomalyAwareModel().to(device)
    model_ours = evaluate_model("Anomaly-Aware Model (Ours)", model_ours, train_dataset, test_dataset, device,
                                is_anomaly_aware=True)

    # Render the attention heatmap.
    plot_attention_heatmap(model_ours, test_dataset, device)