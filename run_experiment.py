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


# Experiment configuration.

NUM_GRAPHS_PER_CLASS = 30  # Total dataset size is 2 * NUM_GRAPHS_PER_CLASS.
DATASET_PATH = "graph_ood_dataset.pt"


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
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
    # Generate in-distribution graphs.
    print("\n--- Generating in-distribution (ID) graphs ---")
    for i in range(NUM_GRAPHS_PER_CLASS):
        texts = generate_node_semantics(client, node_roles, is_ood=False)
        data = build_pyg_data(edge_index, texts, text_encoder, y_label=0.0)
        dataset.append(data)
        print(f"  - ID graph {i + 1}/{NUM_GRAPHS_PER_CLASS} generated")

    # Generate out-of-distribution graphs.
    print("\n--- Generating out-of-distribution (OOD) graphs ---")
    for i in range(NUM_GRAPHS_PER_CLASS):
        texts = generate_node_semantics(client, node_roles, is_ood=True)
        data = build_pyg_data(edge_index, texts, text_encoder, y_label=1.0)
        dataset.append(data)
        print(f"  - OOD graph {i + 1}/{NUM_GRAPHS_PER_CLASS} generated")

    # Cache the dataset for later runs.
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

    # Prepare the dataset.
    full_dataset = prepare_dataset(device)
    random.shuffle(full_dataset)

    # Split into training and test sets.
    split_idx = int(len(full_dataset) * 0.8)
    train_dataset = full_dataset[:split_idx]
    test_dataset = full_dataset[split_idx:]
    print(f"Dataset split completed: {len(train_dataset)} training graphs, {len(test_dataset)} test graphs")

    # Initialize the model and optimizer.
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()

    # Train with classification and alignment losses.
    print("\nStarting model training...")
    epochs = 40
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)

            # Forward pass.
            logits, alphas, z_topo, z_sem = model(data.x, data.edge_index, batch_index)

            # Primary classification loss.
            loss_cls = criterion(logits.unsqueeze(0), data.y.float())
            # Apply the alignment loss to ID graphs only.
            mse_loss = nn.MSELoss()(z_topo, z_sem)
            loss_align = mse_loss * (1.0 - data.y.float())

            # Joint optimization.
            loss = loss_cls + 0.5 * loss_align

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {total_loss / len(train_dataset):.4f}")

    # Evaluate on the test split.
    print("\nEvaluation results")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
            logits, alphas, _, _ = model(data.x, data.edge_index, batch_index)

            # Convert logits to probabilities.
            prob = torch.sigmoid(logits.unsqueeze(0)).item()
            all_preds.append(prob)
            all_labels.append(data.y.item())

    # Compute summary metrics.
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUROC: {auc * 100:.2f}%")

    if auc > 0.8:
        print("\nResult summary: AUROC exceeded 80% on the current split.")


if __name__ == "__main__":
    train_and_evaluate()