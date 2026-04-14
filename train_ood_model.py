import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax


# Graph-level anomaly-aware readout model.

class AnomalyAwareModel(nn.Module):
    def __init__(self, sem_dim=384, topo_hidden=64, align_dim=32):
        super().__init__()

        # Topology encoder.
        self.gcn = GCNConv(sem_dim, topo_hidden)

        # Modality alignment layers.
        self.proj_topo = nn.Linear(topo_hidden, align_dim)
        self.proj_sem = nn.Linear(sem_dim, align_dim)

        # Mismatch scorer. Input dimension: 32 + 32 + 32 = 96.
        self.scorer = nn.Sequential(
            nn.Linear(align_dim * 3, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

        # Graph-level classifier.
        self.classifier = nn.Linear(topo_hidden, 1)

    def forward(self, x_sem, edge_index, batch_index):
        """
        Args:
            x_sem: Node semantic features with shape [N, 384].
            edge_index: Graph connectivity in COO format.
            batch_index: Graph assignment for each node.
        """
        # Step 1: extract topology-aware features.
        h_topo = self.gcn(x_sem, edge_index).relu()

        # Step 2: align topology and semantic representations.
        z_topo = self.proj_topo(h_topo)
        z_sem = self.proj_sem(x_sem)

        # Step 3: score representation mismatch at the node level.
        diff = torch.abs(z_topo - z_sem)
        concat_feat = torch.cat([z_topo, z_sem, diff], dim=-1)
        s_i = self.scorer(concat_feat)

        # Step 4: aggregate graph features with node-wise attention.
        alpha = softmax(s_i, batch_index)
        weighted_h = h_topo * alpha
        z_graph = global_add_pool(weighted_h, batch_index)

        # Graph-level logits.
        logits = self.classifier(z_graph)
        return logits.squeeze(), alpha, z_topo, z_sem


# Minimal forward-pass smoke test.

if __name__ == "__main__":
    print("Initializing Anomaly-Aware Readout model...")
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32)

    # Create a small synthetic graph for a quick forward-pass test.
    dummy_x = torch.randn(5, 384)
    dummy_edge_index = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4],
                                     [1, 2, 3, 4, 0, 0, 0, 0]], dtype=torch.long)
    dummy_batch = torch.zeros(5, dtype=torch.long)

    # Forward-pass test.
    logits, alphas, _, _ = model(dummy_x, dummy_edge_index, dummy_batch)

    print("\nForward pass completed successfully.")
    print(f"Graph-level logit: {logits.item():.4f}")
    print(f"Attention weights (alphas):\n{alphas.detach().numpy()}")
    print("\nThe model is ready for integration with the generated graph dataset.")


# Standard GCN baseline used for comparison.

class StandardGCN(nn.Module):
    def __init__(self, sem_dim=384, topo_hidden=64):
        super().__init__()
        self.gcn = GCNConv(sem_dim, topo_hidden)
        # Plain classifier without alignment or attention weighting.
        self.classifier = nn.Linear(topo_hidden, 1)

    def forward(self, x_sem, edge_index, batch_index):
        # Step 1: extract node features.
        h = self.gcn(x_sem, edge_index).relu()
        # Step 2: global pooling.
        z_graph = global_add_pool(h, batch_index)
        # Step 3: graph classification.
        logits = self.classifier(z_graph)
        return logits.squeeze()