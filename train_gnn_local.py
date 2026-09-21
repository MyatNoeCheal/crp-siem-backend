"""
train_gnn_local.py

Trains a Graph Autoencoder (GAE) -- the GNN half of the Smart SIEM's
detection suite -- over the entity-relationship graph built by
build_entity_graph.py (IP nodes <-> user nodes, edges = "this IP
generated an event as this user").

METHOD (same reconstruction-based anomaly idea as every other model in
this project -- see train_autoencoder_local.py / train_lstm_local.py /
train_network_intrusion_local.py for the same pattern applied to
transactions/sequences/network flows):
    1. A 2-layer Graph Convolutional Network (GCN) encodes each node into
       a low-dimensional embedding, using both the node's own features
       AND its neighbours' (standard message passing).
    2. A simple inner-product decoder reconstructs the adjacency matrix
       from those embeddings (Kipf & Welling, 2016, "Variational Graph
       Auto-Encoders" -- this project uses the non-variational GAE for
       simplicity/speed, consistent with using MLPRegressor instead of a
       full VAE for the fraud Autoencoder).
    3. Entities whose local connectivity the model reconstructs POORLY
       are structurally unusual -- e.g. an IP suddenly connected to many
       accounts with no shared history, or an account reachable from an
       abnormal number of IPs -- and get a high graph anomaly score.

PURE PYTORCH, NO torch_geometric: this project's graphs (a few hundred
to a few thousand entities) are small enough that a dense adjacency
matrix and a couple of matmuls are simpler and far less fragile to
install than pulling in torch_geometric this close to the deadline.

UNSUPERVISED, SAME HONESTY NOTE AS THE LSTM MODEL: there's no
ground-truth "this IP/account pair is part of a fraud ring" label
anywhere in this project's data, live or CERT/UNSW/creditcard. The
threshold below is therefore a distributional cutoff (percentile of
validation reconstruction error), not F1-optimized -- report this
honestly in the Evaluation Framework section, exactly like
lstm_threshold.json's note already does for the LSTM model.

RUN THIS ON YOUR OWN MACHINE, with MongoDB reachable (local or Atlas --
same MONGODB_URI env var database.py already uses) and enough log
history for a meaningful graph. If you've only got a few dozen events,
seed more first (seed_test_data.py / simulate_ecommerce_traffic.py /
simulate_iot_devices.py) -- a graph AE needs enough edges to learn what
"normal connectivity" looks like.

Usage:
    python train_gnn_local.py

Output (same folder):
    - gnn_autoencoder.pt     (trained PyTorch model weights)
    - gnn_scaler.pkl         (StandardScaler fit on node features)
    - gnn_threshold.json     (threshold + config + node/feature metadata)
"""

import json

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from build_entity_graph import build_entity_graph

MODEL_OUT = "gnn_autoencoder.pt"
SCALER_OUT = "gnn_scaler.pkl"
THRESHOLD_OUT = "gnn_threshold.json"

HIDDEN_DIM = 32
EMBED_DIM = 16
MAX_EPOCHS = 300
LR = 0.01
THRESHOLD_PERCENTILE = 95
NEG_SAMPLE_RATIO = 1.0   # negative (non-)edges sampled per positive edge, per epoch
RANDOM_STATE = 42


def _normalize_adjacency(adj):
    """Symmetric normalization D^-1/2 (A + I) D^-1/2, the standard GCN
    propagation rule (Kipf & Welling, 2017)."""
    n = adj.shape[0]
    a_hat = adj + np.eye(n, dtype=np.float32)
    deg = a_hat.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    d_mat = np.diag(deg_inv_sqrt).astype(np.float32)
    return d_mat @ a_hat @ d_mat


class GCNEncoder(nn.Module):
    """Two-layer GCN: node features + graph structure -> embedding.
    Layer sizes are saved in gnn_threshold.json so gnn_inference.py can
    rebuild this exact architecture before loading weights (same pattern
    as LSTMAutoencoder in lstm_inference.py)."""
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=True)

    def forward(self, x, a_norm):
        h = F.relu(a_norm @ self.w1(x))
        z = a_norm @ self.w2(h)
        return z


def gae_reconstruct(z):
    """Inner-product decoder: sigmoid(Z Z^T) -> predicted edge
    probability for every node pair."""
    return torch.sigmoid(z @ z.t())


def _sample_negative_edges(adjacency, n_samples, rng):
    """Uniform random non-edges for negative sampling during GAE
    training -- standard practice, since the decoder needs both positive
    and negative examples to learn anything (an all-ones prediction
    would trivially "reconstruct" every real edge)."""
    n = adjacency.shape[0]
    negs = []
    tries = 0
    while len(negs) < n_samples and tries < n_samples * 20:
        i, j = rng.integers(0, n, size=2)
        tries += 1
        if i == j or adjacency[i, j] > 0:
            continue
        negs.append((i, j))
    return negs


def main():
    print("Building entity graph from MongoDB...")
    node_ids, adjacency, features, feature_names = build_entity_graph()
    n_nodes = len(node_ids)
    n_edges = int((adjacency > 0).sum() / 2)
    print(f"Graph: {n_nodes} nodes ({n_edges} edges)")

    if n_nodes < 10 or n_edges < 10:
        raise RuntimeError(
            f"Graph too small to train on ({n_nodes} nodes, {n_edges} edges). "
            f"Seed more events first (seed_test_data.py / "
            f"simulate_ecommerce_traffic.py) and retry."
        )

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features).astype(np.float32)
    a_norm_np = _normalize_adjacency(adjacency)

    rng = np.random.default_rng(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    x = torch.tensor(features_scaled, dtype=torch.float32)
    a_norm = torch.tensor(a_norm_np, dtype=torch.float32)

    # Positive edge list (upper triangle only, undirected)
    pos_i, pos_j = np.triu_indices(n_nodes, k=1)
    pos_mask = adjacency[pos_i, pos_j] > 0
    pos_edges = list(zip(pos_i[pos_mask].tolist(), pos_j[pos_mask].tolist()))
    print(f"Training on {len(pos_edges)} positive edges "
          f"(+ {int(len(pos_edges) * NEG_SAMPLE_RATIO)} negatives resampled each epoch)")

    model = GCNEncoder(in_dim=features.shape[1], hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pos_i_t = torch.tensor([e[0] for e in pos_edges])
    pos_j_t = torch.tensor([e[1] for e in pos_edges])

    print("\nTraining Graph Autoencoder...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        z = model(x, a_norm)
        recon = gae_reconstruct(z)

        neg_edges = _sample_negative_edges(adjacency, int(len(pos_edges) * NEG_SAMPLE_RATIO), rng)

        pos_pred = recon[pos_i_t, pos_j_t]
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))

        if neg_edges:
            neg_i_t = torch.tensor([e[0] for e in neg_edges])
            neg_j_t = torch.tensor([e[1] for e in neg_edges])
            neg_pred = recon[neg_i_t, neg_j_t]
            neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        else:
            neg_loss = torch.tensor(0.0)

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} "
                  f"(pos={pos_loss.item():.4f} neg={neg_loss.item():.4f})")

    # ---- Per-node anomaly score: how poorly is THIS node's real
    # connectivity reconstructed, averaged over its actual edges (and,
    # for isolated nodes, over a small random negative sample so a
    # degree-0 node still gets a score instead of an undefined one) ----
    model.eval()
    with torch.no_grad():
        z = model(x, a_norm)
        recon = gae_reconstruct(z).numpy()

    node_errors = np.zeros(n_nodes, dtype=np.float32)
    for idx in range(n_nodes):
        neighbours = np.where(adjacency[idx] > 0)[0]
        if len(neighbours):
            node_errors[idx] = float(np.mean((1.0 - recon[idx, neighbours]) ** 2))
        else:
            sample = rng.choice(n_nodes, size=min(10, n_nodes), replace=False)
            node_errors[idx] = float(np.mean(recon[idx, sample] ** 2))

    threshold = float(np.percentile(node_errors, THRESHOLD_PERCENTILE))
    flagged = int((node_errors > threshold).sum())
    print(f"\nSelected threshold ({THRESHOLD_PERCENTILE}th percentile of "
          f"node reconstruction error): {threshold:.6f}")
    print(f"Flags {flagged}/{n_nodes} entities as structurally anomalous.")

    torch.save(model.state_dict(), MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    with open(THRESHOLD_OUT, "w") as f:
        json.dump({
            "threshold": threshold,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "in_dim": features.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "embed_dim": EMBED_DIM,
            "feature_names": feature_names,
            "node_error_stats": {
                "mean": float(np.mean(node_errors)),
                "std": float(np.std(node_errors)),
                "min": float(np.min(node_errors)),
                "max": float(np.max(node_errors)),
            },
            "training_graph_size": {"nodes": n_nodes, "edges": n_edges},
            "note": (
                "Unsupervised model -- no ground-truth fraud-ring/entity-"
                "linkage labels exist anywhere in this project's data, so "
                "this threshold is a distributional cutoff (95th percentile "
                "of node reconstruction error), not F1-optimized. Report "
                "this honestly in the Evaluation Framework section, same "
                "pattern as lstm_threshold.json's note for the LSTM model."
            ),
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved scaler to {SCALER_OUT}")
    print(f"Saved threshold + config to {THRESHOLD_OUT}")
    print("\nNext: wire gnn_inference.py's score_entity_graph() into main.py "
          "(see the /entity-risk/graph endpoint) so the dashboard can "
          "surface structurally-anomalous entities.")


if __name__ == "__main__":
    main()