"""
gnn_inference.py

Live-serving counterpart to train_gnn_local.py: rebuilds the current
entity graph from MongoDB, runs it through the trained Graph Autoencoder,
and returns per-entity structural anomaly scores.

WHY RE-BUILD THE GRAPH EVERY CALL (not just look up a stored score, the
way entity_risk.py's get_entity_risk() does): the GNN's whole point is
RELATIONAL -- an entity's anomaly score depends on the CURRENT shape of
the graph around it, not just its own history. Re-scoring on the graph
built from recent activity is the honest way to serve this, the same
way network_stream.py is upfront that its model needs a real
flow-feature vector rather than silently approximating one. For a
small-to-medium graph (hundreds to low thousands of entities) this is
cheap -- a couple of dense matmuls -- so it's fine to do per-request
rather than needing a background job.

Put this file in the same folder as gnn_autoencoder.pt, gnn_scaler.pkl,
and gnn_threshold.json (the three files train_gnn_local.py produces).

Usage from main.py:
    from gnn_inference import score_entity_graph
    result = score_entity_graph(top_n=15)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from build_entity_graph import build_entity_graph
from train_gnn_local import GCNEncoder, gae_reconstruct, _normalize_adjacency

_DIR = Path(__file__).parent

_model = None
_scaler = None
_meta = None


def _load():
    global _model, _scaler, _meta
    if _model is not None:
        return

    with open(_DIR / "gnn_threshold.json") as f:
        _meta = json.load(f)

    _model = GCNEncoder(
        in_dim=_meta["in_dim"], hidden_dim=_meta["hidden_dim"], embed_dim=_meta["embed_dim"],
    )
    _model.load_state_dict(torch.load(_DIR / "gnn_autoencoder.pt", map_location="cpu"))
    _model.eval()

    _scaler = joblib.load(_DIR / "gnn_scaler.pkl")


def score_entity_graph(scan_limit=20000, top_n=15, rng_seed=42):
    """
    Rebuilds the current entity graph and scores every node.

    Returns:
        {
          "available": bool,
          "graph_size": {"nodes": int, "edges": int},
          "threshold": float,
          "results": [
            {"entity": "ip:203.0.113.14" | "user:cust_1120",
             "entity_type": "ip"|"user", "entity_id": str,
             "graph_anomaly_score": float,   # 0-100, same scale
                                              # convention as
                                              # fraud_score/intrusion_score
             "is_anomaly": bool,
             "degree": int,
             "top_neighbours": [str, ...]},  # XAI: a relational model's
                                              # most natural explanation
                                              # IS its neighbourhood
            ...
          ],
        }
    Never raises past graph-build -- if the graph is too small to score
    meaningfully, returns available=False with a reason, same pattern as
    lstm_inference.py's score_user_sequence() min_events guard.
    """
    _load()

    node_ids, adjacency, features, _ = build_entity_graph(scan_limit=scan_limit)
    n_nodes = len(node_ids)

    if n_nodes < 10:
        return {
            "available": False,
            "reason": f"Graph too small to score ({n_nodes} entities). "
                      f"Needs more logged activity first.",
        }

    features_scaled = _scaler.transform(features).astype(np.float32)
    a_norm = torch.tensor(_normalize_adjacency(adjacency), dtype=torch.float32)
    x = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        z = _model(x, a_norm)
        recon = gae_reconstruct(z).numpy()

    rng = np.random.default_rng(rng_seed)
    threshold = _meta["threshold"]

    scored = []
    for idx, node in enumerate(node_ids):
        neighbours = np.where(adjacency[idx] > 0)[0]
        if len(neighbours):
            error = float(np.mean((1.0 - recon[idx, neighbours]) ** 2))
        else:
            sample = rng.choice(n_nodes, size=min(10, n_nodes), replace=False)
            error = float(np.mean(recon[idx, sample] ** 2))

        entity_type, entity_id = node.split(":", 1)
        top_neighbour_idx = (
            neighbours[np.argsort(-adjacency[idx, neighbours])][:3]
            if len(neighbours) else []
        )

        scored.append({
            "entity": node,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "graph_anomaly_score": round(float(np.clip((error / threshold) * 50, 0, 100)), 1),
            "is_anomaly": error > threshold,
            "degree": int(len(neighbours)),
            "top_neighbours": [node_ids[i] for i in top_neighbour_idx],
        })

    scored.sort(key=lambda r: -r["graph_anomaly_score"])

    return {
        "available": True,
        "graph_size": {"nodes": n_nodes, "edges": int((adjacency > 0).sum() / 2)},
        "threshold": threshold,
        "results": scored[:top_n],
    }


if __name__ == "__main__":
    result = score_entity_graph(top_n=10)
    if not result.get("available"):
        print(result["reason"])
    else:
        print(f"Graph: {result['graph_size']}")
        print("\nTop structurally-anomalous entities:")
        for r in result["results"]:
            print(f"  {r['entity']:<30} score={r['graph_anomaly_score']:>5.1f} "
                  f"degree={r['degree']:<4} anomaly={r['is_anomaly']} "
                  f"neighbours={r['top_neighbours']}")