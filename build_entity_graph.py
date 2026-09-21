"""
build_entity_graph.py

Builds a bipartite entity-relationship graph (IP nodes <-> user_id nodes)
from the SIEM's own event history in MongoDB (db.logs), for the Graph
Neural Network (GNN) anomaly detector.

WHY A GRAPH HERE: entity_risk.py already tracks IP/user risk individually,
with time-decay, but treats each entity independently -- it has no notion
of *who talks to whom*. Fraud rings, compromised-account reuse across
IPs, and coordinated attacker infrastructure all show up as unusual
*connectivity patterns* (e.g. one IP touching many otherwise-unrelated
accounts, or a cluster of accounts sharing an unusual number of IPs) --
exactly the kind of structural signal a GNN can pick up that a
per-entity time series cannot. This directly extends the survey-driven
Autoencoder preference into relational data: same reconstruction-based
anomaly idea, applied to graph structure instead of a flat feature
vector.

GRAPH DEFINITION:
    - Node types: "ip" and "user" (bipartite -- no ip-ip or user-user
      edges are created directly; two entities are connected only if a
      logged event links them, i.e. IP X actually generated an event as
      user Y).
    - Edge: one undirected edge per distinct (ip, user_id) pair seen in
      logs_collection, weighted by event count.
    - Node features (kept small and explainable, same "transparent over
      black-box" preference as _compute_alert_priority in main.py):
        event_count_norm        -- log-scaled total events for this entity
        anomaly_rate            -- fraction of this entity's events flagged anomalous
        avg_risk_score_norm     -- mean risk_score / 100
        failed_login_rate       -- fraction of events that were failed_login (IP nodes)
        admin_flag              -- 1.0 if role ever seen as "admin" (user nodes)
        degree_norm             -- log-scaled degree (how many counterpart
                                    entities this one has touched) -- also
                                    literally what the GNN is meant to
                                    model, but including it as an input
                                    feature alongside the *learned*
                                    structural embedding is standard
                                    practice (a GAE augments, doesn't
                                    replace, node features)

Kept deliberately dependency-light: pure PyTorch, no torch_geometric --
avoids a finicky install this close to the deadline, and the graphs this
project will actually see (hundreds to a few thousand entities) are
small enough that a dense adjacency matrix is completely fine.
"""

import math
from collections import defaultdict

import numpy as np

from database import get_db


def _safe_div(a, b):
    return a / b if b else 0.0


def build_entity_graph(scan_limit=20000):
    """
    Reads recent logs_collection events and builds:
        node_ids:      list[str], "ip:<addr>" or "user:<id>", index == node id
        adjacency:     (N, N) float32 numpy array, symmetric, edge-weighted
        features:      (N, F) float32 numpy array, per-node feature vector
        feature_names: list[str]

    Anything present in the user_id field becomes a "user" node, whether
    it's a real customer, admin, or IoT device id (see
    simulate_iot_devices.py) -- this module doesn't distinguish; that's a
    label-quality question for the report, not a graph-construction bug.
    """
    db = get_db()
    cursor = db.logs.find(
        {},
        {"ip": 1, "user_id": 1, "event_type": 1, "risk_score": 1,
         "anomaly": 1, "role": 1},
    ).sort("_id", -1).limit(scan_limit)

    ip_stats = defaultdict(lambda: {"count": 0, "anomalies": 0, "risk_sum": 0.0, "failed_logins": 0})
    user_stats = defaultdict(lambda: {"count": 0, "anomalies": 0, "risk_sum": 0.0, "is_admin": False})
    edge_weights = defaultdict(float)

    for doc in cursor:
        ip = doc.get("ip")
        user_id = doc.get("user_id")
        risk = doc.get("risk_score") or 0
        is_anom = bool(doc.get("anomaly"))
        event_type = doc.get("event_type") or ""
        role = doc.get("role") or "user"

        if ip:
            s = ip_stats[ip]
            s["count"] += 1
            s["anomalies"] += int(is_anom)
            s["risk_sum"] += risk
            s["failed_logins"] += int(event_type == "failed_login")

        if user_id:
            s = user_stats[user_id]
            s["count"] += 1
            s["anomalies"] += int(is_anom)
            s["risk_sum"] += risk
            s["is_admin"] = s["is_admin"] or (role == "admin")

        if ip and user_id:
            edge_weights[(f"ip:{ip}", f"user:{user_id}")] += 1.0

    node_ids = (
        [f"ip:{ip}" for ip in ip_stats]
        + [f"user:{uid}" for uid in user_stats]
    )
    node_index = {n: i for i, n in enumerate(node_ids)}
    n_nodes = len(node_ids)

    feature_names = [
        "event_count_norm", "anomaly_rate", "avg_risk_score_norm",
        "failed_login_rate", "admin_flag", "degree_norm",
    ]
    features = np.zeros((n_nodes, len(feature_names)), dtype=np.float32)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    degree = defaultdict(float)
    for (a, b), w in edge_weights.items():
        ia, ib = node_index[a], node_index[b]
        adjacency[ia, ib] += w
        adjacency[ib, ia] += w
        degree[a] += w
        degree[b] += w

    for ip, s in ip_stats.items():
        i = node_index[f"ip:{ip}"]
        features[i, 0] = math.log1p(s["count"]) / 10.0
        features[i, 1] = _safe_div(s["anomalies"], s["count"])
        features[i, 2] = _safe_div(s["risk_sum"], s["count"]) / 100.0
        features[i, 3] = _safe_div(s["failed_logins"], s["count"])
        features[i, 4] = 0.0
        features[i, 5] = math.log1p(degree[f"ip:{ip}"]) / 10.0

    for uid, s in user_stats.items():
        i = node_index[f"user:{uid}"]
        features[i, 0] = math.log1p(s["count"]) / 10.0
        features[i, 1] = _safe_div(s["anomalies"], s["count"])
        features[i, 2] = _safe_div(s["risk_sum"], s["count"]) / 100.0
        features[i, 3] = 0.0
        features[i, 4] = 1.0 if s["is_admin"] else 0.0
        features[i, 5] = math.log1p(degree[f"user:{uid}"]) / 10.0

    return node_ids, adjacency, features, feature_names


if __name__ == "__main__":
    ids, adj, feats, names = build_entity_graph()
    print(f"Built graph: {len(ids)} nodes, {int((adj > 0).sum() / 2)} edges")
    print(f"Feature columns: {names}")