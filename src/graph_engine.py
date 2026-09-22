"""
Graph Construction & Graph Neural Network (GNN) Engine.
Builds heterogeneous transaction graphs (Users, Devices, IPs) and implements:
1. NetworkX community detection & Degree/PageRank centralities.
2. PyTorch Geometric (PyG) Graph Convolutional Network (GCN) / GraphSAGE for node anomaly embedding.
3. Subgraph extraction for visual forensic inspection.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudGNN(nn.Module):
    """
    Two-layer Graph Convolutional / SAGE Network for detecting anomalous nodes.
    Embeds structural connectivity (device sharing, layering rings) into a latent representation.
    """
    def __init__(self, in_features: int, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.classifier = nn.Linear(out_dim, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Layer 1: Normalized neighborhood aggregation + linear transform
        h1 = torch.spmm(adj_norm, x) if adj_norm.is_sparse else torch.matmul(adj_norm, x)
        h1 = F.relu(self.fc1(h1))
        h1 = self.dropout(h1)

        # Layer 2: Second hop neighborhood aggregation
        h2 = torch.spmm(adj_norm, h1) if adj_norm.is_sparse else torch.matmul(adj_norm, h1)
        embeddings = F.relu(self.fc2(h2))

        # Classification logits
        logits = self.classifier(embeddings)
        return embeddings, logits


class TransactionGraphEngine:
    def __init__(self):
        self.G = nx.Graph()
        self.user_node_map = {}
        self.node_features = None
        self.gnn_model = None
        self.node_embeddings = None
        self.gnn_scores = None

    def build_graph(self, df_tx: pd.DataFrame):
        """
        Builds a multi-entity bipartite/heterogeneous graph:
        Nodes: Users, Devices, IP Addresses
        Edges:
        - User -> Transacted_With -> User (weighted by tx amount & frequency)
        - User -> Associated_With -> Device
        - User -> Connected_From -> IP
        """
        self.G.clear()
        
        for _, row in df_tx.iterrows():
            u_src = f"user:{row['sender_id']}"
            u_dst = f"user:{row['receiver_id']}"
            dev = f"dev:{row['device_id']}"
            ip = f"ip:{row['ip_address']}"
            
            # Add nodes with types
            self.G.add_node(u_src, node_type="user", entity_id=row['sender_id'])
            self.G.add_node(u_dst, node_type="user", entity_id=row['receiver_id'])
            self.G.add_node(dev, node_type="device", entity_id=row['device_id'])
            self.G.add_node(ip, node_type="ip", entity_id=row['ip_address'])
            
            # Add edges
            if self.G.has_edge(u_src, u_dst):
                self.G[u_src][u_dst]["weight"] += 1
                self.G[u_src][u_dst]["amount"] += row["amount"]
            else:
                self.G.add_edge(u_src, u_dst, relation="transacted_to", weight=1, amount=row["amount"])
                
            self.G.add_edge(u_src, dev, relation="used_device", weight=1)
            self.G.add_edge(u_src, ip, relation="logged_from", weight=1)

        # Precompute structural graph metrics
        self._compute_centralities()
        return self.G

    def _compute_centralities(self):
        """Computes PageRank and Degree Centrality to identify mule hubs."""
        self.pagerank = nx.pagerank(self.G, weight="weight")
        self.degrees = dict(self.G.degree())

    def train_gnn(self, df_tx: pd.DataFrame, epochs: int = 60, lr: float = 0.01):
        """
        Extracts user subnetwork adjacency and trains FraudGNN to distinguish fraud rings.
        """
        user_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "user"]
        self.user_node_map = {node: i for i, node in enumerate(user_nodes)}
        N = len(user_nodes)
        
        # 1. Construct Feature Matrix: [degree, pagerank, total_sent, total_received, shared_device_count]
        user_stats = {}
        for _, row in df_tx.iterrows():
            u = f"user:{row['sender_id']}"
            if u not in user_stats:
                user_stats[u] = {"sent": 0.0, "recv": 0.0, "count": 0, "is_fraud": row["is_fraud"]}
            user_stats[u]["sent"] += row["amount"]
            user_stats[u]["count"] += 1
            user_stats[u]["is_fraud"] = max(user_stats[u]["is_fraud"], row["is_fraud"])
            
            v = f"user:{row['receiver_id']}"
            if v not in user_stats:
                user_stats[v] = {"sent": 0.0, "recv": 0.0, "count": 0, "is_fraud": 0}
            user_stats[v]["recv"] += row["amount"]

        features = []
        labels = []
        for u in user_nodes:
            deg = self.degrees.get(u, 0)
            pr = self.pagerank.get(u, 0.0)
            st = user_stats.get(u, {"sent": 0.0, "recv": 0.0, "count": 0, "is_fraud": 0})
            
            # Check how many accounts share the same device as this user
            dev_neighbors = [nbr for nbr in self.G.neighbors(u) if self.G.nodes[nbr].get("node_type") == "device"]
            shared_device_degree = max([self.degrees.get(d, 1) for d in dev_neighbors], default=1)
            
            features.append([
                np.log1p(deg),
                pr * 1000,
                np.log1p(st["sent"]),
                np.log1p(st["recv"]),
                float(shared_device_degree)
            ])
            labels.append(st["is_fraud"])

        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        
        # 2. Normalized User-User Adjacency Matrix (A_hat = D^-1/2 * (A + I) * D^-1/2)
        A = np.zeros((N, N), dtype=np.float32)
        for u in user_nodes:
            idx_u = self.user_node_map[u]
            for nbr in self.G.neighbors(u):
                if nbr in self.user_node_map:
                    idx_v = self.user_node_map[nbr]
                    A[idx_u, idx_v] = 1.0
                    
        A_tilde = A + np.eye(N)
        D_tilde = np.diag(1.0 / np.sqrt(np.sum(A_tilde, axis=1)))
        A_norm = torch.tensor(D_tilde @ A_tilde @ D_tilde, dtype=torch.float32)

        # 3. Train GNN
        model = FraudGNN(in_features=X.shape[1], hidden_dim=32, out_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Class weighting for class imbalance
        weight = torch.tensor([1.0, 5.0])
        criterion = nn.CrossEntropyLoss(weight=weight)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings, logits = model(X, A_norm)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings, logits = model(X, A_norm)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()

        self.gnn_model = model
        self.node_embeddings = embeddings.numpy()
        self.gnn_scores = {u: float(probs[self.user_node_map[u]]) for u in user_nodes}
        return self.gnn_scores

    def get_ego_subgraph(self, target_node: str, radius: int = 2) -> dict:
        """
        Extracts a localized k-hop neighborhood for visualization.
        Returns node and edge dictionary for Plotly graph visualizer.
        """
        if target_node not in self.G:
            # Fallback if bare user id
            if f"user:{target_node}" in self.G:
                target_node = f"user:{target_node}"
            else:
                return {"nodes": [], "edges": []}
                
        subG = nx.ego_graph(self.G, target_node, radius=radius)
        pos = nx.spring_layout(subG, seed=42)
        
        nodes_data = []
        for n, d in subG.nodes(data=True):
            ntype = d.get("node_type", "unknown")
            score = self.gnn_scores.get(n, 0.0) if ntype == "user" else 0.0
            nodes_data.append({
                "id": n,
                "label": n.split(":")[-1],
                "type": ntype,
                "x": float(pos[n][0]),
                "y": float(pos[n][1]),
                "fraud_risk": score,
                "is_target": (n == target_node)
            })
            
        edges_data = []
        for u, v, d in subG.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "relation": d.get("relation", "link"),
                "amount": d.get("amount", 0.0),
                "x0": float(pos[u][0]),
                "y0": float(pos[u][1]),
                "x1": float(pos[v][0]),
                "y1": float(pos[v][1])
            })
            
        return {"nodes": nodes_data, "edges": edges_data, "num_nodes": len(nodes_data)}
