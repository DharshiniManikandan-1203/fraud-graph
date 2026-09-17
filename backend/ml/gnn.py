import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..core.graph_engine import FraudGraphEngine
from ..models.schema import EntityType


class VectorizedGraphSAGE:
    """
    High-performance, inductive Graph Neural Network using GraphSAGE message passing.
    Aggregates multi-hop structural neighbor features and outputs:
    1. Node topological embeddings (16-D / 32-D vectors)
    2. GNN-derived inductive fraud probability
    3. 2D embedding coordinates (PCA/t-SNE projection) for frontend latent space visualization.
    """

    def __init__(self, in_dim: int = 23, hidden_dim: int = 32, embed_dim: int = 16, seed: int = 42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.rng = np.random.RandomState(seed)

        # Layer 1 Weights (Self + Neighbor)
        self.W_self_1 = self.rng.randn(in_dim, hidden_dim) * np.sqrt(2.0 / (in_dim + hidden_dim))
        self.W_neigh_1 = self.rng.randn(in_dim, hidden_dim) * np.sqrt(2.0 / (in_dim + hidden_dim))
        self.b_1 = np.zeros((1, hidden_dim))

        # Layer 2 Weights (Self + Neighbor)
        self.W_self_2 = self.rng.randn(hidden_dim, embed_dim) * np.sqrt(2.0 / (hidden_dim + embed_dim))
        self.W_neigh_2 = self.rng.randn(hidden_dim, embed_dim) * np.sqrt(2.0 / (hidden_dim + embed_dim))
        self.b_2 = np.zeros((1, embed_dim))

        # Fraud Classification Head
        self.W_out = self.rng.randn(embed_dim, 1) * np.sqrt(2.0 / embed_dim)
        self.b_out = np.zeros((1, 1))

        # Scaler parameters
        self.mean_ = None
        self.std_ = None

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + eps
        return (x - mean) / std

    def _l2_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True) + eps
        return x / norm

    def fit_and_embed(
        self,
        node_ids: List[str],
        X_features: np.ndarray,
        adj_matrix: np.ndarray,
        labels: Optional[np.ndarray] = None,
        epochs: int = 150,
        lr: float = 0.03
    ) -> Dict[str, Any]:
        """
        Trains GNN weights via inductive supervised/contrastive loss and computes embeddings.
        """
        N, D = X_features.shape
        # Normalize input features
        self.mean_ = np.mean(X_features, axis=0, keepdims=True)
        self.std_ = np.std(X_features, axis=0, keepdims=True) + 1e-5
        X_norm = (X_features - self.mean_) / self.std_

        # Degree-normalized adjacency matrix D^(-1) A
        deg = np.sum(adj_matrix, axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        A_norm = adj_matrix / deg

        # If labels are provided, optimize weights with cross-entropy and class-weighting
        if labels is not None and len(labels) == N:
            y = labels.reshape(-1, 1).astype(float)
            pos_weight = float(np.sum(y == 0)) / max(1.0, float(np.sum(y == 1)))
            pos_weight = min(10.0, max(1.0, pos_weight))

            for epoch in range(epochs):
                # Forward Pass
                # Layer 1
                neigh_1 = np.dot(A_norm, X_norm)
                h1_raw = np.dot(X_norm, self.W_self_1) + np.dot(neigh_1, self.W_neigh_1) + self.b_1
                h1 = self._layer_norm(self._relu(h1_raw))

                # Layer 2
                neigh_2 = np.dot(A_norm, h1)
                h2_raw = np.dot(h1, self.W_self_2) + np.dot(neigh_2, self.W_neigh_2) + self.b_2
                embeddings = self._l2_norm(self._relu(h2_raw))

                # Classification Head
                logits = np.dot(embeddings, self.W_out) + self.b_out
                probs = self._sigmoid(logits)

                # Loss & Backprop gradients
                # Weighted binary cross-entropy gradient
                grad_logits = (probs - y)
                grad_logits[y == 1] *= (pos_weight * 0.4)

                # Gradient on W_out
                dW_out = np.dot(embeddings.T, grad_logits) / N
                db_out = np.sum(grad_logits, axis=0, keepdims=True) / N

                # Backprop to embeddings
                grad_emb = np.dot(grad_logits, self.W_out.T)
                grad_h2_raw = grad_emb * (h2_raw > 0)

                dW_self_2 = np.dot(h1.T, grad_h2_raw) / N
                dW_neigh_2 = np.dot(neigh_2.T, grad_h2_raw) / N
                db_2 = np.sum(grad_h2_raw, axis=0, keepdims=True) / N

                # Update weights
                self.W_out -= lr * dW_out
                self.b_out -= lr * db_out
                self.W_self_2 -= lr * dW_self_2
                self.W_neigh_2 -= lr * dW_neigh_2
                self.b_2 -= lr * db_2

        # Final forward pass
        neigh_1 = np.dot(A_norm, X_norm)
        h1 = self._layer_norm(self._relu(np.dot(X_norm, self.W_self_1) + np.dot(neigh_1, self.W_neigh_1) + self.b_1))
        neigh_2 = np.dot(A_norm, h1)
        h2 = self._l2_norm(self._relu(np.dot(h1, self.W_self_2) + np.dot(neigh_2, self.W_neigh_2) + self.b_2))
        final_probs = self._sigmoid(np.dot(h2, self.W_out) + self.b_out).flatten()

        # Compute 2D projection using SVD / PCA for visualization
        h2_centered = h2 - np.mean(h2, axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(h2_centered, full_matrices=False)
        coords_2d = U[:, :2] * S[:2]
        # Normalize 2D coords to range [-100, 100]
        max_abs = np.max(np.abs(coords_2d)) + 1e-6
        coords_2d = (coords_2d / max_abs) * 100.0

        results = {
            "embeddings": {node_ids[i]: [round(float(v), 4) for v in h2[i]] for i in range(N)},
            "gnn_fraud_scores": {node_ids[i]: round(float(final_probs[i]), 4) for i in range(N)},
            "pca_2d_coords": {
                node_ids[i]: {"x": round(float(coords_2d[i, 0]), 2), "y": round(float(coords_2d[i, 1]), 2)}
                for i in range(N)
            },
        }
        return results

    def predict_node(self, x_vec: np.ndarray, nbr_x_mean: np.ndarray) -> Tuple[float, List[float]]:
        """Inductive prediction for a new/unseen account given its features and its 1-hop neighborhood mean."""
        if self.mean_ is None:
            return 0.1, [0.0] * self.embed_dim

        x_norm = (x_vec.reshape(1, -1) - self.mean_) / self.std_
        nbr_norm = (nbr_x_mean.reshape(1, -1) - self.mean_) / self.std_

        h1 = self._layer_norm(self._relu(np.dot(x_norm, self.W_self_1) + np.dot(nbr_norm, self.W_neigh_1) + self.b_1))
        h2 = self._l2_norm(self._relu(np.dot(h1, self.W_self_2) + np.dot(h1, self.W_neigh_2) + self.b_2))
        prob = float(self._sigmoid(np.dot(h2, self.W_out) + self.b_out)[0, 0])
        emb = [round(float(v), 4) for v in h2[0]]
        return round(prob, 4), emb
