"""
Adaptive Generative Synthetic Sampling (AGSS)

Implements the oversampling algorithm from:
  Nama & Sharmila Banu, "Credit Card Fraud Detection Using Deep Learning
  Techniques and Handling Unbalanced Class Distributions With AGSS",
  IEEE Access, Jan 2026.

Algorithm (Section IV.A):
  1. Extract minority class samples
  2. Cluster with DBSCAN (eps=0.8, min_samples=3)
  3. Within each dense cluster, generate synthetic points via KNN + curvature interpolation:
       x_syn = p + alpha*(q-p) + gamma*sin(theta)*perp(q-p)
  4. Combine with original data
"""

import numpy as np
from math import ceil
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class AGSS:
    def __init__(
        self,
        eps: float = 0.8,
        min_samples: int = 3,
        n_neighbors: int = 3,
        random_state: int = 42,
        adaptive_eps: bool = True,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        # If True, fall back to data-driven eps when paper's default finds <2 clusters.
        # Necessary for high-dimensional data (30D) where eps=0.8 is too small.
        self.adaptive_eps = adaptive_eps

    def _choose_eps(self, X_min: np.ndarray) -> float:
        """Return data-driven eps using the p25 of 3-NN distances (elbow heuristic)."""
        k = min(self.n_neighbors + 1, len(X_min) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        dists, _ = nbrs.kneighbors(X_min)
        knn_dists = dists[:, k]  # distance to k-th nearest neighbor
        return float(np.percentile(knn_dists, 25))

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)

        minority_mask = y == 1
        X_min = X[minority_mask]
        n_majority = int((y == 0).sum())
        n_minority = len(X_min)
        n_synthetic = n_majority - n_minority

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        # Step 1: DBSCAN on minority samples
        eps = self.eps
        labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(X_min)
        unique_clusters = [c for c in np.unique(labels) if c != -1]

        if len(unique_clusters) <= 1 and self.adaptive_eps:
            # Paper's eps=0.8 was tuned for their specific preprocessing; in high-dim
            # data (30D) it's often too small. Fall back to a data-driven eps.
            eps = self._choose_eps(X_min)
            print(f"AGSS: eps={self.eps} found <2 clusters; using adaptive eps={eps:.3f}")
            labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(X_min)
            unique_clusters = [c for c in np.unique(labels) if c != -1]

        if len(unique_clusters) <= 1:
            print("AGSS: No clusters found by DBSCAN, returning original data.")
            return X.copy(), y.copy()

        # Step 2: Build cluster dictionary (exclude noise)
        clusters = {c: X_min[labels == c] for c in unique_clusters}

        # Step 3: Generate synthetic samples
        synthetic = []
        per_cluster = ceil(n_synthetic / len(clusters))

        for cluster_pts in clusters.values():
            if len(cluster_pts) < 2:
                continue

            # Fit KNN within the cluster
            k = min(self.n_neighbors, len(cluster_pts) - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(cluster_pts)

            n_to_generate = min(per_cluster, n_synthetic - len(synthetic))
            if n_to_generate <= 0:
                break

            for _ in range(n_to_generate):
                # Pick random base point p and a KNN neighbor q
                idx_p = rng.integers(0, len(cluster_pts))
                p = cluster_pts[idx_p]
                dists, neighbor_idxs = nbrs.kneighbors([p])
                # neighbor_idxs[0][0] is p itself, so take from [1:]
                neighbor_pool = neighbor_idxs[0][1:]
                if len(neighbor_pool) == 0:
                    continue
                idx_q = rng.choice(neighbor_pool)
                q = cluster_pts[idx_q]

                alpha = rng.uniform(0.0, 1.0)
                delta = q - p

                # Curvature: gamma*sin(theta) * perp(delta)
                perp = _perpendicular(delta, rng)
                gamma = rng.uniform(0.0, 0.1)
                theta = rng.uniform(0.0, 2 * np.pi)
                curvature = gamma * np.sin(theta) * perp

                x_syn = p + alpha * delta + curvature
                synthetic.append(x_syn)

        if len(synthetic) == 0:
            print("AGSS: No synthetic samples generated, returning original data.")
            return X.copy(), y.copy()

        X_syn = np.array(synthetic)
        y_syn = np.ones(len(X_syn), dtype=y.dtype)

        X_out = np.vstack([X, X_syn]).astype(X.dtype)
        y_out = np.concatenate([y, y_syn])
        return X_out, y_out


def _perpendicular(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a unit vector perpendicular to v using random orthonormalization."""
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        r = rng.standard_normal(v.shape)
        return r / (np.linalg.norm(r) + 1e-12)

    v_hat = v / norm_v
    r = rng.standard_normal(v.shape)
    r -= np.dot(r, v_hat) * v_hat
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        return r
    return r / r_norm
