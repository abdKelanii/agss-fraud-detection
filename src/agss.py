"""
Adaptive Generative Synthetic Sampling (AGSS)

Implements both the oversampling and undersampling variants from:
  Nama & Sharmila Banu, "Credit Card Fraud Detection Using Deep Learning
  Techniques and Handling Unbalanced Class Distributions With AGSS",
  IEEE Access, Jan 2026.

Oversampling algorithm (Section IV.A):
  1. Extract minority class samples
  2. Cluster with DBSCAN (eps=0.8, min_samples=3)
  3. Within each dense cluster, generate synthetic points via KNN + curvature interpolation:
       x_syn = p + alpha*(q-p) + gamma*sin(theta)*perp(q-p)
  4. Combine with original data

Enhancement over paper: when eps=0.8 finds <2 clusters (common in high-dim data),
use _find_multi_cluster_eps to search for the largest eps that still yields >=2 clusters,
rather than collapsing everything into one cluster (which degrades AGSS to ROS).
Within each cluster, base point selection is weighted by local density so denser
sub-regions generate proportionally more synthetic samples.

Undersampling variant (AGSSUnderSampler):
  1. Apply DBSCAN to the majority class
  2. Remove noisy/outlier majority samples (DBSCAN label == -1)
  3. Randomly subsample remaining clustered majority points down to n_minority
  4. Combine with original minority class
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
        density_weighted: bool = True,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.adaptive_eps = adaptive_eps
        # Weight base point selection within each cluster by local k-NN density.
        # Makes AGSS genuinely density-aware even within a single large cluster.
        self.density_weighted = density_weighted

    def _find_multi_cluster_eps(self, X_min: np.ndarray) -> tuple:
        """
        Find the largest eps that yields >= 2 clusters by scanning percentiles
        from p20 down to p5 of k-NN distances. Starting large → smallest eps with
        max cluster coverage (fewest noise points) while preserving density structure.
        Falls back to p25 if no multi-cluster eps is found.
        """
        k = min(self.n_neighbors + 1, len(X_min) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        dists, _ = nbrs.kneighbors(X_min)
        knn_dists = dists[:, k]

        for pct in range(20, 4, -1):  # 20, 19, ..., 5
            eps = float(np.percentile(knn_dists, pct))
            labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(X_min)
            n_clusters = len([c for c in np.unique(labels) if c != -1])
            if n_clusters >= 2:
                print(f"AGSS: multi-cluster eps at p{pct}={eps:.3f} → {n_clusters} clusters")
                return eps, labels

        # No multi-cluster eps found — fall back to p25
        eps = float(np.percentile(knn_dists, 25))
        labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(X_min)
        n_clusters = len([c for c in np.unique(labels) if c != -1])
        print(f"AGSS: no multi-cluster eps found; using p25={eps:.3f} → {n_clusters} clusters")
        return eps, labels

    def _density_weights(self, cluster_pts: np.ndarray) -> np.ndarray:
        """Local density weights: inverse of mean k-NN distance per point."""
        k = min(self.n_neighbors, len(cluster_pts) - 1)
        if k < 1:
            return np.ones(len(cluster_pts)) / len(cluster_pts)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(cluster_pts)
        dists, _ = nbrs.kneighbors(cluster_pts)
        density = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-8)
        return density / density.sum()

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
            # Search for eps that yields >= 2 clusters instead of collapsing to one
            eps, labels = self._find_multi_cluster_eps(X_min)
            unique_clusters = [c for c in np.unique(labels) if c != -1]

        # Step 2: Build cluster dictionary
        if len(unique_clusters) == 0:
            # All points marked as noise — treat entire minority set as one cluster
            clusters = {0: X_min}
        else:
            clusters = {c: X_min[labels == c] for c in unique_clusters}

        # Step 3: Generate synthetic samples
        synthetic = []
        per_cluster = ceil(n_synthetic / len(clusters))

        for cluster_pts in clusters.values():
            if len(cluster_pts) < 2:
                continue

            k = min(self.n_neighbors, len(cluster_pts) - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(cluster_pts)

            weights = self._density_weights(cluster_pts) if self.density_weighted else None

            n_to_generate = min(per_cluster, n_synthetic - len(synthetic))
            if n_to_generate <= 0:
                break

            for _ in range(n_to_generate):
                idx_p = rng.choice(len(cluster_pts), p=weights)
                p = cluster_pts[idx_p]
                dists, neighbor_idxs = nbrs.kneighbors([p])
                neighbor_pool = neighbor_idxs[0][1:]
                if len(neighbor_pool) == 0:
                    continue
                idx_q = rng.choice(neighbor_pool)
                q = cluster_pts[idx_q]

                alpha = rng.uniform(0.0, 1.0)
                delta = q - p
                perp = _perpendicular(delta, rng)
                gamma = rng.uniform(0.0, 0.1)
                theta = rng.uniform(0.0, 2 * np.pi)
                x_syn = p + alpha * delta + gamma * np.sin(theta) * perp
                synthetic.append(x_syn)

        if len(synthetic) == 0:
            print("AGSS: No synthetic samples generated, returning original data.")
            return X.copy(), y.copy()

        X_syn = np.array(synthetic)
        y_syn = np.ones(len(X_syn), dtype=y.dtype)
        return np.vstack([X, X_syn]).astype(X.dtype), np.concatenate([y, y_syn])


class AGSSUnderSampler:
    """
    AGSS undersampling: cluster the majority class with DBSCAN, discard noise
    points, then subsample clustered majority points down to n_minority.

    For large majority classes (> dbscan_max_samples), pre-subsamples before
    DBSCAN to keep runtime manageable — captures the same cluster structure
    without running DBSCAN on hundreds of thousands of points.
    """

    def __init__(self, eps=0.8, min_samples=3, random_state=42,
                 adaptive_eps=True, dbscan_max_samples=10_000):
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.adaptive_eps = adaptive_eps
        self.dbscan_max_samples = dbscan_max_samples

    def _choose_eps(self, X_maj):
        k = min(self.min_samples, len(X_maj) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_maj)
        dists, _ = nbrs.kneighbors(X_maj)
        return float(np.percentile(dists[:, k], 25))

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_maj, X_min = X[y == 0], X[y == 1]
        n_target = len(X_min)

        if len(X_maj) > self.dbscan_max_samples:
            idx_pre = rng.choice(len(X_maj), size=self.dbscan_max_samples, replace=False)
            X_maj_sample = X_maj[idx_pre]
        else:
            X_maj_sample = X_maj

        eps = self.eps
        labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(X_maj_sample)

        if len(set(labels) - {-1}) < 1 and self.adaptive_eps:
            eps = self._choose_eps(X_maj_sample)
            print(f"AGSSUnder: adaptive eps={eps:.3f}")
            labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(X_maj_sample)

        X_maj_clean = X_maj_sample[labels != -1]

        if len(X_maj_clean) == 0:
            idx = rng.choice(len(X_maj_sample), size=n_target, replace=False)
            X_maj_clean = X_maj_sample[idx]
        elif len(X_maj_clean) > n_target:
            idx = rng.choice(len(X_maj_clean), size=n_target, replace=False)
            X_maj_clean = X_maj_clean[idx]

        X_out = np.vstack([X_maj_clean, X_min]).astype(X.dtype)
        y_out = np.concatenate([
            np.zeros(len(X_maj_clean), dtype=y.dtype),
            np.ones(len(X_min), dtype=y.dtype),
        ])
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
