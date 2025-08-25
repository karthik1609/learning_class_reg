from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main() -> None:
    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42)

    model = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels)
    centers = model.cluster_centers_
    print(f"Silhouette score: {sil:.3f}")
    print(f"Cluster centers shape: {centers.shape}")


if __name__ == "__main__":
    main()


