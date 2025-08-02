import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from typing import Any


def find_cluster_centroids(embeddings, max_k=10, sample_size=1000) -> Any:
    """Estimate centroids for a set of embeddings.

    The original implementation ran standard :class:`KMeans` for every
    possible ``k`` up to ``max_k`` on the full dataset.  For large numbers of
    embeddings this became very expensive.  This version speeds things up by
    using :class:`MiniBatchKMeans` and evaluating candidate ``k`` values on a
    random subset of the data using the silhouette score.

    Args:
        embeddings: Array of feature vectors to cluster.
        max_k: Maximum number of clusters to evaluate.
        sample_size: Number of embeddings to sample when estimating the best
            ``k``.  The full dataset is still used to compute the final
            centroids once ``k`` is determined.

    Returns:
        Array of cluster centroids for the selected ``k``.
    """

    embeddings = np.array(embeddings)
    if len(embeddings) == 0:
        return []

    # Sample a subset of embeddings to estimate the optimal number of clusters
    n_samples = embeddings.shape[0]
    sample_size = min(sample_size, n_samples)
    if n_samples > sample_size:
        subset_idx = np.random.choice(n_samples, sample_size, replace=False)
        subset = embeddings[subset_idx]
    else:
        subset = embeddings

    best_k = 1
    best_score = -1.0
    for k in range(2, max_k + 1):
        mbk = MiniBatchKMeans(n_clusters=k, random_state=0)
        labels = mbk.fit_predict(subset)

        # Silhouette score requires at least 2 clusters
        if len(set(labels)) > 1:
            score = silhouette_score(subset, labels)
            if score > best_score:
                best_score = score
                best_k = k

    final_kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=0)
    final_kmeans.fit(embeddings)
    return final_kmeans.cluster_centers_

def find_closest_centroid(centroids: list, normed_face_embedding) -> list:
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)
        
        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        return None