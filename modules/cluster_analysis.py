import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from typing import Any, Iterable


def find_cluster_centroids(
    embeddings: Iterable,
    max_k: int = 10,
    sample_size: int = 1000
) -> np.ndarray:
    """Estimate centroids for a set of embeddings using MiniBatchKMeans.

    This function determines the optimal number of clusters (k) by evaluating
    the silhouette score on a random subset of the data. It then computes
    the final centroids on the full dataset using the optimal k.

    Args:
        embeddings: Iterable of embedding vectors to cluster.
        max_k: Maximum number of clusters to evaluate.
        sample_size: Number of embeddings to sample for estimating k.

    Returns:
        A NumPy array of cluster centroids. Returns an empty array for empty input.
    """
    embeddings = np.asarray(list(embeddings))
    n_samples = embeddings.shape[0]

    if n_samples == 0:
        return np.array([])
    # If there's only one embedding, it's its own centroid.
    if n_samples == 1:
        return embeddings

    # Use a subset of embeddings to estimate the optimal number of clusters
    if n_samples > sample_size:
        # Use a fixed seed for reproducible sampling
        rng = np.random.default_rng(seed=42)
        subset_idx = rng.choice(n_samples, sample_size, replace=False)
        subset = embeddings[subset_idx]
    else:
        subset = embeddings

    best_k = 1
    best_score = -1.0
    # Ensure max_k is not greater than the number of samples available for clustering
    k_upper_bound = min(max_k, subset.shape[0] -1) # K must be < n_samples

    for k in range(2, k_upper_bound + 1):
        mbk = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = mbk.fit_predict(subset)

        # Silhouette score requires at least 2 distinct cluster labels
        if len(np.unique(labels)) > 1:
            score = silhouette_score(subset, labels)
            if score > best_score:
                best_score = score
                best_k = k

    # If the best k is 1, return the mean of all embeddings as the single centroid
    if best_k == 1:
        return np.mean(embeddings, axis=0, keepdims=True)

    # Fit on the full dataset with the optimal k
    final_kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init='auto')
    final_kmeans.fit(embeddings)
    return final_kmeans.cluster_centers_


def find_closest_centroid(centroids: list, normed_face_embedding) -> list:
    """Compute the index and value of the closest centroid to a face embedding.

    Computes the dot product of the centroids and the face embedding to
    find the most similar centroid. Returns the index and value of the
    closest centroid. If either input is an empty list, returns None.

    Args:
        centroids: A list of k cluster centroids.
        normed_face_embedding: A normalized face embedding vector.

    Returns:
        A tuple of the index and value of the closest centroid, or None.
    """
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)
        
        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        return None