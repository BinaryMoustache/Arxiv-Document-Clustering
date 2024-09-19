import numpy as np
from bayes_opt import BayesianOptimization
from models.models import umap_reduction, hdbscan_clustering, TopicModel
from typing import Dict, List, Tuple, Any


def objective(documents: List[str], embeddings: np.ndarray, n_neighbors: int, min_cluster_size: int, n_components: int, word2vec_model) -> float:
    """
    Objective function for Bayesian Optimization.

    Args:
        documents (List[str]): List of document texts.
        embeddings (np.ndarray): Embeddings data for clustering.
        n_neighbors (int): Number of neighbors for UMAP.
        min_cluster_size (int): Minimum cluster size for HDBSCAN.
        n_components (int): Number of components for UMAP.
        word2vec_model: Pre-trained Word2Vec model for coherence evaluation.

    Returns:
        float: Coherence score to be maximized.
    """
    n_neighbors = int(n_neighbors)
    min_cluster_size = int(min_cluster_size)
    n_components = int(n_components)

    umap_model = umap_reduction(n_components, n_neighbors)
    hdbscan_model = hdbscan_clustering(min_cluster_size)

    model = TopicModel(documents, embeddings, umap_model, hdbscan_model)

    topics, _ = model.fit_transform()
    unique_topics = set(topics)

    if len(unique_topics) <5:
        return -np.inf

    scores, _ = model.evaluate_coherence(topics, word2vec_model)
    return scores


def bayesian_search(param_grid: Dict[str, Tuple[float, float]], documents: List[str], embeddings: np.ndarray, word2vec_model, n_iter: int = 10) -> Dict[str, Any]:
    """
    Perform Bayesian Optimization to find the best hyperparameters.

    Args:
        documents (List[str]): List of documents.
        embeddings (np.ndarray): Embeddings data for clustering.
        param_grid (Dict[str, Tuple[float, float]]): Dictionary defining the parameter bounds for Bayesian Optimization.
        word2vec_model: Pre-trained Word2Vec model for coherence evaluation.
        n_iter (int, optional): Number of iterations for Bayesian Optimization. Defaults to 10.

    Returns:
        Dict[str, Any]: Dictionary containing the best hyperparameters found.
    """
    pbounds = param_grid

    optimizer = BayesianOptimization(
        f=lambda n_neighbors, min_cluster_size, n_components: objective(
            documents, embeddings, n_neighbors, min_cluster_size, n_components, word2vec_model
        ),
        pbounds=pbounds,
        random_state=512,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=n_iter,
    )

    print("Best Parameters:", optimizer.max)

    best_params = optimizer.max['params']
    return best_params
