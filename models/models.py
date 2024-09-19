import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from helpers.utils import compute_document_topic_similarity
from typing import Dict, List, Any, Tuple



def umap_reduction(n_components: int, n_neighbors: int) -> UMAP:
    """
    Create and configure a UMAP (Uniform Manifold Approximation and Projection) model for dimensionality reduction.

    Args:
        n_components (int): Number of dimensions to reduce the data to.
        n_neighbors (int): Number of neighbors used for local approximations in UMAP.

    Returns:
        UMAP: Configured UMAP object.
    """
    return UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.05,
        n_components=n_components,
        metric='cosine',
        random_state=512,
        n_jobs=1
    )


def hdbscan_clustering(min_cluster_size: int) -> HDBSCAN:
    """
    Create and configure an HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) model for clustering.

    Args:
        min_cluster_size (int): Minimum size of clusters to be formed.

    Returns:
        HDBSCAN: Configured HDBSCAN object.
    """
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        gen_min_span_tree=True,
        metric='euclidean'
    )


class TopicModel:
    def __init__(self, documents: List[str], embeddings: np.ndarray, dim_reduction_model: UMAP, clustering_model: HDBSCAN):
        """
        Initialize the TopicModel with documents, embeddings, and model configurations.

        Args:
            documents (List[str]): List of document texts.
            embeddings (np.ndarray): Array of document embeddings.
            dim_reduction_model (UMAP): UMAP model for dimensionality reduction.
            clustering_model (HDBSCAN): HDBSCAN model for clustering.
        """
        self.documents = documents
        self.embeddings = embeddings
        self.dim_reduction_model = dim_reduction_model
        self.clustering_model = clustering_model
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer()

        self.topic_model = BERTopic(
            umap_model=self.dim_reduction_model,
            hdbscan_model=self.clustering_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            nr_topics='auto'
        )

    def fit_transform(self) -> Tuple[List[str], BERTopic]:
        """
        Fit the BERTopic model to the documents and embeddings, and transform the data to get topics.

        Returns:
            Tuple[List[str], BERTopic]: Topics and the fitted BERTopic model.
        """
        topics, _ = self.topic_model.fit_transform(
            self.documents, self.embeddings)
        return topics, self.topic_model

    def evaluate_coherence(self, topics: List[str], word2vec_model) -> Tuple[float, List[float]]:
        """
        Evaluate topic coherence using the Word2Vec model.

        Args:
            topics (List[str]): List of topic labels.
            word2vec_model: Word2Vec model used for coherence evaluation.

        Returns:
            Tuple[float, List[float]]: Average coherence score and a list of coherence scores for each document.
        """
        topic_words = self.topic_model.get_topic_info()
        topic_words_dict = {row['Topic']: row['Name'].split(
            "_")[1:] for _, row in topic_words.iterrows()}

        vectorizer = self.topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()
        tokens = [analyzer(doc) for doc in self.documents]
        coherence_scores = []
        for idx, doc_words in enumerate(tokens):
            assigned_topic = topics[idx]
            if assigned_topic in topic_words_dict:
                topic_words = topic_words_dict[assigned_topic]
                similarity_score = compute_document_topic_similarity(
                    doc_words, topic_words, word2vec_model)
                coherence_scores.append(similarity_score)

        avg_score = np.mean(coherence_scores)
        return avg_score, coherence_scores

    def get_results(self, topics: List[str], word2vec_model) -> Dict[str, Any]:
        """
        Generate results including topic labels, topic names, and coherence scores.

        Args:
            topics (List[str]): List of topic labels.
            word2vec_model: Word2Vec model used for coherence evaluation.

        Returns:
            Dict[str, Any]: Dictionary containing document texts, topic labels, topic names, and coherence scores.
        """
        _, coherences = self.evaluate_coherence(topics, word2vec_model)

        topic_words = self.topic_model.get_topic_info()
        topic_words_dict = {row['Topic']: row['Name'].split(
            "_")[1:] for index, row in topic_words.iterrows()}

        results = {
            "text": self.documents,
            "topic": ["_".join(topic_words_dict.get(label, [])) for label in topics],
            "label": topics,
            "coherence": coherences
        }

        return results
