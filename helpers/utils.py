import time
import os
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
from typing import List, Callable, Any

def timeit(func: Callable) -> Callable:
    """
    A decorator that wraps a function and prints the time it took to execute.

    Args:
        func (Callable): The function to be timed.

    Returns:
        Callable: The wrapped function with timing logic.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

@timeit
def load_word2vec_model(binary: bool = True) -> KeyedVectors:
    """
    Load a Word2Vec model from a predefined file path. If the model file is not found
    or if loading from the file fails, load a pre-trained 'word2vec-google-news-300' model from Gensim.

    Args:
        binary (bool): Indicates whether the model file is in binary format (default is True).

    Returns:
        KeyedVectors: A Gensim KeyedVectors object containing the word embeddings.
    """
    assets_dir = 'assets'
    bin_file = 'GoogleNews-vectors-negative300.bin'
    filepath = os.path.join(assets_dir, bin_file)

    if os.path.exists(assets_dir) and os.path.exists(filepath):
        try:
            model = KeyedVectors.load_word2vec_format(filepath, binary=binary)
            print(f"Model successfully loaded from file: {filepath}")
        except Exception as e:
            print(f"Error loading model from file '{filepath}': {e}")
            print("Loading pre-trained 'word2vec-google-news-300' model from Gensim...")
            model = api.load('word2vec-google-news-300')
    else:
        print(f"File or directory not found. Loading pre-trained 'word2vec-google-news-300' model from Gensim...")
        model = api.load('word2vec-google-news-300')

    return model

def compute_document_topic_similarity(document_words: List[str], topic_words: List[str], model: KeyedVectors) -> float:
    """
    Compute the cosine similarity between a document and a topic based on their word embeddings.

    Args:
        document_words (List[str]): List of words representing the document.
        topic_words (List[str]): List of words representing the topic.
        model (KeyedVectors): Gensim KeyedVectors model where each word maps to its corresponding word vector.

    Returns:
        float: Cosine similarity between the document vector and the topic vector.
    """

    def get_valid_vectors(words: List[str]) -> List[np.ndarray]:
        """
        Retrieve word vectors for words that are present in the model and do not contain NaN values.

        Args:
            words (List[str]): List of words for which vectors are to be retrieved.

        Returns:
            List[np.ndarray]: List of valid word vectors.
        """
        vectors = []
        for word in words:
            if word in model:
                vector = model[word]
                if not np.isnan(vector).any():  
                    vectors.append(vector)
        return vectors

    doc_vectors = get_valid_vectors(document_words)
    topic_vectors = get_valid_vectors(topic_words)

    if not doc_vectors or not topic_vectors:
        return 0.0  

    doc_vector = np.mean(doc_vectors, axis=0)
    topic_vector = np.mean(topic_vectors, axis=0)

    similarity = np.dot(doc_vector, topic_vector) / \
        (np.linalg.norm(doc_vector) * np.linalg.norm(topic_vector))

    return similarity