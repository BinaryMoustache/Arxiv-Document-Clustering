import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List



class CreateEmbeddings:
    def __init__(self, model_name: str = "thenlper/gte-base"):
        """
        Initialize the CreateEmbeddings class with a specified model name.

        Args:
            model_name (str, optional): Name of the SentenceTransformer model to load. Defaults to "thenlper/gte-base".
        """
        self.model_name = model_name
        self.model = SentenceTransformer(
            self.model_name, trust_remote_code=True)

    def fit_transform(self, data: List[str]) -> np.ndarray:
        """
        Encode input data into embeddings using the initialized SentenceTransformer model.

        Args:
            data (List[str]): Input data to encode into embeddings.

        Returns:
            np.ndarray: Array of embeddings (float arrays) corresponding to the input data.
        """
        # Encode the data into embeddings using the SentenceTransformer model
        embeddings = self.model.encode(data, show_progress_bar=True)
        return embeddings
