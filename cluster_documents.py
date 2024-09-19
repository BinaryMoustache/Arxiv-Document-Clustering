import os
import json
import argparse
import sys
import numpy as np
from helpers.text_preprocess import TextProcessor
from models.embeddings import CreateEmbeddings
from helpers.utils import load_word2vec_model
from models.bayesian_optimization import bayesian_search
from models.models import umap_reduction, hdbscan_clustering, TopicModel
from typing import Dict, Optional


def cluster_abstracts(
    data_path: str,
    param_grid: Dict[str, any],
    model_name: Optional[str] = None,
    n_iter: int = 10
) -> None:
    """
    Processes abstracts from a JSON file, generates embeddings, performs clustering, and saves the results.

    Args:
        data_path (str): Path to the JSON file containing the data.
        param_grid (Dict[str, any]): Dictionary of hyperparameters for Bayesian optimization.
        model_name (Optional[str]): Name of the SentenceTransformer model to use. If None, a default model is used.
        n_iter (int): Number of iterations for Bayesian optimization. Defaults to 10.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Error reading data file. Please ensure it's a valid JSON file.") from e

    processor = TextProcessor()
    abstracts = np.array([processor.process_text(
        item["abstract"].strip()) for item in data])

    try:
        if model_name:
            sentence_encoder = CreateEmbeddings(model_name=model_name)
        else:
            sentence_encoder = CreateEmbeddings()
        embeddings = sentence_encoder.fit_transform(abstracts)
    except Exception as e:
        print(f"Error occurred while generating embeddings: {e}")
        if model_name:
            print(f"Check if the model name '{model_name}' is correct.")
        print("Find a suitable model on 'https://huggingface.co/models'")
        sys.exit(1)

    w2v = load_word2vec_model()

    try:
        print(f"\nRunning Bayesian optimization for {n_iter + 5} iterations.")
        params = bayesian_search(
            param_grid, abstracts, embeddings, w2v, n_iter=n_iter)
    except Exception as e:
        print("No topics were found during an iteration of Bayesian optimization. Please review and adjust your parameter ranges.")
        print(f"Details: {e}")
        sys.exit(1)

    n_components = int(params["n_components"])
    n_neighbors = int(params["n_neighbors"])
    min_cluster_size = int(params["min_cluster_size"])

    dim_reduction = umap_reduction(n_components, n_neighbors)
    clustering_model = hdbscan_clustering(min_cluster_size)
    model = TopicModel(abstracts, embeddings, dim_reduction, clustering_model)
    topics, _ = model.fit_transform()
    print(f"There are {len(set(topics))} unique topics extracted")

    documents = model.get_results(topics, w2v)
    documents["coherence"] = [float(item) for item in documents["coherence"]]
    documents["abstract"] = [item["abstract"] for item in data]
    documents["title"] = [item["title"] for item in data]

    del documents["text"]

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, 'documents.json')

    with open(results_file_path, 'w') as f:
        json.dump(documents, f, indent=4)

    print(f"\nResults saved to '{results_file_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster and model topics using UMAP and HDBSCAN.")

    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the JSON file containing the data.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the configuration JSON file.")

    try:
        args = parser.parse_args()
    except SystemExit:
        print("\nExample usage:")
        print("python cluster_documents.py --data_path data/filtered_papers_from_year.json --config /path/to/config.json")
        parser.print_help()
        sys.exit(1)

    try:
        with open(args.config, 'r') as config_file:
            config = json.load(config_file)
            param_grid = config.get("param_grid", {})
            model_name = config.get("model_name")
            n_iter = config.get("n_iter", 10)
    except Exception as e:
        raise ValueError(f"Error loading config file: {args.config}. Ensure it's a valid JSON file.") from e

    cluster_abstracts(args.data_path, param_grid,
                      model_name=model_name, n_iter=n_iter)
