# Arxiv-Document-Clustering

This project clusters document abstracts by first converting them into embeddings using a SentenceTransformer model. It then uses UMAP and HDBSCAN to group the documents. The best settings for these methods are found through Bayesian optimization. To measure how well each topic represents its documents, the project calculates cosine similarity between the topic's and document's word embeddings using a Word2Vec model. Finally, it saves the clustering results and coherence scores to a JSON file

## Download and Install

To download the dataset, visit this [Kaggle page](https://www.kaggle.com/datasets/Cornell-University/arxiv).

Download the dataset from Kaggle.
1) Create a folder named ````data```` in your project directory.
2) Extract the downloaded contents into the data folder.

To install the dependencies, follow these steps:
1) Create a new Conda environment and activate it:
````
conda create --name myenv python=3.12
conda activate myenv
````
2) Install the required packages using ````pip````:
````
pip install -r requirements.txt

````

**Note:**

This project uses a pre-trained Word2Vec (W2V) model (Google News, 300 dimensions). To speed up the process, you can manually download the model from [here](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300).

1) Create a folder named assets in your project directory.
2) Place the downloaded model file in the assets folder.

If you don't manually download the model, the script will automatically download and load it during execution.

## Usage

### Grab data for analysis
First, filter the papers for the year of interest. Optionally, you can limit the number of documents to speed up computation. Run the following command:
```
python filter_papers_for_analysis.py data/arxiv_metadata.json 2024 --max_docs 1000
```

- ```data/arxiv_metadata.json```: Path to the dataset.
- ````2024````: The year of interest (change this to your desired year).
- ````--max_docs 1000```` (optional): Randomly selects up to 1000 papers from the specified year.

This command will filter papers with titles and abstracts from the year 2024 and later, and will randomly select up to 1000 papers if the ````--max_docs```` option is used.

### Extract Topics 
To run the script for topic extraction, use the following command:

```
python cluster_documents.py --data_path data/filtered_papers_from_{year}.json --config configs/your_config_file.json
```
- ````--data_path data/filtered_papers_from_{year}.json````: Specify the path to the JSON file containing the filtered papers from the previous step.
- ````--config configs/your_config_file.json````: Specify the path to your configuration file.
Ensure that both the data_path and config arguments point to the correct files before running the script.

Here’s a breakdown of the configuration file you’ll need to provide in the configs folder:
````
{
  "model_name": null,
  "param_grid": {
    "n_neighbors": [3, 18],
    "min_cluster_size": [5, 40],
    "n_components": [3, 12]
  },
  "n_iter": 100
}
````

- ````model_name````:
The pre-trained BERT model from Hugging Face to use for topic extraction. You can specify a model like "all-MiniLM-L6-v2". If left as null, the script may use a default model.
- ````param_grid````: A dictionary that specifies the parameter ranges for UMAP and HDBSCAN. These ranges will be explored during Bayesian optimization to identify the optimal configuration:
    - ````n_components````: UMAP parameter for the number of dimensions to reduce the data to.
    - ````n_neighbors````: UMAP parameter for controlling the number of neighbors to consider for each point.
    - ````min_cluster_size````: HDBSCAN parameter for setting the minimum number of samples required to form a cluster.

- ````n_iter````: The number of iterations for the Bayesian optimization process, which will search for the best combination of parameters in param_grid. The script will run 100 iterations in this case.



