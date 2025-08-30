# GLOW: Learning Global-Local Multi-Scale Node Embeddings with Random Walks and Landmark-Guided Optimization

GLOW is a Python-based graph embedding framework that generates low-dimensional vector representations for nodes in a graph. It implements a hybrid approach that captures both local neighborhood structure and global graph topology.

The core idea is to combine:
1.  **Local Information**: Captured by running random walks and using a Word2Vec model, similar to DeepWalk.
2.  **Global Information**: Captured by approximating global node positioning by measuring the distance from all nodes to a set of strategically selected "landmark" nodes. 
The model then optimizes the node embeddings to ensure that their similarity in the embedding space reflects the global, landmark-based structural similarity of the graph.

## Key Features

*   **Hybrid Embedding Strategy**: Combines local and global graph features for rich representations.
*   **Landmark-based Global Similarity**: Uses an efficient method to encode global graph structure.
*   **Disconnected Graph Support**: Handles graphs with multiple connected components by introducing a "fictive" node to bridge them.
*   **Parallel Processing**: Leverages `joblib` for speeding up random walk generation and distance computations.
*   **Configurable**: All key hyperparameters are managed through a `config.yaml` file.
*   **PyTorch-based**: Built on PyTorch, enabling GPU acceleration (CUDA) and Apple Silicon (MPS) for the optimization process.

## Requirements

*   Python 3.8+
*   PyTorch
*   NetworkX
*   Gensim
*   NumPy
*   Pandas
*   tqdm
*   Joblib
*   PyYAML

You can install the required packages using pip. It is recommended to create a `requirements.txt` file.

**requirements.txt:**
```
torch
networkx
gensim
numpy
pandas
tqdm
joblib
```

Install with:
```sh
pip install -r requirements.txt
```

## Usage

### 1. Project Structure

Make sure your project has the following directory structure. The script will save intermediate and final results here.

```
/
├── GLOW.py
├── load_datasets.py
├── config.yaml
└── data/
    └── <dataset_name>/
```

### 2. Configuration

Create a `config.yaml` file in the root directory to control the experiment. Here is an example configuration:

```ini
[experiment]
dataset_name = cora

[paths]
data = ./data

[training_params]
num_walks = 80
walk_length = 40
embedding_dim = 128
num_landmarks = 50
num_candidate_landmarks = 100
learning_rate = 0.01
num_epochs = 2000
patience = 50
workers = -1
alpha = 0.0
```

### 3. Running the Model

Once the configuration is set, you can run the script from your terminal:

```sh
python GLOW.py
```

The other scripts are optional and can be run independently, upgraded or simply ignored at your discretion: 

link_prediction.py: to experiment with link prediction
node_classiication.py: to experiment with node classification
