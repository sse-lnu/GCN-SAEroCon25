# GCN-CodeMapping

This repository provides a framework for mapping source code entities to architectural modules using Graph Convolutional Networks (GCNs). This is particularly useful for automating static architecture compliance checking. The repository contains preprocessing code for datasets and the ability to train and evaluate Graph Convolutional Networks (GCNs), Relational Graph Convolutional Networks (RGCNs), and Naïve Bayes models.

## Project Structure

GCNCodeMap/ ├── data/ # Contains processed and raw datasets │ ├── processed/ │ └── raw/ ├── graphs/ # Graph construction code ├── models/ # Models like GCN, RGCN ├── notebooks/ # Jupyter notebooks (for experimentation, tutorials) ├── preprocessing/ # Code for data preprocessing ├── results/ # Directory to save results from experiments ├── utils/ # Utility functions 

## Setup

### Installing Dependencies

1. Clone the repository:

   git clone https://github.com/yourusername/GCN-CodeMapping.git  
   cd GCN-CodeMapping

2. Install the dependencies using the requirements.txt:

```bash
   pip install -r requirements.txt
```
---

## Preprocessing
To preprocess the data, use the preprocess.py script with the following command:

```bash
   python preprocess.py --labels data/raw/<dataset_name>_labels.txt --json data/raw/<dataset_name>.json --name <dataset_name>
```

## Running Experiments

To train and evaluate the models, use the main.py script with the following arguments:

### Command for Running GCN/RGCN Experiments:

### Arguments:

- `--model_type`: Type of model to use. Choose from `gcn`, `rgcn` (Relational GCN), or `nb` (Naïve Bayes).
- `--data_name`: Name of the dataset (e.g., `ant`, `prom`).
- `--data_path`: Path to the preprocessed data (CSV format).
- `--dependencies_path`: Path to the dependency CSV file.
- `--epochs`: Number of epochs for training (default is 50).
- `--lr`: Learning rate (default is 0.001).
- `--max_norm`: Max norm for gradient clipping (default is 5).
- `--lambda_t`: Threshold scaling factor for iterative learning (can be a float, int, or lambda function).
- `--split_ratio`: Train-test split ratio (default is 0.05).
- `--q_threshold`: Quantile threshold for centrality filtering (default is 0.3).
- `--hidden_channels`: Number of hidden channels for the GCN model.
- `--dropout`: Dropout rate (default is 0.2).
- `--gcn_embed_dim`: Embedding dimension for GCN (default is 128).
- `--num_layers`: Number of hidden layers (default is 1).
- `--num_runs`: Number of experiments to run (default is 2).
- `--verbose`: Set verbosity for the training and evaluation process.

### Example command for RGCN:

```bash
python main.py --model_type <model_type> --data_name <dataset_name> --data_path data/processed/<dataset_name>.csv --dependencies_path data/processed/dependencies_<dataset_name>.csv --epochs 50 --lr 0.001 --max_norm 5 --lambda_t 0.8 --split_ratio 0.05 --q_threshold 0.3 --hidden_channels 16 --dropout 0.2 --gcn_embed_dim 128 --num_layers 1 --num_runs 2 --verbose True
```
---

## Results

The results of the experiments will be saved in the `results/` directory. This includes metrics like F1 score, Precision, Recall, and other evaluation metrics for each model and dataset. The results will be saved in CSV format for easy analysis.
