# C2A Graph — Code-to-Architecture Mapping

Semi-supervised node classification pipeline for mapping source files to architectural modules using Graph Neural Networks (GCN, GAT) and an MLP baseline, with an iterative self-training loop. Includes a Naive Bayes (NBA) baseline for comparison.

This repository is the extended version of the SAEroCon 2025 paper "Graph Convolutional Networks for Mapping Source Code Entities to Architectural Modules." The original `GCNCodeMap/` pipeline had a self-training bug (pseudo-labeled entities were retrained on their true labels instead of the model's own predictions, leaking test labels into training) and only ever reported metrics on the confidently-mapped subset, never the full test set. This extension replaces that pipeline with a corrected, expanded one (`c2a_mapping/`) covering more systems, GAT in addition to GCN, and edge-direction/depth ablations.

---

## Repository Layout

```
c2a_mapping/
├── baselines/                       # NBA (Naive Bayes) baseline
├── data/                            # Sample datasets (Bash, JabRef, Teammates)
├── data_pipeline/                   # Feature extraction and graph construction
├── evaluation/                      # Metric helpers
├── experiments/                     # config.json + GNN experiment runner
├── models/                          # GCN, GAT, MLP definitions
├── training/                        # Self-training loop
├── fixed_splits_experiment.ipynb    # Main experiment notebook (GCN/GAT/MLP/NBA, shared splits)
└── requirements.txt
```

---

## Installation

```bash
cd c2a_mapping/
pip install -r requirements.txt
```

---

## Data

The `c2a_mapping/data/` directory contains three sample systems (Bash, JabRef, Teammates). To add more systems, place their CSV files there:

```
c2a_mapping/data/
├── {stem}.csv         # columns: File, Entity, Module, Member_Name
└── {stem}_deps.csv    # columns: Source_File, Target_File, Dependency_Type, Dependency_Count
```

---

## Running GNN Experiments (GCN · GAT · MLP)

```bash
cd c2a_mapping/

# All datasets, default config
python experiments/run_experiments.py

# Specific ablation
python experiments/run_experiments.py --ablation homo_directed

# Single dataset, quick test
python experiments/run_experiments.py --dataset bash --runs 5
```

Results are written per model/ablation to `results/{gcn,gat}_{hetero,homo,undirected,reversed}_results.csv` and `results/mlp_results.csv`.

`fixed_splits_experiment.ipynb` runs the same models against a shared set of pre-generated train/test splits, so every model in a given run sees the exact same labeled seed set — use this for paired comparisons across models.

### Ablations

| Name | Mode | Directed |
|------|------|----------|
| `hetero_directed` *(default)* | one conv per relation type | yes |
| `homo_directed` | merged adjacency | yes |
| `hetero_undirected` | one conv per relation type | no |
| `hetero_reversed` | one conv per relation type, edges flipped | yes |
| `hetero_directed_1layer` | one conv per relation type, 1 message-passing layer | yes |

---

## Running the NBA Baseline

```bash
cd c2a_mapping/

python baselines/run_nba.py                        # all datasets
python baselines/run_nba.py --datasets bash,jabref  # specific datasets
python baselines/run_nba.py --runs 10               # quick test
```

Results are written to `results/nba_results.csv` (separate from GNN results). `f1_macro`/`f1_micro`/`precision_macro`/`recall_macro` are computed on the subset of entities the model mapped with confidence; the same metrics on the full test set (including forced predictions for everything below the confidence threshold) are prefixed `full_`.
