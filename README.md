# GCN-CodeMapping

Mapping source code entities to architectural modules is a critical step for automating static architecture compliance checking. This repository provides preprocessing code for datasets used in the paper *"Graph Convolutional Networks for Mapping Source Code Entities to Architectural Modules"*.

## Preprocessing

To preprocess the data for a specific dataset, you can use the following command in the terminal (bash or similar shell):

```bash
python preprocess.py --labels data/raw/<dataset_name>_labels.txt --json data/raw/<dataset_name>.json --name <dataset_name>
