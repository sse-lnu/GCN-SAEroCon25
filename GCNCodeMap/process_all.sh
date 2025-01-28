#!/bin/bash

RAW_DATA_DIR="data/raw"
PROCESSED_DATA_DIR="data/processed"
PYTHON_SCRIPT="preprocess.py"

mkdir -p $PROCESSED_DATA_DIR

for json_file in $RAW_DATA_DIR/*.json; do
  base_name=$(basename "$json_file" .json)
  labels_file="$RAW_DATA_DIR/${base_name}_labels.txt"

  # Check if the labels file exists
  if [[ -f "$labels_file" ]]; then
    echo "Processing dataset: $base_name"
    
    # Run the Python script for this dataset
    python $PYTHON_SCRIPT --labels "$labels_file" --json "$json_file" --name "$base_name"
  else
    echo "Labels file not found for dataset: $base_name. Skipping..."
  fi
done
