"""
Get a code-to-architecture mapping without requiring every file to have a
known ground-truth Module. Every file with a known Module is used as seed
training data (no held-out split — there's nothing to hold out for, since
this doesn't evaluate accuracy); every other file gets a predicted_module.

No F1/precision/recall is computed anywhere in this path, since there's no
guarantee the full dataset has ground truth to score against — that's what
experiments/run_experiments.py and baselines/run_nba.py are for.

Usage:
    python run_mapping.py --dataset bash --model gat
    python run_mapping.py --dataset bash --model gcn --data-dir path/to/data
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import datasets, features_w2v, loc_features, graph
from data_pipeline.label_utils import encode_labels
from models.gcn import RelationalGCN
from models.gat import RelationalGAT
from models.mlp import MLP
from training.self_train import self_train

SCRIPT_DIR  = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "experiments" / "config.json"
RESULTS_DIR = SCRIPT_DIR / "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_mapping(stem, model_type, data_dir=None, cache_dir=None):
    cfg       = json.loads(CONFIG_PATH.read_text())
    model_cfg = cfg["model_params"][model_type.upper()]
    train_cfg = cfg["training"]

    file_df, file_dep = datasets.load_file_level(stem, data_dir)
    w2v = features_w2v.build_w2v_features(file_df, stem=stem, cache_dir=cache_dir, data_dir=data_dir)
    loc = loc_features.build_loc_features(file_df)
    x   = torch.cat([loc, w2v], dim=1)
    in_channels = x.shape[1]

    model_type = model_type.lower()
    if model_type == "mlp":
        data = HeteroData()
        data["file"].x = x
        y, label_encoder = encode_labels(file_df["Module"])
        data["file"].y = y
        data["file", "none", "file"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        data, label_encoder, _ = graph.build_graph(file_df, file_dep, x, directed=True, reverse_relations=False)

    num_classes = len(label_encoder.classes_)
    seed_idx    = torch.where(data["file"].y != -1)[0]
    if seed_idx.numel() == 0:
        raise ValueError(f"No labeled files found for '{stem}' — need at least one "
                          f"known Module to seed training.")

    data     = data.to(DEVICE)
    seed_idx = seed_idx.to(DEVICE)

    if model_type == "gcn":
        model = RelationalGCN(data.metadata(), in_channels, model_cfg["hidden"], num_classes,
                               num_layers=model_cfg["num_layers"], dropout=model_cfg["dropout"], mode="hetero")
    elif model_type == "gat":
        model = RelationalGAT(data.metadata(), in_channels, model_cfg["hidden"], num_classes,
                               num_layers=model_cfg["num_layers"], heads=model_cfg["heads"],
                               dropout=model_cfg["dropout"], mode="hetero")
    else:
        model = MLP(in_channels=in_channels, hidden_channels=model_cfg["hidden"],
                    out_channels=num_classes, dropout=model_cfg["dropout"])

    result = self_train(
        data, model, seed_idx, lr=train_cfg["lr"], device=DEVICE,
        threshold=train_cfg["threshold"], warmup_epochs=train_cfg["warmup_epochs"],
        self_train_rounds=train_cfg["self_train_rounds"], self_train_epochs=train_cfg["self_train_epochs"],
        return_predictions=True,
    )

    rows = [
        {"File": file_df["File"].iloc[p["node_idx"]],
         "predicted_module": label_encoder.inverse_transform([p["forced_pred_label"]])[0]}
        for p in result["predictions"]
    ]

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{stem}_mapping.csv"
    df.to_csv(out_path, index=False)
    print(f"Mapped {len(df)} files ({int(seed_idx.numel())} used as seed) -> {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Get a code-to-architecture mapping, no evaluation.")
    parser.add_argument("--dataset", required=True, help="dataset stem, e.g. bash")
    parser.add_argument("--model", default="gat", choices=["gcn", "gat", "mlp"])
    parser.add_argument("--data-dir", default=None, help="defaults to c2a_mapping/data")
    parser.add_argument("--cache-dir", default=None, help="defaults to c2a_mapping/cache/w2v")
    args = parser.parse_args()
    run_mapping(args.dataset, args.model, data_dir=args.data_dir, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
