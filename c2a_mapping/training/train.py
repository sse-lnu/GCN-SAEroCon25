"""
Top-level train() — wires features -> graph -> model -> self_train.

For GCN/GAT: builds HeteroData graph, runs relational message passing.
For MLP:     skips graph build; wraps features in minimal HeteroData so
             the same self_train loop works unchanged.
"""
import copy

import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

from data_pipeline import datasets, features_w2v, loc_features, graph
from data_pipeline.seeds import generate_split, assert_hard_constraints
from models.gcn import RelationalGCN
from models.gat import RelationalGAT
from models.mlp import MLP
from training.self_train import self_train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_features(stem, cache_dir=None):
    file_df, file_dep = datasets.load_file_level(stem)
    w2v = features_w2v.build_w2v_features(file_df, stem=stem, cache_dir=cache_dir)
    loc = loc_features.build_loc_features(file_df)
    x   = torch.cat([loc, w2v], dim=1)
    return file_df, file_dep, x


def build_model(data, model_type, mode, in_channels, num_classes, model_cfg):
    """Instantiate the right model using per-model params from config."""
    model_type = model_type.lower()
    hidden     = model_cfg.get("hidden", 256)
    dropout    = model_cfg.get("dropout", 0.01)

    if model_type == "gcn":
        return RelationalGCN(
            data.metadata(), in_channels, hidden, num_classes,
            num_layers=model_cfg.get("num_layers", 2),
            dropout=dropout,
            mode=mode,
        )
    elif model_type == "gat":
        return RelationalGAT(
            data.metadata(), in_channels, hidden, num_classes,
            num_layers=model_cfg.get("num_layers", 2),
            heads=model_cfg.get("heads", 4),
            dropout=dropout,
            mode=mode,
        )
    else:
        return MLP(in_channels=in_channels, hidden_channels=hidden,
                   out_channels=num_classes, dropout=dropout)


def train(stem, model_type, ablation, model_cfg=None, train_cfg=None,
          n_runs=1, start_run=1, cache_dir=None, return_predictions=False):
    """
    Run n_runs repetitions for one (stem, model_type, ablation) combo.

    ablation keys : mode, directed, reverse_relations
    model_cfg     : per-model params (hidden, num_layers, heads, dropout)
    train_cfg     : training params (lr, threshold, warmup_epochs, self_train_rounds, self_train_epochs)
    """
    if model_cfg  is None: model_cfg  = {}
    if train_cfg  is None: train_cfg  = {}

    model_type  = model_type.lower()
    mode        = ablation.get("mode", "hetero")
    directed    = ablation.get("directed", True)
    reverse_rel = ablation.get("reverse_relations", False)

    lr                = train_cfg.get("lr", 0.001)
    threshold         = train_cfg.get("threshold", 0.95)
    warmup_epochs     = train_cfg.get("warmup_epochs", 100)
    self_train_rounds = train_cfg.get("self_train_rounds", 4)
    self_train_epochs = train_cfg.get("self_train_epochs", 30)

    file_df, file_dep, x = build_features(stem, cache_dir=cache_dir)
    num_classes = file_df["Module"].nunique()
    ratio       = datasets.split_ratio(stem)
    in_channels = x.shape[1]

    if model_type in ("gcn", "gat"):
        data, label_encoder, _ = graph.build_graph(
            file_df, file_dep, x,
            directed=directed, reverse_relations=reverse_rel,
        )
    else:
        data = HeteroData()
        data["file"].x = x
        le = LabelEncoder()
        data["file"].y = torch.tensor(le.fit_transform(file_df["Module"]), dtype=torch.long)
        label_encoder  = le
        # Dummy empty edge so data.edge_index_dict works (MLP ignores edges)
        data["file", "none", "file"].edge_index = torch.zeros(2, 0, dtype=torch.long)

    rows = []
    for run in range(n_runs):
        run_id   = start_run + run
        data_dev = copy.deepcopy(data).to(DEVICE)
        seed_idx, _ = generate_split(data_dev["file"].y, split_ratio=ratio)
        assert_hard_constraints(data_dev["file"].y, seed_idx)

        model = build_model(data_dev, model_type, mode, in_channels, num_classes, model_cfg)

        result = self_train(
            data_dev, model, seed_idx.to(DEVICE), lr=lr, device=DEVICE,
            threshold=threshold, warmup_epochs=warmup_epochs,
            self_train_rounds=self_train_rounds, self_train_epochs=self_train_epochs,
            return_predictions=return_predictions,
        )
        result.pop("metrics_history", None)
        result.update({
            "dataset": datasets.display_name(stem), "data_name": stem,
            "model": model_type, "mode": mode,
            "directed": directed, "reverse_relations": reverse_rel,
            "run": run_id, "run_id": run_id, "split_ratio": ratio,
        })
        rows.append(result)
        print(f"{datasets.display_name(stem)} [{model_type}/{mode}] run {run_id}: "
              f"f1_macro={result.get('final_f1_macro', 'N/A')}")

    return rows, label_encoder, file_df
