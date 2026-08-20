"""
HeteroData graph construction.

Ablation parameters:
  directed          — keep dependency edges directed (True) or symmetrize (False)
  reverse_relations — use dependencies as-is (False) or flip src<->dst (True)
"""
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData

from data_pipeline.label_utils import encode_labels


def build_graph(file_df, file_dep, x,
                directed=True,
                reverse_relations=False):
    """
    Returns (HeteroData, label_encoder, relations).

    Args:
        file_df          : file-level table with File/Entity/Module columns
        file_dep         : dependency table with Source_ID/Target_ID/Dependency_Type
        x                : precomputed feature tensor, row-aligned to file_df
        directed         : if False, symmetrize all dependency edges
        reverse_relations: if True, flip src<->dst in all dependency edges
    """
    data = HeteroData()
    data["file"].x = x

    y, label_encoder = encode_labels(file_df["Module"])
    data["file"].y = y

    relations = sorted(file_dep["Dependency_Type"].dropna().unique().tolist())
    edge_dict = defaultdict(list)
    for _, row in file_dep.iterrows():
        src_id, tgt_id = row["Source_ID"], row["Target_ID"]
        if reverse_relations:
            src_id, tgt_id = tgt_id, src_id
        edge_dict[row["Dependency_Type"]].append((src_id, tgt_id))

    for dep_type, edges in edge_dict.items():
        src, tgt = zip(*edges)
        src_l, tgt_l = list(src), list(tgt)
        if directed:
            data["file", str(dep_type), "file"].edge_index = torch.tensor(
                [src_l, tgt_l], dtype=torch.long
            )
        else:
            data["file", str(dep_type), "file"].edge_index = torch.tensor(
                [src_l + tgt_l, tgt_l + src_l], dtype=torch.long
            )

    return data, label_encoder, relations
