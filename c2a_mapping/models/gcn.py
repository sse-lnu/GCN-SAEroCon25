"""
Relational GCN — exact match to C2A_pipeline.ipynb RelationalGNN with encoder='gcn'.

mode='hetero' : one GCNConv per relation type (HeteroConv)
mode='homo'   : merge all file->file edges into one adjacency, single GCNConv
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv


class RelationalGCN(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.0, mode="hetero"):
        super().__init__()
        self.mode    = mode
        self.dropout = dropout
        _, edge_types = metadata

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            if mode == "hetero":
                conv_dict = {
                    et: GCNConv(in_ch, hidden_channels)
                    for et in edge_types
                }
                self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            else:
                self.convs.append(GCNConv(in_ch, hidden_channels))

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def _merge_edges(self, x_dict, edge_index_dict):
        parts = [ei for et, ei in edge_index_dict.items()
                 if et[0] == "file" and et[2] == "file"]
        if not parts:
            return torch.zeros(2, 0, dtype=torch.long, device=x_dict["file"].device)
        return torch.unique(torch.cat(parts, dim=1), dim=1)

    def encode(self, x_dict, edge_index_dict):
        dev = next(self.parameters()).device
        x_dict          = {k: v.to(dev) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(dev) for k, v in edge_index_dict.items()}

        if self.mode == "hetero":
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                          for k, v in x_dict.items()}
            return x_dict["file"]
        else:
            x          = x_dict["file"]
            edge_index = self._merge_edges(x_dict, edge_index_dict)
            for conv in self.convs:
                x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
            return x

    def forward(self, x_dict, edge_index_dict):
        return self.lin(self.encode(x_dict, edge_index_dict))
