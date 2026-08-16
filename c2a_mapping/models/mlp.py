"""
Plain 2-layer MLP baseline — no graph, uses node features only.

Forward signature matches GCN/GAT (x_dict, edge_index_dict) so the same
self-training loop works for all three models. edge_index_dict is ignored.
"""
import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict=None):
        x = x_dict["file"].to(next(self.parameters()).device)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=self.training)
        return self.fc2(x)
