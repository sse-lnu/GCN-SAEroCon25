import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_channels, out_channels, num_layers, dropout=0.1, embed_dim=None):
        """
        Graph Convolutional Network (GCN) model.

        Args:
            input_dim (int): Number of input features per node.
            hidden_channels (int): Number of hidden units per layer.
            out_channels (int): Number of output features (final embedding size).
            num_layers (int): Number of GCN layers.
            dropout (float, optional): Dropout probability applied between layers. Default is 0.1.
            embed_dim (int, optional): If provided, applies a linear transformation to project 
                                       input features to a higher-dimensional space before passing them to GCN layers.
        """
        super(GCN, self).__init__()

        # Feature embedding (optional)
        self.feature_embedder = (
            nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) if embed_dim else None
        )
        
        input_channels = embed_dim if embed_dim else input_dim
        self.convs = nn.ModuleList(
            [GCNConv(input_channels, hidden_channels)] +  # First layer
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 2)] +  # Middle layers
            [GCNConv(hidden_channels, out_channels)]  # Last layer
        )
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, input_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format of shape [2, num_edges].

        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, out_channels].
        """
        device = next(self.parameters()).device
        x, edge_index = x.to(device), edge_index.to(device)
        if self.feature_embedder:
            x = self.feature_embedder(x)

        for conv in self.convs[:-1]: 
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training) 
        x = self.convs[-1](x, edge_index)  
        
        return x


####################
class RGCN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, relations, dropout=0.2, embed_dim=None, input_dim_dict=None):
        """
        Heterogeneous GCN model.
        Args:
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Output feature size.
            num_layers (int): Number of graph convolution layers.
            relations (list of str): List of relation types for heterogeneous graph.
            dropout (float): Dropout probability.
            embed_dim (int, optional): Embedding dimension for node features.
            input_dim_dict (dict, optional): Dictionary mapping node types to input feature dimensions.
        """
        super(RGCN, self).__init__()

        # Feature embedding layers for different node types (if embed_dim is used)
        self.feature_embedder = (
            nn.ModuleDict({
                node_type: nn.Sequential(
                    nn.Linear(input_dim, embed_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                for node_type, input_dim in input_dim_dict.items()
            }) if embed_dim and input_dim_dict else None
        )
        self.convs = nn.ModuleList([
            HeteroConv({
                ('entity', relation, 'entity'): GCNConv(
                    embed_dim if i == 0 else hidden_channels,
                    hidden_channels
                )
                for relation in relations
            })
            for i in range(num_layers)
        ])
        
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass of HeteroGCN.
        Args:
            x_dict (dict): Dictionary of node feature tensors for each node type.
            edge_index_dict (dict): Dictionary of edge index tensors for each relation.

        Returns:
            torch.Tensor: Output node embeddings.
        """
        device = next(self.parameters()).device  # Get the model’s device dynamically
        x_dict = {key: x.to(device) for key, x in x_dict.items()}
        edge_index_dict = {key: ei.to(device) for key, ei in edge_index_dict.items()}

        # Apply feature embedding if available
        if self.feature_embedder:
            x_dict = {key: self.feature_embedder[key](x) for key, x in x_dict.items()}

        # Apply graph convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        return self.lin(x_dict['entity'])
