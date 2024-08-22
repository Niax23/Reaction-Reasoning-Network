import torch
from torch_geometric.nn import global_mean_pool
from ..molecular_embedding.GATconv import SelfLoopGATConv as MyGATConv
from ..molecular_embedding.sparse_backBone import SparseEdgeUpdateLayer
from typing import Any, Dict, List, Tuple, Optional, Union

class RxnNetworkGNN(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4, embedding_dim: int = 64,
        dropout: float = 0.7, negative_slope: float = 0.2,
        n_class: Optional[int] = None
    ):
        super(RxnNetworkGNN, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = num_layers, num_heads
        self.dropout_fun = torch.nn.Dropout(dropout)
        assert embedding_dim % num_heads == 0, \
            'The embedding dim should be evenly divided by num_heads'
        for layer in range(self.num_layers):
            self.convs.append(MyGATConv(
                in_channels=embedding_dim, heads=num_heads,
                out_channels=embedding_dim // num_heads,
                negative_slope=negative_slope,
                dropout=dropout, edge_dim=embedding_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                embedding_dim, embedding_dim
            ))


    def forward(self, G) -> torch.Tensor:
        node_feats = G.x
        edge_feats = G.edge_attr
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats, edge_index=G.edge_index,
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats

            edge_feats = torch.relu(self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=G.edge_index
            ))

        return node_feats, edge_feats
