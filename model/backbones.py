import torch
from .layers import SelfLoopGATConv as MyGATConv
from .layers import SparseEdgeUpdateLayer
import numpy as np

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GATBase(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4, embedding_dim: int = 64,
        dropout: float = 0.7, negative_slope: float = 0.2,
    ):
        super(GATBase, self).__init__()
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
        self.atom_encoder = AtomEncoder(embedding_dim)
        self.bond_encoder = BondEncoder(embedding_dim)
        self.mask_embedding = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, G) -> torch.Tensor:
        node_feats = self.atom_encoder(G.x)
        edge_feats = self.bond_encoder(G.edge_attr)
        if G.get('pesudo_mask', None) is not None:
            node_feats[G.pesudo_mask] = self.mask_embedding
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


class RxnNetworkGNN(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4, embedding_dim: int = 64,
        dropout: float = 0.7, negative_slope: float = 0.2,
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

    def forward(self, node_feats, edge_feats, edge_index) -> torch.Tensor:
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats, edge_index=edge_index,
            ))
            node_feats = self.dropout_fun(torch.relu(conv_res)) + node_feats

            edge_feats = torch.relu(self.edge_update[layer](
                edge_feats=edge_feats, node_feats=node_feats,
                edge_index=edge_index
            ))

        return node_feats, edge_feats
