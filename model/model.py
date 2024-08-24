import torch
from .backbones import GATBase, RxnNetworkGNN


def graph2batch(
    node_feat: torch.Tensor, batch_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_node = batch_mask.shape
    answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    answer = answer.to(node_feat.device)
    answer[batch_mask] = node_feat
    return answer


class MyModel(nn.Module):
    def __init__(
        self, gnn1_args, gnn2_args, molecule_dim, net_dim,
        heads, dropout, dec_layers, n_words
    ):
        super(MyModel, self).__init__()
        self.gnn1 = GATBase(**gnn1_args)  # 第一个GNN模型
        self.gnn2 = RxnNetworkGNN(**gnn2_args)  # 第二个GNN模型
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, net_dim))
        self.edge_emb = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim))
        })
        self.pooler = torch.nn.MultiheadAttention(
            embed_dim=net_dim, kdim=molecule_dim, vdim=molecule_dim,
            num_heads=pool_heads, batch_first=True, dropout=dropout
        )
        t_layer = torch.nn.TransformerEncoderLayer(
            net_dim, heads, dim_feedforward=net_dim << 1,
            batch_first=True, dropout=dropout
        )
        self.decoder = torch.nn.TransformerEncoder(t_layer, dec_layers)
        self.reaction_init = torch.nn.Parameter(torch.randn(net_dim))
        self.molecule_dim = molecule_dim
        self.net_dim = net_dim
        self.word_emb = torch.nn.Embedding(n_words, net_dim)

    def encode(
        self, molecules, molecule_mask, reaction_mask, required_mask,
        edge_index, edge_types,
    ):
        x, _ = self.gnn1(molecules)
        memory = graph2batch(x, molecules.batch_mask)
        pool_keys = self.pool_keys.repeat(memory.shape[0], 1, 1)

        molecule_feats, _ = self.pooler(
            query=pool_keys, key=memory, value=memory,
            key_padding_mask=torch.logical_not(molecules.batch_mask)
        )

        x_feat_shape = (molecule_mask.shape[0], self.net_dim)
        x_feat = torch.zeros(x_feat_shape).to(molecule_feats)
        x_feat[molecule_mask] = molecule_feats
        x_feat[reaction_mask] = self.reaction_init
        edge_feats = torch.stack([self.edge_emb[x] for x in edge_types], dim=0)
        net_x, _ = self.gnn2(x_feat, edge_feats, edge_index)
        return net_x[required_mask]

    def decode(self, memory, labels, attn_mask, key_padding_mask=None):
        seq_input = [memory.unsqueeze(dim=1), self.word_emb(labels)]
        seq_input = torch.cat(seq_input, dim=1)
        seq_output = self.decoder(
            src=seq_input, mask=attn_mask,
            key_padding_mask=key_padding_mask
        )

        return seq_output

    def forward(
        self, molecules, molecule_mask, reaction_mask, required_mask,
        edge_index, edge_types, labels, attn_mask, key_padding_mask=None
    ):
        reaction_embs = self.encode(
            molecules, molecule_mask, reaction_mask,
            required_mask, edge_index, edge_types
        )

        result = self.decode(
            reaction_embs, labels, attn_mask,
            key_padding_mask=key_padding_mask
        )
        return result
