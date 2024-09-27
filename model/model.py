import torch
import torch.nn as nn
import math


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
        self, gnn1, gnn2, PE, molecule_dim, net_dim, heads, dropout,
        dec_layers, n_words, with_type=False, ntypes=None
    ):
        super(MyModel, self).__init__()
        self.pos_enc = PE
        self.gnn1 = gnn1  # 第一个GNN模型
        self.gnn2 = gnn2  # 第二个GNN模型
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, net_dim))
        self.edge_emb = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim))
        })
        self.pooler = torch.nn.MultiheadAttention(
            embed_dim=net_dim, kdim=molecule_dim, vdim=molecule_dim,
            num_heads=heads, batch_first=True, dropout=dropout
        )
        t_layer = torch.nn.TransformerEncoderLayer(
            net_dim, heads, dim_feedforward=net_dim << 1,
            batch_first=True, dropout=dropout
        )
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(net_dim, net_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(net_dim, n_words)
        )
        self.decoder = torch.nn.TransformerEncoder(t_layer, dec_layers)
        self.reaction_init = torch.nn.Parameter(torch.randn(net_dim))
        self.molecule_dim = molecule_dim
        self.net_dim = net_dim
        self.word_emb = torch.nn.Embedding(n_words, net_dim)
        self.with_type = with_type
        if self.with_type:
            assert ntypes is not None, "require type numbers"
            self.type_embs = torch.nn.Embedding(ntypes, net_dim)

    def encode(
        self, molecules, molecule_mask, reaction_mask, required_ids,
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
        x_feat[molecule_mask] = molecule_feats.squeeze(1)
        x_feat[reaction_mask] = self.reaction_init
        edge_feats = torch.stack([self.edge_emb[x] for x in edge_types], dim=0)
        net_x, _ = self.gnn2(x_feat, edge_feats, edge_index)
        return net_x[required_ids]

    def decode(
        self, memory, labels, attn_mask, key_padding_mask=None, seq_types=None
    ):
        x_input = self.word_emb(labels)
        if self.with_type:
            assert seq_types is not None, "Require type inputs"
            x_input += self.type_embs(seq_types)
        seq_input = torch.cat([memory.unsqueeze(dim=1), x_input], dim=1)
        seq_output = self.decoder(
            src=self.pos_enc(seq_input), mask=attn_mask,
            src_key_padding_mask=key_padding_mask
        )

        return self.out_layer(seq_output)

    def forward(
        self, molecules, molecule_mask, reaction_mask, required_ids,
        edge_index, edge_types, labels, attn_mask,
        key_padding_mask=None, seq_types=None
    ):
        reaction_embs = self.encode(
            molecules, molecule_mask, reaction_mask,
            required_ids, edge_index, edge_types
        )

        result = self.decode(
            reaction_embs, labels, attn_mask,
            key_padding_mask=key_padding_mask, seq_types=seq_types
        )
        return result


class PretrainedModel(nn.Module):
    def __init__(
        self, gnn2, PE, net_dim, heads, dropout, dec_layers,  n_words,
        mol_dim=300, with_type=False, ntypes=None, init_rxn=False
    ):
        super(PretrainedModel, self).__init__()
        self.pos_enc = PE
        self.gnn2 = gnn2
        self.init_rxn = init_rxn
        self.edge_emb = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim))
        })
        t_layer = torch.nn.TransformerEncoderLayer(
            net_dim, heads, dim_feedforward=net_dim << 1,
            batch_first=True, dropout=dropout
        )
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(net_dim, net_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(net_dim, n_words)
        )
        self.linear = torch.nn.Linear(mol_dim, net_dim)
        self.decoder = torch.nn.TransformerEncoder(t_layer, dec_layers)
        if init_rxn:
            self.rxn_linear = torch.nn.Linear(mol_dim * 2, net_dim)
        else:
            self.reaction_init = torch.nn.Parameter(torch.randn(net_dim))
        self.net_dim = net_dim
        self.word_emb = torch.nn.Embedding(n_words, net_dim)
        self.with_type = with_type
        if self.with_type:
            assert ntypes is not None, "require type numbers"
            self.type_embs = torch.nn.Embedding(ntypes, net_dim)

    def encode(
        self, mole_embs, molecule_ids, rxn_ids, required_ids,
        edge_index, edge_types, n_nodes, rxn_embs=None
    ):
        x_feat = torch.zeros((n_nodes, self.net_dim)).to(mole_embs)
        x_feat[molecule_ids] = self.linear(mole_embs)
        if self.init_rxn:
            assert rxn_embs is not None, "Require Init Emb input for rxns"
            x_feat[rxn_ids] = self.rxn_linear(rxn_embs)
        else:
            x_feat[rxn_ids] = self.reaction_init
        edge_feats = torch.stack([self.edge_emb[x] for x in edge_types], dim=0)
        net_x, _ = self.gnn2(x_feat, edge_feats, edge_index)
        return net_x[required_ids]

    def decode(
        self, memory, labels, attn_mask, key_padding_mask=None, seq_types=None
    ):
        x_input = self.word_emb(labels)
        if self.with_type:
            assert seq_types is not None, "Require type inputs"
            x_input += self.type_embs(seq_types)
        seq_input = torch.cat([memory.unsqueeze(dim=1), x_input], dim=1)
        seq_output = self.decoder(
            src=self.pos_enc(seq_input), mask=attn_mask,
            src_key_padding_mask=key_padding_mask
        )

        return self.out_layer(seq_output)

    def forward(
        self, mole_embs, molecule_ids, rxn_ids, required_ids, edge_index,
        edge_types, labels, attn_mask, n_nodes, rxn_embs=None,
        key_padding_mask=None, seq_types=None,
    ):
        reaction_embs = self.encode(
            mole_embs, molecule_ids, rxn_ids, required_ids,
            edge_index, edge_types, n_nodes, rxn_embs
        )

        result = self.decode(
            reaction_embs, labels, attn_mask,
            key_padding_mask=key_padding_mask, seq_types=seq_types
        )
        return result


class SemiModel(nn.Module):
    def __init__(
        self, gnn2, PE, net_dim, heads, dropout, dec_layers,  n_words,
        mol_dim=300, with_type=False, ntypes=None, init_rxn=False
    ):
        super(SemiModel, self).__init__()
        self.pos_enc = PE
        self.gnn2 = gnn2
        self.init_rxn = init_rxn
        self.edge_emb = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim))
        })
        t_layer = torch.nn.TransformerEncoderLayer(
            net_dim, heads, dim_feedforward=net_dim << 1,
            batch_first=True, dropout=dropout
        )
        self.edge_linear = torch.nn.Linear(mol_dim + net_dim, net_dim)
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(net_dim, net_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(net_dim, n_words)
        )
        self.linear = torch.nn.Linear(mol_dim, net_dim)
        self.decoder = torch.nn.TransformerEncoder(t_layer, dec_layers)
        if init_rxn:
            self.rxn_linear = torch.nn.Linear(mol_dim * 2, net_dim)
        else:
            self.reaction_init = torch.nn.Parameter(torch.randn(net_dim))
        self.net_dim = net_dim
        self.word_emb = torch.nn.Embedding(n_words, net_dim)
        self.with_type = with_type
        if self.with_type:
            assert ntypes is not None, "require type numbers"
            self.type_embs = torch.nn.Embedding(ntypes, net_dim)

    def encode(
        self, mole_embs, molecule_ids, rxn_ids, required_ids,
        edge_index, edge_types, edge_feats, n_nodes, rxn_embs=None
    ):
        x_feat = torch.zeros((n_nodes, self.net_dim)).to(mole_embs)
        x_feat[molecule_ids] = self.linear(mole_embs)
        if self.init_rxn:
            assert rxn_embs is not None, "Require Init Emb input for rxns"
            x_feat[rxn_ids] = self.rxn_linear(rxn_embs)
        else:
            x_feat[rxn_ids] = self.reaction_init
        edge_tf = torch.stack([self.edge_emb[x] for x in edge_types], dim=0)
        edge_feats = self.edge_linear(torch.cat([edge_tf, edge_feats], dim=-1))
        net_x, _ = self.gnn2(x_feat, edge_feats, edge_index)
        return net_x[required_ids]

    def decode(
        self, memory, labels, attn_mask, key_padding_mask=None, seq_types=None
    ):
        x_input = self.word_emb(labels)
        if self.with_type:
            assert seq_types is not None, "Require type inputs"
            x_input += self.type_embs(seq_types)
        seq_input = torch.cat([memory.unsqueeze(dim=1), x_input], dim=1)
        seq_output = self.decoder(
            src=self.pos_enc(seq_input), mask=attn_mask,
            src_key_padding_mask=key_padding_mask
        )

        return self.out_layer(seq_output)

    def forward(
        self, mole_embs, molecule_ids, rxn_ids, required_ids, edge_index,
        edge_types, edge_feats, labels, attn_mask, n_nodes, rxn_embs=None,
        key_padding_mask=None, seq_types=None,
    ):
        reaction_embs = self.encode(
            mole_embs, molecule_ids, rxn_ids, required_ids,
            edge_index, edge_types, edge_feats, n_nodes, rxn_embs
        )

        result = self.decode(
            reaction_embs, labels, attn_mask,
            key_padding_mask=key_padding_mask, seq_types=seq_types
        )
        return result


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 2000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        token_len = token_embedding.shape[1]
        return self.dropout(token_embedding + self.pos_embedding[:token_len])


class SepSemiModel(nn.Module):
    def __init__(
        self, gnn2, PE, net_dim, heads, dropout, dec_layers,  n_words,
        mol_dim=300, with_type=False, ntypes=None, init_rxn=False
    ):
        super(SepSemiModel, self).__init__()
        self.pos_enc = PE
        self.gnn2 = gnn2
        self.init_rxn = init_rxn
        self.edge_emb = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim))
        })
        self.node_type = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim))
        })
        t_layer = torch.nn.TransformerEncoderLayer(
            net_dim, heads, dim_feedforward=net_dim << 1,
            batch_first=True, dropout=dropout
        )
        self.edge_linear = torch.nn.Linear(mol_dim + net_dim, net_dim)
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(net_dim, net_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(net_dim, n_words)
        )
        self.linear = torch.nn.Linear(mol_dim + net_dim, net_dim)
        self.decoder = torch.nn.TransformerEncoder(t_layer, dec_layers)
        if init_rxn:
            self.rxn_linear = torch.nn.Linear(mol_dim * 2, net_dim)
        else:
            self.reaction_init = torch.nn.Parameter(torch.randn(net_dim))
        self.net_dim = net_dim
        self.word_emb = torch.nn.Embedding(n_words, net_dim)
        self.with_type = with_type
        if self.with_type:
            assert ntypes is not None, "require type numbers"
            self.type_embs = torch.nn.Embedding(ntypes, net_dim)

    def encode(
        self, mole_embs, mts, molecule_ids, rxn_ids, required_ids,
        edge_index, edge_types, edge_feats, n_nodes, rxn_embs=None
    ):
        x_feat = torch.zeros((n_nodes, self.net_dim)).to(mole_embs)
        x_temb = torch.stack([self.node_type[x] for x in mts], dim=0)
        mole_embs = torch.cat([mole_embs, x_temb], dim=-1)
        x_feat[molecule_ids] = self.linear(mole_embs)
        if self.init_rxn:
            assert rxn_embs is not None, "Require Init Emb input for rxns"
            x_feat[rxn_ids] = self.rxn_linear(rxn_embs)
        else:
            x_feat[rxn_ids] = self.reaction_init
        edge_tf = torch.stack([self.edge_emb[x] for x in edge_types], dim=0)
        edge_feats = self.edge_linear(torch.cat([edge_tf, edge_feats], dim=-1))
        net_x, _ = self.gnn2(x_feat, edge_feats, edge_index)
        return net_x[required_ids]

    def decode(
        self, memory, labels, attn_mask, key_padding_mask=None, seq_types=None
    ):
        x_input = self.word_emb(labels)
        if self.with_type:
            assert seq_types is not None, "Require type inputs"
            x_input += self.type_embs(seq_types)
        seq_input = torch.cat([memory.unsqueeze(dim=1), x_input], dim=1)
        seq_output = self.decoder(
            src=self.pos_enc(seq_input), mask=attn_mask,
            src_key_padding_mask=key_padding_mask
        )

        return self.out_layer(seq_output)

    def forward(
        self, mole_embs, mts, molecule_ids, rxn_ids, required_ids, edge_index,
        edge_types, edge_feats, labels, attn_mask, n_nodes, rxn_embs=None,
        key_padding_mask=None, seq_types=None,
    ):
        reaction_embs = self.encode(
            mole_embs, mts, molecule_ids, rxn_ids, required_ids,
            edge_index, edge_types, edge_feats, n_nodes, rxn_embs
        )

        result = self.decode(
            reaction_embs, labels, attn_mask,
            key_padding_mask=key_padding_mask, seq_types=seq_types
        )
        return result


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 2000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        token_len = token_embedding.shape[1]
        return self.dropout(token_embedding + self.pos_embedding[:token_len])
