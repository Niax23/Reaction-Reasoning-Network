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


class FullModel(nn.Module):
    def __init__(
        self, gnn1, gnn2, PE, net_dim, heads, dropout, dec_layers,
        n_words, mol_dim, with_type=False, ntypes=None, init_rxn=False
    ):
        super(FullModel, self).__init__()
        self.pos_enc = PE
        self.gnn2 = gnn2
        self.gnn1 = gnn1
        self.init_rxn = init_rxn
        self.net_dim = net_dim
        self.mole_dim = mol_dim
        self.edge_emb = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim)),
        })
        self.semi_init = torch.nn.Parameter(torch.randn(1, 1, net_dim))
        self.pooler = torch.nn.MultiheadAttention(
            embed_dim=net_dim, kdim=mol_dim, vdim=mol_dim,
            num_heads=heads, batch_first=True, dropout=dropout
        )
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
        self.decoder = torch.nn.TransformerEncoder(t_layer, dec_layers)
        if init_rxn:
            self.rxn_linear = torch.nn.Linear(mol_dim * 2, net_dim)
        else:
            self.reaction_init = torch.nn.Parameter(torch.randn(net_dim))
        self.word_emb = torch.nn.Embedding(n_words, net_dim)
        self.with_type = with_type
        if self.with_type:
            assert ntypes is not None, "require type numbers"
            self.type_embs = torch.nn.Embedding(ntypes, net_dim)

    def encode(
        self, mole_graphs, mts, molecule_ids, rxn_ids, required_ids,
        edge_index, edge_types, semi_graphs, semi_keys, semi_key2idxs,
        n_nodes, reactant_pairs=None, product_pairs=None
    ):
        x_keys = torch.stack([self.node_type[x] for x in mts], dim=0)
        x_keys = x_keys.unsqueeze(dim=1)
        mole_feats, _ = self.gnn1(mole_graphs)
        mole_feats = graph2batch(mole_feats, mole_graphs.batch_mask)

        mole_embs, _ = self.pooler(
            query=x_keys, key=mole_feats, value=mole_feats,
            key_padding_mask=torch.logical_not(mole_graphs.batch_mask)
        )

        x_feat = torch.zeros((n_nodes, self.net_dim)).to(mole_embs)
        x_feat[molecule_ids] = mole_embs.squeeze(dim=1)
        if self.init_rxn:
            assert reactant_pairs is not None and product_pairs is not None, \
                "Require reaction comp mapper for rxns embedding generation"
            rxn_embs = average_mole_for_rxn(
                x_feat, n_nodes, rxn_ids, reactant_pairs, product_pairs
            )
            x_feat[rxn_ids] = self.rxn_linear(rxn_embs)
        else:
            x_feat[rxn_ids] = self.reaction_init
        edge_tf = torch.stack([self.edge_emb[x] for x in edge_types], dim=0)
        semi_n_embs, _ = self.gnn1(semi_graphs)
        semi_b_embs = graph2batch(semi_n_embs, semi_graphs.batch_mask)
        semi_qry = self.semi_init.repeat(semi_b_embs.shape[0], 1, 1)
        semi_feats, _ = self.pooler(
            query=semi_qry, key=semi_b_embs, value=semi_b_embs,
            key_padding_mask=torch.logical_not(semi_graphs.batch_mask)
        )
        semi_feats = semi_feats.squeeze(dim=1)

        # print('[model]', semi_keys[0])
        # print('[model]', any(type(x) == list for x in semi_keys))

        semi_edges = [semi_feats[semi_key2idxs[t]] for t in semi_keys]
        semi_egx = torch.zeros((edge_tf.shape[0], self.mole_dim)).to(edge_tf)
        semi_rec_mask = [x == 'reactant' for x in edge_types]
        semi_egx[semi_rec_mask] = torch.stack(semi_edges, dim=0)
        edge_feats = torch.cat([semi_egx, edge_tf], dim=-1)
        net_x, _ = self.gnn2(x_feat, self.edge_linear(edge_feats), edge_index)
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
        self, mole_graphs, mts, molecule_ids, rxn_ids, required_ids,
        edge_index, edge_types, semi_graphs, semi_keys, semi_key2idxs,
        n_nodes, labels, attn_mask, reactant_pairs=None, product_pairs=None,
        key_padding_mask=None, seq_types=None,
    ):
        reaction_embs = self.encode(
            mole_graphs, mts, molecule_ids, rxn_ids, required_ids,
            edge_index, edge_types, semi_graphs, semi_keys, semi_key2idxs,
            n_nodes, reactant_pairs, product_pairs
        )

        result = self.decode(
            reaction_embs, labels, attn_mask,
            key_padding_mask=key_padding_mask, seq_types=seq_types
        )
        return result


def average_mole_for_rxn(
    mole_embs, n_nodes, rxn_ids, reactant_pairs, product_pairs
):
    rxn_reac_embs = torch.zeros_like(mole_embs)
    rxn_prod_embs = torch.zeros_like(mole_embs)
    rxn_reac_cnt = torch.zeros(n_nodes).to(mole_embs)
    rxn_prod_cnt = torch.zeros(n_nodes).to(mole_embs)

    rxn_reac_embs.index_add_(
        index=reactant_pairs[:, 0], dim=0,
        source=mole_embs[reactant_pairs[:, 1]]
    )
    rxn_prod_embs.index_add_(
        index=product_pairs[:, 0], dim=0,
        source=mole_embs[product_pairs[:, 1]]
    )

    rxn_prod_cnt.index_add_(
        index=product_pairs[:, 0], dim=0,
        source=torch.ones(product_pairs.shape[0]).to(mole_embs)
    )

    rxn_reac_cnt.index_add_(
        index=reactant_pairs[:, 0], dim=0,
        source=torch.ones(reactant_pairs.shape[0]).to(mole_embs)
    )

    assert torch.all(rxn_reac_cnt[rxn_ids] > 0).item(), \
        "Some rxn Missing reactant embeddings"
    assert torch.all(rxn_prod_cnt[rxn_ids] > 0).item(), \
        "Some rxn missing product embeddings"

    rxn_reac_embs = rxn_reac_embs[rxn_ids] / \
        rxn_reac_cnt[rxn_ids].unsqueeze(-1)
    rxn_prod_embs = rxn_prod_embs[rxn_ids] / \
        rxn_prod_cnt[rxn_ids].unsqueeze(-1)
    rxn_embs = torch.cat([rxn_reac_embs, rxn_prod_embs], dim=-1)
    return rxn_embs


class AblationModel(nn.Module):
    def __init__(
        self, gnn1, PE, net_dim, heads, dropout, dec_layers,
        n_words, mol_dim, with_type=False, ntypes=None,
    ):
        super(AblationModel, self).__init__()
        self.pos_enc = PE
        self.gnn1 = gnn1
        self.rxn_linear = torch.nn.Linear(mol_dim * 2, net_dim)
        self.net_dim = net_dim
        self.mole_dim = mol_dim
        self.edge_emb = torch.nn.ParameterDict({
            'reactant': torch.nn.Parameter(torch.randn(net_dim)),
            'product': torch.nn.Parameter(torch.randn(net_dim)),
        })
        self.semi_init = torch.nn.Parameter(torch.randn(1, 1, net_dim))
        self.pooler = torch.nn.MultiheadAttention(
            embed_dim=net_dim, kdim=mol_dim, vdim=mol_dim,
            num_heads=heads, batch_first=True, dropout=dropout
        )
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
        self.decoder = torch.nn.TransformerEncoder(t_layer, dec_layers)
        self.word_emb = torch.nn.Embedding(n_words, net_dim)
        self.with_type = with_type
        if self.with_type:
            assert ntypes is not None, "require type numbers"
            self.type_embs = torch.nn.Embedding(ntypes, net_dim)

    def encode(
        self, reac_graphs, prod_graphs, n_reac, n_prod, n_reaction, reactant_pairs=None, product_pairs=None
    ):

        reac_keys = self.node_type['reactant'].unsqueeze(
            0).repeat(n_reac, 1).unsqueeze(1)
        prod_keys = self.node_type['product'].unsqueeze(
            0).repeat(n_prod, 1).unsqueeze(1)

        reac_feats, _ = self.gnn1(reac_graphs)
        reac_feats = graph2batch(reac_feats, reac_graphs.batch_mask)

        prod_feats, _ = self.gnn1(prod_graphs)
        prod_feats = graph2batch(prod_feats, prod_graphs.batch_mask)

        reac_embs, _ = self.pooler(
            query=reac_keys, key=reac_feats, value=reac_feats,
            key_padding_mask=torch.logical_not(reac_graphs.batch_mask)
        )
        reac_embs = reac_embs.squeeze(1)

        prod_embs, _ = self.pooler(
            query=prod_keys, key=prod_feats, value=prod_feats,
            key_padding_mask=torch.logical_not(prod_graphs.batch_mask)
        )
        prod_embs = prod_embs.squeeze(1)

        assert reactant_pairs is not None and product_pairs is not None, \
            "Require reaction comp mapper for rxns embedding generation"
        rxn_embs = average_mole(
            reac_embs, prod_embs, reactant_pairs, product_pairs, n_reaction
        )
        return self.rxn_linear(rxn_embs)

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
        self, reac_graphs, prod_graphs, n_reac, n_prod, n_nodes,
        labels, attn_mask, reactant_pairs=None, product_pairs=None,
        key_padding_mask=None, seq_types=None,
    ):
        reaction_embs = self.encode(
            reac_graphs,  prod_graphs, n_reac, n_prod, n_nodes, reactant_pairs, product_pairs
        )

        result = self.decode(
            reaction_embs, labels, attn_mask,
            key_padding_mask=key_padding_mask, seq_types=seq_types
        )
        return result


def average_mole(
    reac_embs, prod_embs, reactant_pairs, product_pairs, n_nodes
):
    rxn_reac_embs = torch.zeros(n_nodes, reac_embs.shape[1]).to(reac_embs)
    rxn_prod_embs = torch.zeros(n_nodes, reac_embs.shape[1]).to(reac_embs)
    rxn_reac_cnt = torch.zeros(n_nodes).to(reac_embs)
    rxn_prod_cnt = torch.zeros(n_nodes).to(reac_embs)

    rxn_reac_embs.index_add_(
        index=reactant_pairs[:, 0], dim=0,
        source=reac_embs[reactant_pairs[:, 1]]
    )
    rxn_prod_embs.index_add_(
        index=product_pairs[:, 0], dim=0,
        source=prod_embs[product_pairs[:, 1]]
    )

    rxn_prod_cnt.index_add_(
        index=product_pairs[:, 0], dim=0,
        source=torch.ones(product_pairs.shape[0]).to(reac_embs)
    )

    rxn_reac_cnt.index_add_(
        index=reactant_pairs[:, 0], dim=0,
        source=torch.ones(reactant_pairs.shape[0]).to(reac_embs)
    )

    assert torch.all(rxn_reac_cnt > 0).item(), \
        "Some rxn Missing reactant embeddings"
    assert torch.all(rxn_prod_cnt > 0).item(), \
        "Some rxn missing product embeddings"

    rxn_reac_embs = rxn_reac_embs / \
        rxn_reac_cnt.unsqueeze(-1)
    rxn_prod_embs = rxn_prod_embs / \
        rxn_prod_cnt.unsqueeze(-1)
    rxn_embs = torch.cat([rxn_reac_embs, rxn_prod_embs], dim=-1)
    return rxn_embs
