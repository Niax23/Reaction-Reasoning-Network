import torch
from utils.graph_utils import smiles2graph
from utils.chemistry_utils import get_semi_reaction
import numpy as np
import torch_geometric
from numpy import concatenate as npcat


class ConditionDataset(torch.utils.data.Dataset):
    def __init__(self, reactions, labels):
        super(ConditionDataset, self).__init__()
        self.reactions = reactions
        self.labels = labels

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        return self.reactions[idx], self.labels[idx]


def graph_col_fn(batch):
    batch_size, edge_idx, node_feat, edge_feat = len(batch), [], [], []
    node_ptr, node_batch, lstnode, pes_mask = [0], [], 0, []
    max_node = max(x['num_nodes'] for x in batch)
    batch_mask = torch.zeros(batch_size, max_node).bool()

    for idx, gp in enumerate(batch):
        node_cnt = gp['num_nodes']
        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)
        batch_mask[idx, :node_cnt] = True

        if 'pesudo_mask' in gp:
            pes_mask.append(gp['pesudo_mask'])

        lstnode += node_cnt
        node_batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
        node_ptr.append(lstnode)

    result = {
        'x': torch.from_numpy(npcat(node_feat, axis=0)),
        "edge_attr": torch.from_numpy(npcat(edge_feat, axis=0)),
        'ptr': torch.LongTensor(node_ptr),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
        'num_nodes': lstnode,
        'batch_mask': batch_mask
    }

    if len(pes_mask) > 0:
        result['pesudo_mask'] = torch.from_numpy(npcat(pes_mask, axis=0))

    return torch_geometric.data.Data(**result)


def reaction_graph_colfn(reactions, G, hop, max_neighbors=None):
    mol_strs, edge_index, edge_types, mol_mask, reaction_mask, \
        req_ids = G.sample_multiple_subgraph(reactions, hop, max_neighbors)

    mol_graphs = graph_col_fn([smiles2graph(x) for x in mol_strs])
    edge_index = torch.LongTensor(edge_index).T
    mol_mask = torch.BoolTensor(mol_mask)
    reaction_mask = torch.BoolTensor(reaction_mask)

    return mol_graphs, edge_index, edge_types, mol_mask, reaction_mask, req_ids


def uspto_condition_colfn(batch, G, hop, max_neighbors=None):
    x_infos = reaction_graph_colfn(
        reactions=[x[0] for x in batch], G=G, hop=hop,
        max_neighbors=max_neighbors
    )
    labels = torch.LongTensor([x[1] for x in batch])
    label_types = torch.LongTensor([[0, 1, 1, 2]] * labels.shape[0])

    return x_infos + (labels, label_types)


def fmap(mapper, A):
    return mapper['features'][mapper['smiles2idx'][A]]


def reaction_graph_colfn_rxn(reactions, G, hop, mapper, max_neighbors=None):
    mole_strs, molecule_ids, rxn_sms, rxn_ids, edge_index, edge_types, \
        required_ids, reactant_pairs, product_pairs, n_node =\
        G.sample_multiple_subgraph_rxn(reactions, hop, max_neighbors)

    edge_index = torch.LongTensor(edge_index).T
    product_pairs = torch.LongTensor(product_pairs)
    reactant_pairs = torch.LongTensor(reactant_pairs)
    mole_embs = np.stack([fmap(mapper, x) for x in mole_strs], axis=0)
    mole_embs = torch.from_numpy(mole_embs)

    return mole_embs, molecule_ids, rxn_sms, rxn_ids, edge_index, \
        edge_types, required_ids, reactant_pairs, product_pairs, n_node


def uspto_condition_colfn_rxn(batch, G, hop, mapper, max_neighbors=None):
    x_infos = reaction_graph_colfn_rxn(
        reactions=[x[0] for x in batch], G=G, hop=hop,
        max_neighbors=max_neighbors, mapper=mapper
    )
    labels = torch.LongTensor([x[1] for x in batch])
    label_types = torch.LongTensor([[0, 1, 1, 2]] * labels.shape[0])

    return x_infos + (labels, label_types)


def reaction_graph_colfn_semi(
    reactions, G, hop, fmapper, emapper, max_neighbors=None
):
    molecules, molecule_ids, rxn_sms, rxn_mapped_infos, rxn_ids,  \
        edge_index, edge_types, edge_semi, required_ids, \
        reactant_pairs, product_pairs, n_node = \
        G.sample_multiple_subgraph_rxn(reactions, hop, max_neighbors)

    edge_index = torch.LongTensor(edge_index).T
    product_pairs = torch.LongTensor(product_pairs)
    reactant_pairs = torch.LongTensor(reactant_pairs)
    mole_embs = np.stack([fmap(fmapper, x) for x in molecules], axis=0)
    mole_embs = torch.from_numpy(mole_embs)

    edge_attrs = np.stack([fmap(emapper, x) for x in edge_semi], axis=0)
    edge_attrs = torch.from_numpy(edge_attrs)
    edge_semi_mask = [x == 'reactant' for x in edge_types]
    full_edge_attr = torch.zeros((len(edge_types), edge_attrs.shape[-1]))
    full_edge_attr[edge_semi_mask] = edge_attrs

    return mole_embs, molecule_ids, rxn_sms, rxn_ids, edge_index, edge_types,\
        full_edge_attr, required_ids, reactant_pairs, product_pairs, n_node


def uspto_condition_colfn_semi(
    batch, G, hop, fmapper, emapper, max_neighbors=None
):
    x_infos = reaction_graph_colfn_semi(
        reactions=[x[0] for x in batch], G=G, hop=hop,
        max_neighbors=max_neighbors, fmapper=fmapper, emapper=emapper
    )
    labels = torch.LongTensor([x[1] for x in batch])
    label_types = torch.LongTensor([[0, 1, 1, 2]] * labels.shape[0])

    return x_infos + (labels, label_types)


def reaction_graph_map_final(
    reactions, G, hop, fmapper, emapper, max_neighbors=None
):
    molecules, mts, molecule_ids, rxn_sms, rxn_mapped_infos, rxn_ids,  \
        edge_index, edge_types, edge_semi, required_ids, \
        reactant_pairs, product_pairs, n_node = \
        G.sample_multiple_subgraph_rxn(reactions, hop, max_neighbors)

    edge_index = torch.LongTensor(edge_index).T
    product_pairs = torch.LongTensor(product_pairs)
    reactant_pairs = torch.LongTensor(reactant_pairs)
    mole_embs = np.stack([fmap(fmapper, x) for x in molecules], axis=0)
    mole_embs = torch.from_numpy(mole_embs)

    edge_attrs = np.stack([fmap(emapper, x) for x in edge_semi], axis=0)
    edge_attrs = torch.from_numpy(edge_attrs)
    edge_semi_mask = [x == 'reactant' for x in edge_types]
    full_edge_attr = torch.zeros((len(edge_types), edge_attrs.shape[-1]))
    full_edge_attr[edge_semi_mask] = edge_attrs

    return mole_embs, mts, molecule_ids, rxn_sms, rxn_ids, edge_index, \
        edge_types, full_edge_attr, required_ids, reactant_pairs, \
        product_pairs, n_node


def uspto_condition_map_final(
    batch, G, hop, fmapper, emapper, max_neighbors=None
):
    x_infos = reaction_graph_map_final(
        reactions=[x[0] for x in batch], G=G, hop=hop,
        max_neighbors=max_neighbors, fmapper=fmapper, emapper=emapper
    )
    labels = torch.LongTensor([x[1] for x in batch])
    label_types = torch.LongTensor([[0, 1, 1, 2]] * labels.shape[0])

    return x_infos + (labels, label_types)


def reaction_graph_final(reactions, G, hop, max_neighbors=None):
    molecules, mts, molecule_ids, rxn_sms, rxn_mapped_infos, rxn_ids,  \
        edge_index, edge_types, edge_semi, required_ids, \
        reactant_pairs, product_pairs, n_node = \
        G.sample_multiple_subgraph_rxn(reactions, hop, max_neighbors)

    edge_index = torch.LongTensor(edge_index).T
    product_pairs = torch.LongTensor(product_pairs)
    reactant_pairs = torch.LongTensor(reactant_pairs)
    mole_graphs = [smiles2graph(x, with_amap=False) for x in molecules]
    mole_graphs = graph_col_fn(mole_graphs)

    semi_graphs, semi_keys = [], []
    for idx, (a, b) in enumerate(rxn_mapped_infos):
        rsm, rgp = get_semi_reaction(
            mapped_reac_list=a, mapped_prod_list=b, add_pesudo_node=True,
            trans_fn=lambda x: smiles2graph(x, with_amap=True)
        )
        semi_graphs.extend(rgp)
        semi_keys.extend((rxn_sms[idx], t) for t in rsm)

    smkey2idx = {k: v for v, k in enumerate(semi_keys)}
    semi_graphs = graph_col_fn(semi_graphs)

    return mole_graphs, mts, molecule_ids, rxn_ids, edge_index, \
        edge_types, semi_graphs, edge_semi, smkey2idx, required_ids,\
        reactant_pairs, product_pairs, n_node


def uspto_condition_final(batch, G, hop, max_neighbors=None):
    x_infos = reaction_graph_final(
        reactions=[x[0] for x in batch], G=G, hop=hop,
        max_neighbors=max_neighbors,
    )
    labels = torch.LongTensor([x[1] for x in batch])
    label_types = torch.LongTensor([[0, 1, 1, 2]] * labels.shape[0])

    return x_infos + (labels, label_types)


def uspto_500mt_final(batch, G, hop, begin_tok, end_tok, max_neighbors=None):
    x_infos = reaction_graph_final(
        reactions=[x[0] for x in batch], G=G, hop=hop,
        max_neighbors=max_neighbors,
    )

    labels = [[begin_tok] + x + [end_tok] for x in labels]

    return x_infos + (labels, )
