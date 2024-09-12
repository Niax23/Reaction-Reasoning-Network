import torch
from utils.graph_utils import smiles2graph
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
    node_ptr, node_batch, lstnode = [0], [], 0
    max_node = max(x['num_nodes'] for x in batch)
    batch_mask = torch.zeros(batch_size, max_node).bool()

    for idx, gp in enumerate(batch):
        node_cnt = gp['num_nodes']
        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)
        batch_mask[idx, :node_cnt] = True

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
    return torch_geometric.data.Data(**result)


def reaction_graph_colfn(reactions, G, hop, max_neighbors=None):
    mol_strs, edge_index, edge_types, mol_mask, reaction_mask, \
        req_mask = G.sample_multiple_subgraph(reactions, hop)

    mol_graphs = graph_col_fn([smiles2graph(x) for x in mol_strs])
    edge_index = torch.LongTensor(edge_index).T
    mol_mask = torch.BoolTensor(mol_mask)
    reaction_mask = torch.BoolTensor(reaction_mask)
    req_mask = torch.BoolTensor(req_mask)

    return mol_graphs, edge_index, edge_types, mol_mask, reaction_mask, req_mask


def uspto_condition_colfn(batch, G, hop):
    x_infos = reaction_graph_colfn([x[0] for x in batch], G, hop)
    labels = torch.LongTensor([x[1] for x in batch])
    label_types = torch.LongTensor([[0, 1, 1, 2]] * labels.shape[0])
    
    return x_infos + (labels, label_types)
