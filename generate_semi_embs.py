from pretrain.pretrain_gnn import GNN_graph
from pretrain.pretrain_gnn_utils import smiles2graph_with_am
import json
from tqdm import tqdm
from torch_geometric.data import Data as GData
import torch
import argparse
import numpy as np
from utils.chemistry_utils import (
    canonical_smiles, remove_am, get_semi_reaction
)


def col_fn(batch):
    rc_pd_list, all_graphs = [], []
    for a, b, c in batch:
        r_smi, r_g = get_semi_reaction(
            mapped_reac_list=a, mapped_prod_list=b,
            trans_fn=smiles2graph_with_am, add_pesudo_node=False
        )
        rc_pd_list.extend((c, y) for y in r_smi)
        all_graphs.extend(r_g)
    return rc_pd_list, graph_col_fn(all_graphs)


def graph_col_fn(data_batch):
    batch_size, max_node = len(data_batch), 0
    edge_idxes, edge_feats, node_feats, lstnode = [], [], [], 0
    batch, ptr, node_per_graph = [], [0], []
    for idx, graph in enumerate(data_batch):
        num_nodes = graph['num_nodes']
        num_edges = graph['edge_index'].shape[1]

        if num_nodes > 0:
            edge_idxes.append(graph['edge_index'] + lstnode)
            edge_feats.append(graph['edge_feat'])
            node_feats.append(graph['node_feat'])

        lstnode += num_nodes
        max_node = max(max_node, num_nodes)
        node_per_graph.append(num_nodes)
        batch.append(np.ones(num_nodes, dtype=np.int64) * idx)
        ptr.append(lstnode)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0),
        'ptr': np.array(ptr, dtype=np.int64)
    }

    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode

    all_batch_mask = torch.zeros((batch_size, max_node))
    for idx, mk in enumerate(node_per_graph):
        all_batch_mask[idx, :mk] = 1
    result['batch_mask'] = all_batch_mask.bool()

    return GData(**result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path containing the data'
    )
    parser.add_argument(
        '--output_path', required=True, type=str,
        help='the path storing the generate features'
    )

    parser.add_argument(
        '--ckpt_path', default='pretrain_gnn/supervised_contextpred.pth',
        help='the path containing the pretrained weight', type=str
    )

    parser.add_argument(
        '--batch_size', default=512, type=int,
        help='the batch size for inference'
    )
    parser.add_argument(
        '--dataset', choices=['uspto_condition', 'uspto_500mt'],
        help='the dataset to process', required=True
    )
    parser.add_argument(
        '--device', type=int, default=-1,
        help='the device id for inference, minus for cpu'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='the num workers for data loader'
    )

    args = parser.parse_args()

    with open(args.data_path) as Fin:
        raw_info = json.load(Fin)

    if args.dataset == 'uspto_condition':
        all_mapped_rxn = []
        for entry in tqdm(raw_info):
            info_tup = (
                entry['new']['mapped_reac_list'],
                entry['new']['mapped_prod_list'],
                entry['new']['canonical_rxn']
            )
            all_mapped_rxn.append(info_tup)

    else:
        raise NotImplementedError('Not Implemented Yet')

    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    model = GNN_graph(num_layer=5, emb_dim=300).to(device)
    model.from_pretrained(args.ckpt_path)
    model = model.eval()

    mol_loader = torch.utils.data.DataLoader(
        all_mapped_rxn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=col_fn
    )

    mole_smiles, mole_features = [], []

    for smiles_list, graph in tqdm(mol_loader):
        with torch.no_grad():
            features = model(graph.to(device))
        mole_smiles.extend(smiles_list)
        mole_features.append(features.cpu())

    mole_features = torch.cat(mole_features, dim=0)

    mol2idx = {x: idx for idx, x in enumerate(mole_smiles)}

    torch.save(
        {'smiles2idx': mol2idx, 'features': mole_features},
        args.output_path
    )
