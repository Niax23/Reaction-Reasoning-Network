import torch
from torch.utils.data import DataLoader

from utils.data_utils import fix_seed, load_uspto_1kk
from utils.sep_network import SepNetwork
from utils.dataset import ConditionDataset, reaction_graph_final


from model import GATBase, RxnNetworkGNN, PositionalEncoding, FullModel
import argparse
import pickle
from tqdm import tqdm


class col_x:
    def __init__(self, G, hop, max_neighbor=None):
        self.G = G
        self.hop = hop
        self.max_neighbor = max_neighbor

    def fwd(self, batch):
        return reaction_graph_final(
            batch, self.G, self.hop, self.max_neighbor
        ), batch


def get_x(model, loader, mapper):
    key2idx, tdx, all_f, lbs, model = {}, 0, [], [], model.eval()

    for data, raw in tqdm(loader):
        mole_graphs, mts, molecule_ids, rxn_ids, edge_index, \
            edge_types, semi_graphs, semi_keys, smkey2idx, required_ids, \
            reactant_pairs, product_pairs, n_node = data

        mole_graphs = mole_graphs.to(device)
        edge_index = edge_index.to(device)
        reactant_pairs = reactant_pairs.to(device)
        product_pairs = product_pairs.to(device)
        semi_graphs = semi_graphs.to(device)

        with torch.no_grad():
            features = model.encode(
                mole_graphs=mole_graphs, mts=mts, molecule_ids=molecule_ids,
                rxn_ids=rxn_ids, required_ids=required_ids,
                edge_index=edge_index, edge_types=edge_types,
                semi_graphs=semi_graphs, semi_keys=semi_keys,
                semi_key2idxs=smkey2idx, n_nodes=n_node,
                reactant_pairs=reactant_pairs, product_pairs=product_pairs
            )

        for x in raw:
            key2idx[x] = tdx
            tdx += 1
            lbs.append(mapper[x])
        all_f.append(features.cpu())

    all_f = torch.cat(all_f, dim=0)
    return {'smiles2idx': key2idx, 'labels': lbs, 'features': all_f}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for feature inference')
    # model definition

    parser.add_argument(
        '--mole_layer', default=5, type=int,
        help='the number of layer for mole gnn'
    )
    parser.add_argument(
        '--dim', type=int, default=300,
        help='the num of dim for the model'
    )
    parser.add_argument(
        '--reaction_hop', type=int, default=1,
        help='the number of hop for sampling graphs'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model'
    )
    parser.add_argument(
        '--heads', type=int, default=4,
        help='the number of heads for multihead attention'
    )
    parser.add_argument(
        '--decoder_layer', type=int, default=6,
        help='the num of layers for decoder'
    )
    parser.add_argument(
        '--init_rxn', action='store_true',
        help='use pretrained features to build rxn feat or not'
    )

    # inference args

    parser.add_argument(
        '--bs', type=int, default=32,
        help='the batch size for training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='the number of worker for dataloader'
    )

    parser.add_argument(
        '--seed', type=int, default=2023,
        help='the random seed for training'
    )

    parser.add_argument(
        '--device', type=int, default=-1,
        help='CUDA device to use; -1 for CPU'
    )

    # data config
    parser.add_argument(
        '--train_val_path', required=True, type=str,
        help='the path containing train and val data'
    )
    parser.add_argument(
        '--test_path', required=True, type=str,
        help='the path containing test data'
    )

    parser.add_argument(
        '--max_neighbors', type=int, default=20,
        help='max neighbors when sampling'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='the path of output dir'
    )

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='the path of pretrained checkpoint'
    )
    parser.add_argument(
        '--token_ckpt', type=str, required=True,
        help='the path of pretrained tokenizer'
    )

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    with open(args.token_ckpt, 'rb') as Fin:
        label_mapper = pickle.load(Fin)

    train_val_data = load_uspto_1kk(args.train_val_path)
    test_data = load_uspto_1kk(args.test_path)
    all_net = SepNetwork(train_val_data + test_data)

    data2label = {
        x['canonical_rxn']: x['label'] for x
        in train_val_data + test_data
    }

    train_val_set = [x['canonical_rxn'] for x in train_val_data]
    test_set = [x['canonical_rxn'] for x in test_data]

    train_val_set = list(set(train_val_set))
    test_set = list(set(test_set))

    xcf = col_x(all_net, args.reaction_hop, args.max_neighbors)

    train_val_loader = DataLoader(
        train_val_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=xcf.fwd
    )

    test_loader = DataLoader(
        test_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=xcf.fwd
    )

    mol_gnn = GATBase(
        num_layers=args.mole_layer, num_heads=args.heads, dropout=0,
        embedding_dim=args.dim, negative_slope=args.negative_slope
    )

    net_gnn = RxnNetworkGNN(
        num_layers=args.reaction_hop * 2 + 1, num_heads=args.heads,
        dropout=0, embedding_dim=args.dim, negative_slope=args.negative_slope
    )

    pos_env = PositionalEncoding(args.dim, 0, maxlen=128)

    model = FullModel(
        gnn1=mol_gnn, gnn2=net_gnn, PE=pos_env, net_dim=args.dim,
        heads=args.heads, dropout=0, dec_layers=args.decoder_layer,
        n_words=len(label_mapper), mol_dim=args.dim,
        with_type=True, ntypes=3, init_rxn=args.init_rxn
    ).to(device)

    weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weight)
    model = model.eval()

    train_val_data = get_x(model, train_val_loader, data2label)
    test_data = get_x(model, test_loader, data2label)

    torch.save({'train': train_val_data, 'test': test_data}, args.output_dir)
