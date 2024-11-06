import torch
from torch.utils.data import DataLoader

from utils.data_utils import fix_seed, load_uspto_1kk
from utils.dataset import ConditionDataset, ablation_graph


from model import GATBase, AblationModel, PositionalEncoding
import argparse
import pickle
from tqdm import tqdm


def get_x(model, loader, mapper):
    key2idx, tdx, all_f, lbs, model = {}, 0, [], [], model.eval()

    for data, raw, bs in tqdm(loader):
        reac_graphs, prod_graphs, reactant_pairs, product_pairs = data

        reac_graphs = reac_graphs.to(device)
        prod_graphs = prod_graphs.to(device)
        reactant_pairs = reactant_pairs.to(device)
        product_pairs = product_pairs.to(device)

        with torch.no_grad():
            features = model.encode(
                reac_graphs=reac_graphs, prod_graphs=prod_graphs,
                batch_size=bs, rpairs=reactant_pairs, ppairs=product_pairs
            )

        for x in raw:
            key2idx[x['canonical_rxn']] = tdx
            tdx += 1
            lbs.append(mapper[x['canonical_rxn']])
        all_f.append(features.cpu())

    all_f = torch.cat(all_f, dim=0)
    return {'smiles2idx': key2idx, 'labels': lbs, 'features': all_f}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for main experiment')
    # model definition

    parser.add_argument(
        '--mole_layer', default=5, type=int,
        help='the number of layer for mole gnn'
    )
    parser.add_argument(
        '--dim', type=int, default=512,
        help='the num of dim for the model'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model'
    )
    parser.add_argument(
        '--heads', type=int, default=8,
        help='the number of heads for multihead attention'
    )
    parser.add_argument(
        '--decoder_layer', type=int, default=6,
        help='the num of layers for decoder'
    )

    # inference args

    parser.add_argument(
        '--bs', type=int, default=256,
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
        '--device', type=int, default=3,
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

    data2label = {
        x['canonical_rxn']: x['label'] for x
        in train_val_data + test_data
    }

    shown_train, shown_test = set(), set()
    new_train, new_test = [], []

    for x in train_val_data:
        if x['canonical_rxn'] not in shown_train:
            shown_train.add(x['canonical_rxn'])
            new_train.append(x)

    for x in test_data:
        if x['canonical_rxn'] not in shown_test:
            shown_test.add(x['canonical_rxn'])
            new_test.append(x)

    train_val_loader = DataLoader(
        new_train, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=lambda x: (ablation_graph(x), x, len(x))
    )

    test_loader = DataLoader(
        new_test, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=lambda x: (ablation_graph(x), x, len(x))
    )

    mol_gnn = GATBase(
        num_layers=args.mole_layer, num_heads=args.heads, dropout=0,
        embedding_dim=args.dim, negative_slope=args.negative_slope
    )

    pos_env = PositionalEncoding(args.dim, 0, maxlen=1024)

    model = AblationModel(
        gnn1=mol_gnn, PE=pos_env, net_dim=args.dim,
        heads=args.heads, dropout=0, dec_layers=args.decoder_layer,
        n_words=len(label_mapper), mol_dim=args.dim,
        with_type=False
    ).to(device)

    weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weight)
    model = model.eval()

    train_val_data = get_x(model, train_val_loader, data2label)
    test_data = get_x(model, test_loader, data2label)

    torch.save({'train': train_val_data, 'test': test_data}, args.output_dir)
