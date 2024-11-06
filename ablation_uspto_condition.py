import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from utils.data_utils import fix_seed, parse_uspto_condition_data
from utils.data_utils import check_early_stop
from utils.dataset import ConditionDataset, uspto_condition_ablation


from model import GATBase, AblationModel, PositionalEncoding
from training import (
    train_uspto_condition_ablation, eval_uspto_condition_ablation
)
import argparse
import os
import time
import pickle
import json


def make_dir(args):
    timestamp = time.time()
    detail_dir = os.path.join(args.base_log, f'{timestamp}')
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    log_dir = os.path.join(detail_dir, 'log.json')
    model_dir = os.path.join(detail_dir, 'model.pth')
    token_dir = os.path.join(detail_dir, 'token.pkl')
    return log_dir, model_dir, token_dir


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
        '--dropout', type=float, default=0.1,
        help='the dropout for model'
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

    # training args

    parser.add_argument(
        '--lr', type=float, default=0.000125,
        help='the learning rate for training'
    )
    parser.add_argument(
        '--bs', type=int, default=256,
        help='the batch size for training'
    )
    parser.add_argument(
        '--epoch', type=int, default=200,
        help='the number of epochs for training'
    )

    parser.add_argument(
        '--early_stop', type=int, default=20,
        help='the number of epochs for checking early stop, 0 for invalid'
    )

    parser.add_argument(
        '--step_start', type=int, default=20,
        help='the step to start lr decay'
    )
    parser.add_argument(
        '--base_log', type=str, default='log_condab',
        help='the path for contraining log'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='the number of worker for dataloader'
    )

    parser.add_argument(
        '--warmup', type=int, default=4,
        help='the number of epochs for warmup'
    )
    parser.add_argument(
        '--lrgamma', type=float, default=0.99,
        help='the lr decay rate for training'
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
        '--data_path', required=True, type=str,
        help='the path containing the data'
    )

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    log_dir, model_dir, token_dir = make_dir(args)

    all_data, label_mapper = parse_uspto_condition_data(args.data_path)

    train_set = ConditionDataset(
        reactions=all_data['train_data'],
        labels=[x['label'] for x in all_data['train_data']]
    )

    val_set = ConditionDataset(
        reactions=all_data['val_data'],
        labels=[x['label'] for x in all_data['val_data']]
    )

    test_set = ConditionDataset(
        reactions=all_data['test_data'],
        labels=[x['label'] for x in all_data['test_data']]
    )

    train_loader = DataLoader(
        train_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=True, collate_fn=uspto_condition_ablation)

    val_loader = DataLoader(
        val_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=uspto_condition_ablation
    )

    test_loader = DataLoader(
        test_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=uspto_condition_ablation
    )

    mol_gnn = GATBase(
        num_layers=args.mole_layer, num_heads=args.heads, dropout=args.dropout,
        embedding_dim=args.dim, negative_slope=args.negative_slope
    )

    pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=1024)

    model = AblationModel(
        gnn1=mol_gnn, PE=pos_env, net_dim=args.dim,
        heads=args.heads, dropout=args.dropout, dec_layers=args.decoder_layer,
        n_words=len(label_mapper), mol_dim=args.dim,
        with_type=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sher = ExponentialLR(optimizer, gamma=args.lrgamma)

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    with open(token_dir, 'wb') as Fout:
        pickle.dump(label_mapper, Fout)

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout)

    best_pref, best_ep = None, None

    for ep in range(args.epoch):
        print(f'[INFO] training epoch {ep}')
        loss = train_uspto_condition_ablation(
            loader=train_loader, model=model, optimizer=optimizer,
            device=device, warmup=(ep < args.warmup)
        )
        val_results = eval_uspto_condition_ablation(
            val_loader, model, device
        )
        test_results = eval_uspto_condition_ablation(
            test_loader, model, device
        )

        print('[Train]:', loss)
        print('[Valid]:', val_results)
        print('[Test]:', test_results)

        log_info['train_loss'].append(loss)
        log_info['valid_metric'].append(val_results)
        log_info['test_metric'].append(test_results)

        if ep >= args.warmup and ep >= args.step_start:
            lr_sher.step()
            print('[Lr]', lr_sher.get_last_lr())

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if best_pref is None or val_results['overall'] > best_pref:
            best_pref, best_ep = val_results['overall'], ep
            torch.save(model.state_dict(), model_dir)

        if args.early_stop >= 5 and ep > max(10, args.early_stop):
            tx = log_info['valid_metric'][-args.early_stop:]
            tx = [x['overall'] for x in tx]
            # keys = [
            #     'overall', 'catalyst', 'solvent1', 'solvent2',
            #     'reagent1', 'reagent2'
            # ]
            # tx = [[x[key] for x in tx] for key in keys]
            if check_early_stop(tx):
                break

    print(f'[INFO] best acc epoch: {best_ep}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep]}')
