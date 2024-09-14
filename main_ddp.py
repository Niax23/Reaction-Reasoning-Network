import pickle
import torch
import argparse
import json
import os
import time


from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import pandas
from tqdm import tqdm

from utils.data_utils import fix_seed, parse_uspto_condition_data
from utils.data_utils import check_early_stop
from utils.network import ChemicalReactionNetwork
from utils.dataset import ConditionDataset, uspto_condition_colfn


from model import GATBase, MyModel, RxnNetworkGNN, PositionalEncoding
from training import train_uspto_condition, eval_uspto_condition


import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch.utils.data.distributed import DistributedSampler


def make_dir(args):
    timestamp = time.time()
    detail_dir = os.path.join(args.base_log, f'{timestamp}')
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    log_dir = os.path.join(detail_dir, 'log.json')
    model_dir = os.path.join(detail_dir, 'model.pth')
    token_dir = os.path.join(detail_dir, 'token.pkl')
    return log_dir, model_dir, token_dir


def main_worker(worker_idx, args,  log_dir, model_dir):

    print(f'[INFO] Process {worker_idx} start')
    torch_dist.init_process_group(
        backend='nccl', init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.num_gpus, rank=worker_idx
    )

    device = torch.device(f'cuda:{worker_idx}')
    verbose = (worker_idx == 0)

    all_data, label_mapper = parse_uspto_condition_data(args.data_path)
    all_net = ChemicalReactionNetwork(
        all_data['train_data'] + all_data['val_data'] + all_data['test_data']
    )

    train_net = all_net if args.transductive else\
        ChemicalReactionNetwork(all_data['train_data'])

    train_set = ConditionDataset(
        reactions=[x['canonical_rxn'] for x in all_data['train_data']],
        labels=[x['label'] for x in all_data['train_data']]
    )

    val_set = ConditionDataset(
        reactions=[x['canonical_rxn'] for x in all_data['val_data']],
        labels=[x['label'] for x in all_data['val_data']]
    )

    test_set = ConditionDataset(
        reactions=[x['canonical_rxn'] for x in all_data['test_data']],
        labels=[x['label'] for x in all_data['test_data']]
    )

    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    train_loader = DataLoader(
        train_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=True, collate_fn=lambda x: uspto_condition_colfn(
            x, train_net, args.reaction_hop, args.max_neighbors
        ), pin_memory=True, sampler=train_sampler
    )

    val_loader = DataLoader(
        val_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=lambda x: uspto_condition_colfn(
            x, all_net, args.reaction_hop, args.max_neighbors
        ), pin_memory=True, sampler=val_sampler
    )

    test_loader = DataLoader(
        test_set, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, collate_fn=lambda x: uspto_condition_colfn(
            x, all_net, args.reaction_hop, args.max_neighbors
        ), pin_memory=True, sampler=test_sampler
    )

    mol_gnn = GATBase(
        num_layers=args.mole_layer, num_heads=args.heads, dropout=args.dropout,
        embedding_dim=args.dim, negative_slope=args.negative_slope
    )

    net_gnn = RxnNetworkGNN(
        num_layers=args.reaction_hop * 2 + 1, num_heads=args.heads,
        dropout=args.dropout, embedding_dim=args.dim,
        negative_slope=args.negative_slope
    )

    pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=128)

    model = MyModel(
        gnn1=mol_gnn, gnn2=net_gnn, PE=pos_env, molecule_dim=args.dim,
        net_dim=args.dim, heads=args.heads, dropout=args.dropout,
        dec_layers=args.decoder_layer, n_words=len(label_mapper),
        with_type=True, ntypes=3
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[worker_idx], output_device=worker_idx,
        find_unused_parameters=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scher = ExponentialLR(optimizer, args.lrgamma, verbose=verbose)

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    if verbose:
        with open(token_dir, 'wb') as Fout:
            pickle.dump(label_mapper, Fout)
        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout)

    best_pref, best_ep = None, None

    for ep in range(args.epoch):
        if verbose:
            print(f'[INFO] traing at epoch {ep + 1}')

        train_sampler.set_epoch(ep)
        loss = train_uspto_condition(
            loader=train_loader, model=model, optimizer=optimizer,
            device=device, warmup=(ep < args.warmup)
        )
        val_results = eval_uspto_condition(val_loader, model, device)
        test_results = eval_uspto_condition(test_loader, model, device)

        torch_dist.barrier()
        loss.all_reduct(device)
        val_results.all_reduct(device)
        test_results.all_reduct(device)

        log_info['train_loss'].append(loss.get_all_value_dict())
        log_info['valid_metric'].append(val_results.get_all_value_dict())
        log_info['test_metric'].append(test_results.get_all_value_dict())

        if verbose:
            print('[TRAIN]', log_info['train_loss'][-1])
            print('[VALID]', log_info['val_metric'][-1])
            print('[TEST]', log_info['test_metric'][-1])
            

            with open(log_dir, 'w') as Fout:
                json.dump(log_info, Fout, indent=4)

            if best_pref is None or val_results['overall'] > best_pref:
                best_pref, best_ep = val_results['overall'], ep
                torch.save(model.state_dict(), model_dir)

        if ep >= args.warmup:
            lr_scher.step()

        if args.early_stop >= 5 and ep > max(10, args.early_stop):
            val_his = log_info['test_metric'][-args.early_stop:]
            val_his = [x['trans_acc'] for x in val_his]
            if check_early_stop(val_his):
                print(f'[INFO {worker_idx}] early_stop_break')
                break

    if not verbose:
        return

    print('[BEST EP]', best_ep)
    print('[BEST TEST]', log_info['test_metric'][best_ep])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDP')
    # model definition
    parser.add_argument(
        '--mole_layer', default=5, type=int,
        help='the num layer of molecule gnn'
    )
    parser.add_argument(
        '--dim', type=int, default=256,
        help='the num of dim for the model'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout for model'
    )
    parser.add_argument(
        '--reaction_hop', type=int, default=2,
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

    # training args

    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='the learning rate for training'
    )
    parser.add_argument(
        '--bs', type=int, default=32,
        help='the batch size for training'
    )
    parser.add_argument(
        '--epoch', type=int, default=200,
        help='the number of epochs for training'
    )

    parser.add_argument(
        '--early_stop', type=int, default=0,
        help='the number of epochs for checking early stop, 0 for invalid'
    )

    parser.add_argument(
        '--step_start', type=int, default=10,
        help='the step to start lr decay'
    )
    parser.add_argument(
        '--base_log', type=str, default='log',
        help='the path for contraining log'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='the number of worker for dataloader'
    )

    parser.add_argument(
        '--warmup', type=int, default=0,
        help='the number of epochs for warmup'
    )
    parser.add_argument(
        '--lrgamma', type=float, default=1,
        help='the lr decay rate for training'
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
        '--transductive', action='store_true',
        help='the use transductive training or not'
    )
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path containing the data'
    )

    parser.add_argument(
        '--max_neighbors', type=int, default=15,
        help='max neighbors when sampling'
    )

    # ddp setting

    parser.add_argument(
        '--num_gpus', type=int, default=1,
        help='the number of gpus to train and eval'
    )
    parser.add_argument(
        '--port', type=int, default=12345,
        help='the port for ddp nccl communication'
    )

    # training

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    log_dir, model_dir, token_dir = make_dir(args)

    torch_mp.spawn(
        main_worker, nprocs=args.num_gpus,
        args=(args, log_dir, model_dir, token_dir)
    )
