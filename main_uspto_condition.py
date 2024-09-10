import torch

from utils.data_utils import (
    fix_seed, parse_uspto_condition_data, parse_dataset_by_smiles_500
)

from model import GATBase, MyModel, RxnNetworkGNN
from training import train_uspto_condition, eval_uspto_condition
import argparse
import os


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
        'the number of epochs for checking early stop, 0 for invalid'
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
        '--num_worker', type=int, default=8,
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

    # data config

    parser.add_argument(
        '--transductive', action='store_true',
        help='the use transductive training or not'
    )
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path containing the data'
    )

    args = parser.parse_args()

    fix_seed(args.seed)
