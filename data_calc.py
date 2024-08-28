from utils import ChemicalReactionNetwork, parse_uspto_condition_data
import argparse
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()

    all_data, label_mapper = parse_uspto_condition_data(args.path)
    all_net = ChemicalReactionNetwork(
        all_data['train_data'] + all_data['val_data'] + all_data['test_data']
    )

    mole_nums, total_nums = [], []
    for data in all_data['train_data']:
        infos = all_net.sample_multiple_subgraph([data['canonical_rxn']], 1)
        mole_nums.append(len(infos[0]))
        total_nums.append(len(infos[2]))

    print(np.max(mole_nums), np.mean(mole_nums))
    print(np.max(total_nums), np.mean(total_nums))

    mole_nums, total_nums = [], []
    for data in all_data['train_data']:
        infos = all_net.sample_multiple_subgraph([data['canonical_rxn']], 2)
        mole_nums.append(len(infos[0]))
        total_nums.append(len(infos[2]))

    print(np.max(mole_nums), np.mean(mole_nums))
    print(np.max(total_nums), np.mean(total_nums))
