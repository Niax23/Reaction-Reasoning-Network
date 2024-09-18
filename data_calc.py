from utils import ChemicalReactionNetwork, parse_uspto_condition_data
import argparse
from tqdm import tqdm
import numpy as np
import random
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--bs', required=True, type=int)
    args = parser.parse_args()

    all_data, label_mapper = parse_uspto_condition_data(args.path)
    all_net = ChemicalReactionNetwork(
        all_data['train_data'] + all_data['val_data'] + all_data['test_data']
    )

    mole_nums, total_nums = [], []
    for _ in range(10):
        data = random.choices(all_data['train_data'], k=args.bs)
        infos = all_net.sample_multiple_subgraph([
            x['canonical_rxn'] for x in data
        ], 2)
        mole_nums.append(len(infos[0]))
        total_nums.append(len(infos[3]))

        tlen = []
        for x in infos[0]:
            tlen.append(len(all_net.get_substance_neighbors(x)))

        agmx = np.argmax(tlen)
        print('[max]', tlen[agmx], infos[0][agmx])

    print(mole_nums)

    print(np.max(mole_nums), np.mean(mole_nums))
    print(np.max(total_nums), np.mean(total_nums))

    # mole_nums, total_nums = [], []
    # for _ in tqdm(range(5000)):
    #     data = random.choice(all_data['train_data'])
    #     infos = all_net.sample_multiple_subgraph([data['canonical_rxn']], 2)
    #     mole_nums.append(len(infos[0]))
    #     total_nums.append(len(infos[3]))

    # print(np.max(mole_nums), np.mean(mole_nums))
    # print(np.max(total_nums), np.mean(total_nums))
