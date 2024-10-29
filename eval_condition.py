import torch
from utils import canonical_rxn
import argparse
import json
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_path', required=True, type=str,
        help='the path containing results'
    )
    parser.add_argument(
        '--beams', type=int, default=10,
        help='the number of beams for beam search'
    )
    args = parser.parse_args()

    modified = False

    results = []
    to_display = [1, 3, 5, 10, 20, 30, 50]

    with open(args.file_path) as Fin:
        INFO = json.load(Fin)

    real_answer = {}
    for k, v in INFO['rxn2gt'].items():
        real_answer[k] = set(tuple(x) for x in v)

    for line in tqdm(INFO['answer']):
        if 'query_key' not in line:
            line['query_key'] = canonical_rxn(line['query'])
            modified = True

        this_line = np.zeros(args.beams)
        for idx, (prob, res) in enumerate(line['prob_answer']):
            res = tuple(res)
            if res in real_answer[line['query_key']]:
                this_line[idx:] += 1
                break

        results.append(this_line)

    results = np.stack(results, axis=0)
    results = np.mean(results, axis=0)
    if modified:
        with open(args.file_path, 'w') as Fout:
            json.dump(INFO, Fout)
    print('[Model Config]')
    print(INFO['args'])
    print('[Result]')
    for p in to_display:
        if p <= args.beams:
            print(f'[top-{p}]', float(results[p - 1]))
        else:
            break