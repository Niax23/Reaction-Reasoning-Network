import json
import os
import argparse
import numpy as np


def filter_args(args, filter_ag):
    for k, v in filter_ag.items():
        if args.get(k, None) != v:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument(
        '--filter', type=str, default='{}',
        help='a string as filter dict'
    )
    parser.add_argument(
        '--metric', type=str, default='',
        help='the metric as result, left blank if '
        'the log is stored in list'
    )
    parser.add_argument(
        '--topk', type=int, default=10,
        help='the number of results to list'
    )
    parser.add_argument(
        '--min_better', action='store_true',
        help='if the model is better with smaller values'
    )

    args = parser.parse_args()
    args_ft = eval(args.filter)

    all_pfs = []

    for x in os.listdir(args.dir):
        if os.path.exists(os.path.join(args.dir, x, 'log.json')) and\
                os.path.exists(os.path.join(args.dir, x, 'model.pth')):
            with open(os.path.join(args.dir, x, 'log.json')) as Fin:
                INFO = json.load(Fin)

            if not filter_args(INFO['args'], args_ft):
                continue

            if len(INFO['valid_metric']) == 0:
                continue

            v_metr = [x[args.metric] for x in INFO['valid_metric']]\
                if args.metric != '' else INFO['valid_metric']
            if args.min_better:
                best_idx = np.argmin(v_metr)
            else:
                best_idx = np.argmax(v_metr)

            curr_perf = INFO['test_metric'][best_idx]
            sortkey = curr_perf if args.metric == '' else curr_perf[args.metric]
            sortkey = sortkey if args.min_better else -sortkey
            all_pfs.append((INFO['args'], best_idx, x, curr_perf, sortkey))

    all_pfs.sort(key=lambda x: x[-1])
    for arg, ep, ts, pf, _ in all_pfs[:args.topk]:
        print('=====================================================')
        print(f'[args]\n{arg}')
        print(f'[time] {ts}')
        print(f'[epoch] {ep}')
        print(f'[result] {pf}')
