from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import numpy as np
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--n_jobs', default=8, type=int)

    args = parser.parse_args()

    label2idx = {}

    feat = torch.load(args.data, map_location='cpu')
    for x in feat['train']['labels']:
        if x not in label2idx:
            label2idx[x] = len(label2idx)

    train_x = feat['train']['features'].numpy()
    train_y = [label2idx[x] for x in feat['train']['labels']]

    test_x = feat['test']['features'].numpy()
    test_y = [label2idx[x] for x in feat['test']['labels']]
    print('start training')

    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)

    model = RandomForestClassifier(
        verbose=True, n_jobs=args.n_jobs, n_estimators=100,
        random_state=args.seed
    )
    model.fit(train_x, train_y)
    result = model.predict(test_x)

    print('[Accuracy]', accuracy_score(test_y, result))
