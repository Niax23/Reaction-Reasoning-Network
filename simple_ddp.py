import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader


class A:
    def __init__(self, data):
        self.data = data

    def get_x(self, key):
        return self.data[key]


class B:
    def __init__(self, para, ac):
        self.para = para
        self.ac = ac

    def fll(self, x):
        return [self.ac.get_x(t) * self.para for t in x]


def main_worker(worker_idx, n_gpus):
    print(f'[INFO] Process {worker_idx} start')
    torch_dist.init_process_group(
        backend='nccl', init_method=f'tcp://127.0.0.1:14443',
        world_size=n_gpus, rank=worker_idx
    )

    Ax = A({i: i + 1 for i in range(10)})
    Bx = B(1.5, Ax)

    train_sampler = DistributedSampler(list(range(10)), shuffle=True)

    loader = DataLoader(
        list(range(10)), batch_size=4, num_workers=4, shuffle=False,
        collate_fn=Bx.fll, pin_memory=True, sampler=train_sampler
    )

    for x in loader:
        print(x)


if __name__ == '__main__':
    torch_mp.spawn(main_worker, nprocs=2, args=(2, ))
