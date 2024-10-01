import torch
from tqdm import tqdm
import numpy as np
from torch.nn.functional import cross_entropy

from utils.data_utils import (
    generate_tgt_mask, correct_trans_output, generate_square_subsequent_mask,
    convert_log_into_label, data_eval_trans
)

from training import warmup_lr_scheduler, calc_trans_loss

import torch.distributed as torch_dist
from enum import Enum


class Summary(Enum):
    NONE, SUM, AVERAGE, COUNT = 0, 1, 2, 3


class MetricCollector(object):
    def __init__(self, name, type_fmt=':f', summary_type=Summary.AVERAGE):
        super(MetricCollector, self).__init__()
        self.name, self.type_fmt = name, type_fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val, self.sum, self.cnt, self.avg = [0] * 4

    def update(self, val, num=1):
        self.val = val
        self.sum += val
        self.cnt += num
        self.avg = self.sum / self.cnt

    def all_reduce(self, device):
        infos = torch.FloatTensor([self.sum, self.cnt]).to(device)
        torch_dist.all_reduce(infos, torch_dist.ReduceOp.SUM)
        self.sum, self.cnt = infos.tolist()
        self.avg = self.sum / self.cnt

    def __str__(self):
        return ''.join([
            '{name}: {val', self.type_fmt, '} avg: {avg', self.type_fmt, '}'
        ]).format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {cnt:.3f}'
        else:
            raise ValueError(f'Invaild summary type {self.summary_type} found')

        return fmtstr.format(**self.__dict__)

    def get_value(self):
        if self.summary_type is Summary.AVERAGE:
            return self.avg
        elif self.summary_type is Summary.SUM:
            return self.sum
        elif self.summary_type is Summary.COUNT:
            return self.cnt
        else:
            raise ValueError(
                f'Invaild summary type {self.summary_type} '
                'for get_value()'
            )


class MetricManager(object):
    def __init__(self, metrics):
        super(MetricManager, self).__init__()
        self.metrics = metrics

    def all_reduct(self, device):
        for idx in range(len(self.metrics)):
            self.metrics[idx].all_reduce(device)

    def summary_all(self, split_string='  '):
        return split_string.join(x.summary() for x in self.metrics)

    def get_all_value_dict(self):
        return {x.name: x.get_value() for x in self.metrics}


def ddp_train_uspto_condition(
    loader, model, optimizer, device, warmup=False, verbose=False
):
    model = model.train()
    loss_cur = MetricCollector('loss', type_fmt=':.3f')
    manager = MetricManager([loss_cur])
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    iterx = tqdm(loader) if verbose else loader

    for data in iterx:
        mol_graphs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_ids, labels, label_types = data

        mol_graphs = mol_graphs.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        mol_mask = mol_mask.to(device, non_blocking=True)
        reaction_mask = reaction_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_types = label_types.to(device, non_blocking=True)

        sub_mask = generate_square_subsequent_mask(5, 'cpu')
        sub_mask = sub_mask.to(device, non_blocking=True)

        res = model(
            molecules=mol_graphs, molecule_mask=mol_mask, required_ids=req_ids,
            reaction_mask=reaction_mask,  edge_index=edge_index,
            edge_types=edge_types, labels=labels[:, :-1],
            attn_mask=sub_mask, key_padding_mask=None, seq_types=label_types
        )

        loss = calc_trans_loss(res, labels, -1000)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_cur.update(loss.item())
        if warmup:
            warmup_sher.step()

        if verbose:
            iterx.set_postfix_str(manager.summary_all())

    return manager


def ddp_eval_uspto_condition(loader, model, device, verbose=False):
    model = model.eval()
    cat_acc = MetricCollector('catalyst', type_fmt=':.2f')
    sov1_acc = MetricCollector('solvent1', type_fmt=":.2f")
    sov2_acc = MetricCollector('solvent2', type_fmt=':.2f')
    reg1_acc = MetricCollector('reagent1', type_fmt=':.2f')
    reg2_acc = MetricCollector('reagent2', type_fmt=':.2f')
    ov = MetricCollector('overall', type_fmt=':.2f')
    man = MetricManager([cat_acc, sov1_acc, sov2_acc, reg1_acc, reg2_acc, ov])
    keys = ['catalyst', 'solvent1', 'solvent2', 'reagent1', 'reagent2']

    iterx = tqdm(loader) if verbose else loader
    for data in iterx:
        mol_graphs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_ids, labels, label_types = data

        mol_graphs = mol_graphs.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        mol_mask = mol_mask.to(device, non_blocking=True)
        reaction_mask = reaction_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_types = label_types.to(device, non_blocking=True)

        sub_mask = generate_square_subsequent_mask(5, 'cpu')
        sub_mask = sub_mask.to(device, non_blocking=True)

        with torch.no_grad():
            res = model(
                molecules=mol_graphs, molecule_mask=mol_mask,
                reaction_mask=reaction_mask, required_ids=req_ids,
                edge_index=edge_index, edge_types=edge_types,
                labels=labels[:, :-1], attn_mask=sub_mask,
                key_padding_mask=None, seq_types=label_types
            )
            result = convert_log_into_label(res, mod='softmax')

        ovr = None
        for idx, k in enumerate(keys):
            pt = result[:, idx] == labels[:, idx]
            A, B = pt.sum().item(), pt.shape[0]
            man.metrics[idx].update(val=A, num=B)
            ovr = pt if ovr is None else (ovr & pt)
        ov.update(ovr.sum().item(), ovr.shape[0])
        if verbose:
            iterx.set_postfix_str(man.summary_all(split_string=','))

    return man


def ddp_train_uspto_condition_full(
    loader, model, optimizer, device, warmup=False, verbose=False
):
    model = model.train()
    loss_cur = MetricCollector('loss', type_fmt=':.3f')
    manager = MetricManager([loss_cur])
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    iterx = tqdm(loader) if verbose else loader

    for data in iterx:
        mole_graphs, mts, molecule_ids, rxn_ids, edge_index, \
            edge_types, semi_graphs, semi_keys, smkey2idx, required_ids,\
            reactant_pairs, product_pairs, n_node, labels, label_types = data

        mole_graphs = mole_graphs.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_types = label_types.to(device, non_blocking=True)
        sub_mask = generate_square_subsequent_mask(5, 'cpu')
        sub_mask = sub_mask.to(device, non_blocking=True)
        reactant_pairs = reactant_pairs.to(device, non_blocking=True)
        product_pairs = product_pairs.to(device, non_blocking=True)
        semi_graphs = semi_graphs.to(device, non_blocking=True)
        semi_keys = [tuple(x) for x in semi_keys]

        res = model(
            mole_graphs=mole_graphs, mts=mts, molecule_ids=molecule_ids,
            rxn_ids=rxn_ids, required_ids=required_ids, edge_index=edge_index,
            edge_types=edge_types, semi_graphs=semi_graphs,
            semi_keys=semi_keys, semi_key2idxs=smkey2idx, n_nodes=n_node,
            labels=labels[:, :-1], attn_mask=sub_mask,
            reactant_pairs=reactant_pairs, product_pairs=product_pairs,
            key_padding_mask=None, seq_types=label_types,
        )

        loss = calc_trans_loss(res, labels, -1000)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_cur.update(loss.item())
        if warmup:
            warmup_sher.step()

        if verbose:
            iterx.set_postfix_str(manager.summary_all())

    return manager


def ddp_eval_uspto_condition_full(loader, model, device, verbose=False):
    model = model.eval()
    cat_acc = MetricCollector('catalyst', type_fmt=':.2f')
    sov1_acc = MetricCollector('solvent1', type_fmt=":.2f")
    sov2_acc = MetricCollector('solvent2', type_fmt=':.2f')
    reg1_acc = MetricCollector('reagent1', type_fmt=':.2f')
    reg2_acc = MetricCollector('reagent2', type_fmt=':.2f')
    ov = MetricCollector('overall', type_fmt=':.2f')
    man = MetricManager([cat_acc, sov1_acc, sov2_acc, reg1_acc, reg2_acc, ov])
    keys = ['catalyst', 'solvent1', 'solvent2', 'reagent1', 'reagent2']

    iterx = tqdm(loader) if verbose else loader
    for data in iterx:
        mole_graphs, mts, molecule_ids, rxn_ids, edge_index, \
            edge_types, semi_graphs, semi_keys, smkey2idx, required_ids,\
            reactant_pairs, product_pairs, n_node, labels, label_types = data

        mole_graphs = mole_graphs.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_types = label_types.to(device, non_blocking=True)
        sub_mask = generate_square_subsequent_mask(5, 'cpu')
        sub_mask = sub_mask.to(device, non_blocking=True)
        reactant_pairs = reactant_pairs.to(device, non_blocking=True)
        product_pairs = product_pairs.to(device, non_blocking=True)
        semi_graphs = semi_graphs.to(device, non_blocking=True)
        semi_keys = [tuple(x) for x in semi_keys]

        with torch.no_grad():
            res = model(
                mole_graphs=mole_graphs, mts=mts, molecule_ids=molecule_ids,
                rxn_ids=rxn_ids, required_ids=required_ids,
                edge_index=edge_index, edge_types=edge_types,
                semi_graphs=semi_graphs, semi_keys=semi_keys,
                semi_key2idxs=smkey2idx, n_nodes=n_node,
                labels=labels[:, :-1], attn_mask=sub_mask,
                reactant_pairs=reactant_pairs, product_pairs=product_pairs,
                key_padding_mask=None, seq_types=label_types,
            )
            result = convert_log_into_label(res, mod='softmax')

        ovr = None
        for idx, k in enumerate(keys):
            pt = result[:, idx] == labels[:, idx]
            A, B = pt.sum().item(), pt.shape[0]
            man.metrics[idx].update(val=A, num=B)
            ovr = pt if ovr is None else (ovr & pt)
        ov.update(ovr.sum().item(), ovr.shape[0])
        if verbose:
            iterx.set_postfix_str(man.summary_all(split_string=','))

    return man
