import torch
from tqdm import tqdm
import numpy as np
from torch.nn.functional import cross_entropy

from utils.data_utils import (
    generate_tgt_mask, correct_trans_output, generate_square_subsequent_mask
    convert_log_into_label, data_eval_trans
)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def calc_trans_loss(trans_pred, trans_lb, ignore_index, lbsm=0.0):
    batch_size, maxl, num_c = trans_pred.shape
    trans_pred = trans_pred.reshape(-1, num_c)
    trans_lb = trans_lb.reshape(-1)

    losses = cross_entropy(
        trans_pred, trans_lb, reduction='none',
        ignore_index=ignore_index, label_smoothing=lbsm
    )
    losses = losses.reshape(batch_size, maxl)
    loss = torch.mean(torch.sum(losses, dim=-1))
    return loss


def train_uspto_condition(loader, model, optimizer, device, warmup=False):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for data in tqdm(loader):
        mol_graphs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_mask, labels, label_types = data

        mol_graphs = mol_graphs.to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        req_mask = req_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

        res = model(
            molecules=mol_graphs, molecule_mask=mol_mask,
            reaction_mask=reaction_mask, required_mask=req_mask,
            edge_index=edge_index, edge_types=edge_types, labels=labels[:, 1:],
            attn_mask=sub_mask, key_padding_mask=None, seq_types=label_types
        )

        loss = calc_trans_loss(res, tgt_out, -1000)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_uspto_condition(loader, model, device):
    model, accs, gt = model.eval(), [], []
    for reac, prod, label in tqdm(loader):
        mol_graphs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_mask, labels, label_types = data

        mol_graphs = mol_graphs.to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        req_mask = req_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

        with torch.no_grad():
            res = model(
                molecules=mol_graphs, molecule_mask=mol_mask,
                reaction_mask=reaction_mask, required_mask=req_mask,
                edge_index=edge_index, edge_types=edge_types,
                labels=labels[:, 1:], attn_mask=sub_mask,
                key_padding_mask=None, seq_types=label_types
            )
            result = convert_log_into_label(res, mod='softmax')

        accs.append(result)
        gt.append(tgt_out)

    accs = torch.cat(accs, dim=0)
    gt = torch.cat(gt, dim=0)

    keys = ['catalyst', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    results, overall = {}, None
    for idx, k in enumerate(keys):
        results[k] = accs[:, idx] == gt[:, idx]
        if idx == 0:
            overall = accs[:, idx] == gt[:, idx]
        else:
            overall &= (accs[:, idx] == gt[:, idx])

    results['overall'] = overall
    results = {k: v.float().mean().item() for k, v in results.items()}
    return results
