import torch
from tqdm import tqdm
import numpy as np
from torch.nn.functional import cross_entropy

from utils.data_utils import (
    generate_tgt_mask, correct_trans_output, generate_square_subsequent_mask,
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
            req_ids, labels, label_types = data

        mol_graphs = mol_graphs.to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

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
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_uspto_condition(loader, model, device):
    model, accs, gt = model.eval(), [], []
    for data in tqdm(loader):
        mol_graphs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_ids, labels, label_types = data

        mol_graphs = mol_graphs.to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

        with torch.no_grad():
            res = model(
                molecules=mol_graphs, molecule_mask=mol_mask,
                reaction_mask=reaction_mask, required_ids=req_ids,
                edge_index=edge_index, edge_types=edge_types,
                labels=labels[:, :-1], attn_mask=sub_mask,
                key_padding_mask=None, seq_types=label_types
            )
            result = convert_log_into_label(res, mod='softmax')

        accs.append(result)
        gt.append(labels)

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


def train_uspto_condition_pretrain(loader, model, optimizer, emb_dict, device, warmup=False):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for data in tqdm(loader):
        mol_strs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_ids, labels, label_types = data

        mole_feats = torch.stack([emb_dict[mole]
                                 for mole in mol_strs]).to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

        res = model(
            molecules=mole_feats, molecule_mask=mol_mask, required_ids=req_ids,
            reaction_mask=reaction_mask,  edge_index=edge_index,
            edge_types=edge_types, labels=labels[:, :-1],
            attn_mask=sub_mask, key_padding_mask=None, seq_types=label_types
        )

        loss = calc_trans_loss(res, labels, -1000)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_uspto_condition_pretrain(loader, model, emb_dict, device):
    model, accs, gt = model.eval(), [], []
    for data in tqdm(loader):
        mol_strs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_ids, labels, label_types = data

        mole_feats = torch.stack([emb_dict[mole]
                                 for mole in mol_strs]).to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

        with torch.no_grad():
            res = model(
                molecules=mole_feats, molecule_mask=mol_mask,
                reaction_mask=reaction_mask, required_ids=req_ids,
                edge_index=edge_index, edge_types=edge_types,
                labels=labels[:, :-1], attn_mask=sub_mask,
                key_padding_mask=None, seq_types=label_types
            )
            result = convert_log_into_label(res, mod='softmax')

        accs.append(result)
        gt.append(labels)

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


def train_uspto_condition_react_emb(loader, model, optimizer, emb_dict, network, device, warmup=False):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for data in tqdm(loader):
        mol_strs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_ids, smiles_list, id_list, labels, label_types = data

        mole_feats = torch.stack([emb_dict[mole]
                                 for mole in mol_strs]).to(device)

        reactant_list = [network.get_reaction_substances(
            reaction, 'reactants')for reaction in smiles_list]
        reac_feats = []
        for reactants in reactant_list:
            reac_feats.append(torch.mean(torch.stack(
                [emb_dict[reactant] for reactant in reactants]), dim=0))
        reac_feats = torch.stack(reac_feats)

        product_list = [network.get_reaction_substances(
            reaction, 'products')for reaction in smiles_list]
        prod_feats = []
        for products in product_list:
            prod_feats.append(torch.mean(torch.stack(
                [emb_dict[product] for product in products]), dim=0))
        prod_feats = torch.stack(prod_feats)
        reac_feats = reac_feats.squeeze(1)
        prod_feats = prod_feats.squeeze(1)

        reaction_feats = torch.cat((reac_feats, prod_feats), dim=1).to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

        res = model(
            molecules=mole_feats, molecule_mask=mol_mask, required_ids=req_ids,
            reaction_mask=reaction_mask, reaction_feats=reaction_feats, reaction_ids=id_list,  edge_index=edge_index,
            edge_types=edge_types, labels=labels[:, :-1],
            attn_mask=sub_mask, key_padding_mask=None, seq_types=label_types
        )

        loss = calc_trans_loss(res, labels, -1000)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_uspto_condition_react_emb(loader, model, emb_dict, network, device):
    model, accs, gt = model.eval(), [], []
    for data in tqdm(loader):
        mol_strs, edge_index, edge_types, mol_mask, reaction_mask, \
            req_ids, smiles_list, id_list, labels, label_types = data

        mole_feats = torch.stack([emb_dict[mole]
                                 for mole in mol_strs]).to(device)

        reactant_list = [network.get_reaction_substances(
            reaction, 'reactants')for reaction in smiles_list]
        reac_feats = []
        for reactants in reactant_list:
            reac_feats.append(torch.mean(torch.stack(
                [emb_dict[reactant] for reactant in reactants]), dim=0))
        reac_feats = torch.stack(reac_feats)

        product_list = [network.get_reaction_substances(
            reaction, 'products')for reaction in smiles_list]
        prod_feats = []
        for products in product_list:
            prod_feats.append(torch.mean(torch.stack(
                [emb_dict[product] for product in products]), dim=0))
        prod_feats = torch.stack(prod_feats)
        reac_feats = reac_feats.squeeze(1)
        prod_feats = prod_feats.squeeze(1)
        reaction_feats = torch.cat((reac_feats, prod_feats), dim=1).to(device)
        edge_index = edge_index.to(device)
        mol_mask = mol_mask.to(device)
        reaction_mask = reaction_mask.to(device)
        labels = labels.to(device)
        label_types = label_types.to(device)

        sub_mask = generate_square_subsequent_mask(5, device)

        with torch.no_grad():
            res = model(
                molecules=mole_feats, molecule_mask=mol_mask, required_ids=req_ids,
                reaction_mask=reaction_mask, reaction_feats=reaction_feats, reaction_ids=id_list,  edge_index=edge_index,
                edge_types=edge_types, labels=labels[:, :-1],
                attn_mask=sub_mask, key_padding_mask=None, seq_types=label_types
            )
            result = convert_log_into_label(res, mod='softmax')

        accs.append(result)
        gt.append(labels)

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
