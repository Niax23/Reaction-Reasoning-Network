import pandas
import json
import os
from tqdm import tqdm
from .chemistry_utils import canonical_smiles
import json
import torch
import numpy as np
import random


def clk_x(x):
    return x if x == '' else canonical_smiles(x)


def parse_uspto_condition_mapper(raw_info, verbose=True):
    all_x = set()
    iterx = tqdm(raw_info) if verbose else raw_info
    for i, element in enumerate(iterx):
        cat = clk_x(element['new']['catalyst'])
        sov1 = clk_x(element['new']['solvent1'])
        sov2 = clk_x(element['new']['solvent2'])
        reg1 = clk_x(element['new']['reagent1'])
        reg2 = clk_x(element['new']['reagent2'])
        all_x.add(cat)
        all_x.add(sov1)
        all_x.add(sov2)
        all_x.add(reg1)
        all_x.add(reg2)

    name2idx = {k: idx for idx, k in enumerate(all_x)}
    return name2idx


def parse_uspto_condition_raw(raw_info, name2idx, verbose=True):
    all_data = {'train_data': [], 'val_data': [], 'test_data': []}
    iterx = tqdm(raw_info) if verbose else raw_info
    for i, element in enumerate(iterx):
        rxn_type = element['dataset']
        labels = [
            name2idx[clk_x(element['new']['catalyst'])],
            name2idx[clk_x(element['new']['solvent1'])],
            name2idx[clk_x(element['new']['solvent2'])],
            name2idx[clk_x(element['new']['reagent1'])],
            name2idx[clk_x(element['new']['reagent2'])]
        ]

        this_line = {
            'canonical_rxn': element['new']['canonical_rxn'],
            'label': labels,
            'mapped_rxn': element['new']['mapped_rxn'],
            'reactants': element['new']['reac_list'],
            'products': element['new']['prod_list'],
            'mapped_reac': element['new']['mapped_reac_list'],
            "mapped_prod": element["new"]['mapped_prod_list']
        }
        all_data[f'{rxn_type}_data'].append(this_line)
    return all_data


def parse_uspto_condition_data(data_path, verbose=True):
    with open(data_path) as Fin:
        raw_info = json.load(Fin)
    name2idx = parse_uspto_condition_mapper(raw_info, verbose)
    all_data = parse_uspto_condition_raw(raw_info, name2idx, verbose)

    return all_data, name2idx


def load_uspto_mt_500_gen(data_path, remap=None, part=None):
    if remap is None:
        with open(os.path.join(data_path, 'all_tokens.json')) as F:
            reag_list = json.load(F)
        remap = Tokenizer(reag_list, {'<UNK>', '<CLS>', '<END>', '<PAD>', '`'})

    with open(os.path.join(data_path, 'all_reagents.json')) as F:
        INFO = json.load(F)
    reag_order = {k: idx for idx, k in enumerate(INFO)}

    rxns, px = [[], [], []], 0
    labels = [[], [], []]
    if part is None:
        iterx = ['train.json', 'val.json', 'test.json']
    else:
        iterx = [f'{part}.json']
    for infos in iterx:
        F = open(os.path.join(data_path, infos))
        setx = json.load(F)
        F.close()
        for lin in setx:
            rxns[px].append(lin['new_mapped_rxn'])
            lin['reagent_list'].sort(key=lambda x: reag_order[x])
            lbs = []
            for tdx, x in enumerate(lin['reagent_list']):
                if tdx > 0:
                    lbs.append('`')
                lbs.extend(smi_tokenizer(x))
            labels[px].append(lbs)
        px += 1

    if part is not None:
        return ReactionPredDataset(
            reactions=rxns[0], labels=labels[0],
            cls_id='<CLS>', end_id='<END>'
        ), remap

    train_set = ReactionPredDataset(
        reactions=rxns[0], labels=labels[0],
        cls_id='<CLS>', end_id='<END>'
    )

    val_set = ReactionPredDataset(
        reactions=rxns[1], labels=labels[1],
        cls_id='<CLS>', end_id='<END>'
    )

    test_set = ReactionPredDataset(
        reactions=rxns[2], labels=labels[2],
        cls_id='<CLS>', end_id="<END>"
    )

    return train_set, val_set, test_set, remap


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def correct_trans_output(trans_pred, end_idx, pad_idx):
    batch_size, max_len = trans_pred.shape
    device = trans_pred.device
    x_range = torch.arange(0, max_len, 1).unsqueeze(0)
    x_range = x_range.repeat(batch_size, 1).to(device)

    y_cand = (torch.ones_like(trans_pred).long() * max_len + 12).to(device)
    y_cand[trans_pred == end_idx] = x_range[trans_pred == end_idx]
    min_result = torch.min(y_cand, dim=-1, keepdim=True)
    end_pos = min_result.values
    trans_pred[x_range > end_pos] = pad_idx
    return trans_pred


def data_eval_trans(trans_pred, trans_lb, return_tensor=False):
    batch_size, max_len = trans_pred.shape
    line_acc = torch.sum(trans_pred == trans_lb, dim=-1) == max_len
    line_acc = line_acc.cpu()
    return line_acc if return_tensor else (line_acc.sum().item(), batch_size)


def generate_square_subsequent_mask(sz, device='cpu'):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (mask == 0).to(device)
    return mask


def generate_tgt_mask(tgt, pad_idx, device='cpu'):
    siz = tgt.shape[1]
    tgt_pad_mask = (tgt == pad_idx).to(device)
    tgt_sub_mask = generate_square_subsequent_mask(siz, device)
    return tgt_pad_mask, tgt_sub_mask


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t <= x[0] for t in x[1:])
    return answer


def convert_log_into_label(logits, mod='sigmoid'):
    if mod == 'sigmoid':
        pred = torch.zeros_like(logits)
        pred[logits >= 0] = 1
        pred[logits < 0] = 0
    elif mod == 'softmax':
        pred = torch.argmax(logits, dim=-1)
    else:
        raise NotImplementedError(f'Invalid mode {mod}')
    return pred
