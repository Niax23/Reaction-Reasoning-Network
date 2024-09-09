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


def parse_uspto_condition_data(data_path, verbose=True):
    with open(data_path) as Fin:
        raw_info = json.load(Fin)
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

    all_data = {'train_data': [], 'val_data': [], 'test_data': []}
    name2idx = {k: idx for idx, k in enumerate(all_x)}
    cls_idx = len(name2idx)

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
            'products': element['new']['prod_list']
        }
        all_data[f'{rxn_type}_data'].append(this_line)

    return all_data, name2idx


def parse_dataset_by_smiles_500(json_files):
    # 创建一个空的DataFrame用于存储所有文件的数据
    all_data = pd.DataFrame()

    for json_file in json_files:
        # 读取JSON文件
        with open(json_file, 'r') as file:
            data = json.load(file)

        # 转换为DataFrame并提取需要的列
        df = pd.DataFrame(data)
        if 'canonical_rxn' not in df.columns:
            print(f"Warning: 'canonical_rxn' column not found in {json_file}")
            continue

        # 添加dataset列，以文件名（去掉扩展名）作为值
        df['dataset'] = os.path.splitext(os.path.basename(json_file))[0]

        # 只保留 'canonical_rxn' 和 'dataset' 两列
        df_filtered = df[['canonical_rxn', 'dataset']]

        # 将当前文件的数据追加到总DataFrame中
        all_data = pd.concat([all_data, df_filtered], ignore_index=True)

    # 对 'canonical_rxn' 列应用 edit_function 函数进行修改
    all_data['canonical_rxn'] = all_data['canonical_rxn'].apply(edit)

    # 重新去重
    all_data_deduplicated = all_data.drop_duplicates(
        subset=['canonical_rxn', 'dataset'], keep='first')

    # 重置索引，使id从0开始连续
    all_data_deduplicated = all_data_deduplicated.reset_index(drop=True)

    # 创建 'data_by_id' 字典，以id为键，包含其他字段作为值
    data_by_id = all_data_deduplicated.to_dict(orient='index')

    # 将结果保存到JSON文件
    with open("output500.json", "w") as file:
        json.dump(data_by_id, file, indent=4)

    return data_by_id


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)



def generate_square_subsequent_mask(sz, device='cpu'):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf'))
    # mask = mask.masked_fill(mask == 1, float(0.0)).to(device)
    mask = (mask == 0).to(device)
    return mask


def generate_tgt_mask(tgt, tokenizer, pad='<PAD>', device='cpu'):
    PAD_IDX, siz = tokenizer.token2idx[pad], tgt.shape[1]
    tgt_pad_mask = (tgt == PAD_IDX).to(device)
    tgt_sub_mask = generate_square_subsequent_mask(siz, device)
    return tgt_pad_mask, tgt_sub_mask


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