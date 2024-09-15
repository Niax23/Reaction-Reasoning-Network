import json
from tqdm import tqdm

# 加载 JSON 文件
with open('../data/uspto_condition/clean_results.json', 'r') as f:
    data = json.load(f)

# 创建一个列表来存储所有的 SMILES 序列
all_smiles = []

# 遍历每个对象
for entry in tqdm(data):
    # 获取 'new' 部分的 reac_list, prod_list, shared_list
    new_part = entry.get('new', {})
    reac_list = new_part.get('reac_list', [])
    prod_list = new_part.get('prod_list', [])
    shared_list = new_part.get('shared_list', [])
    
    # 将所有 SMILES 序列添加到 all_smiles 列表中
    all_smiles.extend(reac_list)
    all_smiles.extend(prod_list)
    all_smiles.extend(shared_list)

# 去重 SMILES 序列
unique_smiles = list(set(all_smiles))

# 将唯一的 SMILES 序列保存为 JSON 文件
with open('smiles.jsonl', 'w') as f:
    for smiles in unique_smiles:
        f.write(json.dumps(smiles) + '\n')