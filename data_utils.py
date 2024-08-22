import pandas as pd
import json
import os


def edit(smiles_sequence):
    # 将SMILES序列分成三个部分：反应物、中间体、产物
    reactants, _, products = smiles_sequence.partition('>')
    
    # 再次分割，以去掉反应物之后的所有内容，仅保留反应物和产物
    _, _, products = products.partition('>')
    
    # 将反应物和产物重新组合，并去掉中间体
    edited_sequence = f"{reactants}>>{products}"
    
    return edited_sequence



def parse_dataset_by_smiles_condition():
    # 仅保留不同的反应式 不考虑反应条件


    df = pd.read_csv('USPTO_condition_mapped.csv')
    
    # 只考虑 'canonical_rxn' 和 'dataset' 两列进行去重
    df_deduplicated = df.drop_duplicates(subset=['canonical_rxn', 'dataset'], keep='first')
    
    # 重置索引，使id从0开始连续
    df_deduplicated = df_deduplicated.reset_index(drop=True)
    
    # 创建 'data_by_id' 字典
    data_by_id = df_deduplicated[['canonical_rxn', 'dataset']].to_dict(orient='index')
    
    with open("output.json", "w") as file:
        json.dump(data_by_id, file, indent=4)
    return data_by_id


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
    all_data_deduplicated = all_data.drop_duplicates(subset=['canonical_rxn', 'dataset'], keep='first')
    
    # 重置索引，使id从0开始连续
    all_data_deduplicated = all_data_deduplicated.reset_index(drop=True)
    
    # 创建 'data_by_id' 字典，以id为键，包含其他字段作为值
    data_by_id = all_data_deduplicated.to_dict(orient='index')
    
    # 将结果保存到JSON文件
    with open("output500.json", "w") as file:
        json.dump(data_by_id, file, indent=4)
    
    return data_by_id