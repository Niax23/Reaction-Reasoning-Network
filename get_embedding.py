
import argparse
import json
from tqdm import tqdm
import time
import os
from model import GATBase, PositionalEncoding, AblationModel
import torch
import pickle
from utils.dataset import graph_col_fn
from utils.graph_utils import smiles2graph
def make_rxn_input(reac_list,prod_list):
    reactant_to_id = {}
    reactant_id_counter = 0
    product_to_id = {}
    product_id_counter = 0

    reactant_pairs = []
    product_pairs = []
    reac_molecules, prod_molecules = [], []

    reaction_id = 0
    #reactants, products = rxn.split('>>')
    reactants = reac_list
    products = prod_list
    
    for reactant in reactants:
            if reactant not in reactant_to_id:
                reactant_to_id[reactant] = reactant_id_counter
                reac_molecules.append(reactant)
                reactant_id_counter += 1
            reactant_id = reactant_to_id[reactant]
            reactant_pairs.append((reaction_id, reactant_id))

    for product in products:
            if product not in product_to_id:
                product_to_id[product] = product_id_counter
                prod_molecules.append(product)
                product_id_counter += 1
            product_id = product_to_id[product]
            product_pairs.append((reaction_id, product_id))

        

    product_pairs = torch.LongTensor(product_pairs)
    reactant_pairs = torch.LongTensor(reactant_pairs)
    reac_graphs = [smiles2graph(x, with_amap=False) for x in reac_molecules]
    prod_graphs = [smiles2graph(x, with_amap=False) for x in prod_molecules]
    reac_graphs = graph_col_fn(reac_graphs)
    prod_graphs = graph_col_fn(prod_graphs)

    return reac_graphs, prod_graphs, reactant_pairs, product_pairs,  reactant_id_counter, product_id_counter, 1




def extract_reac_product_template(json_file_path):
    # 打开并读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 初始化结果列表
    result = []
    
    # 处理JSON中的每一项
    for item in data:
        # 提取所需字段
        reac_list = item.get("reac_list", [])
        products = item.get("products", "")
        template_hash = item.get("template_hash", "")
        cano = item.get("canonical_rxn", "")
        
        # 将字段组合为元组并添加到结果列表
        result.append((reac_list, products, template_hash,cano))
    
    return result





if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--log_dir', required=True, type=str,
        help='the path of dir containing logs and ckpt'
    )


    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model'
    )


    parser.add_argument(
        '--device', type=int, default=1,
        help='the device id for traiing, negative for cpu'
    )


    parser.add_argument(
        '--file_name', type=str, default="emb",
        help='the file containing the washed data'
    )
    




    args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')


    

    log_path = os.path.join(args.log_dir, 'log.json')
    model_path = os.path.join(args.log_dir, 'model.pth')
    token_path = os.path.join(args.log_dir, 'token.pkl')
    with open(token_path, 'rb') as Fin:
        remap = pickle.load(Fin)\
    
    with open(log_path, 'r') as f:
        log_file = json.load(f)
    
    config = log_file['args']
    

    mol_gnn = GATBase(
        num_layers=config['mole_layer'], num_heads=config['heads'], dropout=config['dropout'],
        embedding_dim=config['dim'], negative_slope=config['negative_slope']
    )

    pos_env = PositionalEncoding(config['dim'], config['dropout'], maxlen=1024)

    model = AblationModel(
        gnn1=mol_gnn, PE=pos_env, net_dim=config['dim'],
        heads=config['heads'], dropout=config['dropout'], dec_layers=config['decoder_layer'],
        n_words=len(remap), mol_dim=config['dim'],
        with_type=False
    ).to(device)

    model_weight = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weight)
    model = model.eval()


    pair_list = extract_reac_product_template(args.file_name)


    rxn2id = {}
    hash_list = []
    emb_list = []
    print(len(pair_list))
    for idx, (reac_list, products, template_hash,canonical_rxn) in enumerate(tqdm(pair_list)):
        reac_graphs, prod_graphs, reactant_pairs, product_pairs, n_reac, n_prod, n_node = make_rxn_input(reac_list,[products])
        reac_graphs = reac_graphs.to(device)
        prod_graphs = prod_graphs.to(device)
        reactant_pairs = reactant_pairs.to(device)
        product_pairs = product_pairs.to(device)
        with torch.no_grad():
            memory = model.encode(reac_graphs,prod_graphs,n_reac,n_prod,n_node,reactant_pairs,product_pairs ).to('cpu')

        emb_list.append(memory)
        hash_list.append(template_hash)
        rxn2id[canonical_rxn] = idx

    emb_list = torch.cat(emb_list,dim=0)
    final_dict = {'rxn2id':rxn2id,'hash_list':hash_list,'emb':emb_list}

    with open('emb_dicttest.pkl', 'wb') as pickle_file:
        pickle.dump(final_dict, pickle_file)

