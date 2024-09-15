from pretrain_gnn import GNN_graph
import json
from tqdm import tqdm
from pretrain_gnn_utils import smiles2graph
from torch_geometric.data import Data
import torch
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smiles_list = []
with open('smiles.jsonl', 'r') as f:
    for line in tqdm(f):
        smiles = json.loads(line.strip())
        smiles_list.append(smiles)


emb_dict = {}
model = GNN_graph(num_layer=5, emb_dim=300).to(device)
model.from_pretrained("supervised_contextpred.pth")
model.eval()


with torch.no_grad():
    for smiles in tqdm(smiles_list):
        graph = smiles2graph(smiles)
        data = Data(
            x=torch.from_numpy(graph['node_feat']).to(device),          # 节点特征
            edge_index=torch.from_numpy(
                graph['edge_index']).to(device),  # 边的连接
            edge_attr=torch.from_numpy(graph['edge_feat']).to(device),   # 边的特征
            batch=torch.LongTensor(
                [0] * graph['node_feat'].shape[0]).to(device)
        )
        with torch.no_grad():
            emb_dict[smiles] = model(data)

with open('embeddings_dict.pkl', 'wb') as f:
    pickle.dump(emb_dict, f)
