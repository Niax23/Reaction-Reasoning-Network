import torch
from torch import nn
from graph_utils import smiles2pyg
from molecular_embedding.sparse_backBone import GATBase
from network_embedding.NetworkGNN import RxnNet_GNN
from torch_geometric.data import Data


def graph2batch(
    node_feat: torch.Tensor, batch_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_node = batch_mask.shape
    answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    answer = answer.to(node_feat.device)
    answer[batch_mask] = node_feat
    return answer

def make_graph(node_feat, edge_index, edge_feat):
    return Data(x=node_feat,edge_index=edge_index,edge_attr=edge_feat)


class MyModel(nn.Module):
    def __init__(self,dim):
        super(MyModel, self).__init__()
        self.gnn1 = GATBase(num_layers=4,num_heads=4,embedding_dim=dim,dropout=0.7,negative_slope=0.2,n_class=None)  # 第一个GNN模型
        self.gnn2 = RxnNet_GNN(num_layers=4,num_heads=4,embedding_dim=dim,dropout=0.7,negative_slope=0.2,n_class=None)  # 第二个GNN模型
        self.dim = dim  # 节点特征维度
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.edge_emb = nn.ParameterDict({
                'reactant': nn.Parameter(torch.randn(1, dim)),
                'product': nn.Parameter(torch.randn(1, dim))
        })
        self.pooler = nn.MultiheadAttention(dim, num_heads=4)


    def forward(self, smiles_list, molecule_ids, reaction_ids, edge_index, big_graph_edge_feat):
        # 将SMILES转换为分子图
        molecule_graphs = smiles2pyg(smiles_list[molecule_ids])
        
        # 使用gnn1处理分子图并池化以获得分子特征
        x,edge_feats = self.gnn1(molecule_graphs)

        memory = graph2batch(x, molecule_graphs['batch_mask']) # 变成(batchsize,numnodes,dim)形状
        memory_mask = molecule_graphs['batch_mask']
        pool_key = self.pool_keys.repeat(memory.shape[0], 1, 1)

        molecule_features , p_attn = self.pooler(query=pool_key, key=memory, value=memory,key_padding_mask=memory_mask)
        n_nodes = len(smiles_list)
        # 初始化大图的节点特征
        big_graph_features = torch.zeros(n_nodes, self.dim)
        
        # 将分子特征分配给大图中的相应节点
        big_graph_features[molecule_ids] = molecule_features
        
        reaction_embeddings = self.pool_keys.repeat(len(reaction_ids),1,1)

        # 将可学习的反应特征分配给大图中的反应节点
        big_graph_features[reaction_ids] = reaction_embeddings
        edge_feats = [self.edge_emb[x] for x in edge_feats]
        
        # 构建大图
        G2 = make_graph(big_graph_features, edge_index, big_graph_edge_feat)
        
        # 使用gnn2处理大图
        output = self.gnn2(G2)
        
        return output