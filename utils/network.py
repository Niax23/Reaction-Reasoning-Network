from collections import deque
import torch
import random


class ChemicalReactionNetwork:
    def __init__(self, data):
        """
        初始化化学反应网络并从数据中构建网络。

        参数:
        - data (dict): 反应数据字典，包含reaction_id和canonical_rxn。
        """
        self.substance_adj_list = {}  # 化学物质的邻接表
        self.reaction_adj_list = {}   # 化学反应的邻接表
        self.build(data)

    def add_reaction(self, reaction_info):
        """
        添加一个化学反应到网络中。

        参数:
        - canonical_rxn (str): 化学反应的SMILES序列。
        """

        canonical_rxn = reaction_info['canonical_rxn']
        if canonical_rxn in self.reaction_adj_list:
            return

        self.reaction_adj_list[canonical_rxn] = {
            "reactants": reaction_info['reactants'],
            "products": reaction_info['products']
        }

        for reactant in reaction_info['reactants']:
            if reactant not in self.substance_adj_list:
                self.substance_adj_list[reactant] = []

            self.substance_adj_list[reactant].append({
                "reaction_smiles": canonical_rxn,
                "role": "reactant"
            })

        for product in reaction_info['products']:
            if product not in self.substance_adj_list:
                self.substance_adj_list[product] = []
            self.substance_adj_list[product].append({
                "reaction_smiles": canonical_rxn,
                "role": "product"
            })

    def build(self, data):
        """
        从数据中构建化学反应网络。

        参数:
        - data (list): a list of reaction informations
            containing canonical rxn of the smiles
        """
        for reaction_data in data:
            self.add_reaction(reaction_data)

    def get_substance_neighbors(self, substance, role=None):
        """
        获取与给定化学物质相关的所有化学反应，并根据角色筛选。
        参数:
        - substance (str): 化学物质的SMILES序列。
        - role (str, optional): 可选参数，用于筛选“reactant”或“product”角色。
        返回:
        - list: 相关化学反应的ID列表，或包含角色信息的列表。
        """
        return [
            entry['reaction_smiles'] for entry in
            self.substance_adj_list.get(substance, [])
            if role is None or entry['role'] == role
        ]

    def get_reaction_substances(self, reaction_smiles, role=None):
        """
        获取给定化学反应涉及的所有化学物质，并根据角色筛选。
        参数:
        - reaction_smiles (str): 化学反应的SMILES序列。
        - role (str, optional): 可选参数，用于筛选“reactant”或“product”角色。
        返回:
        - list: 相关化学物质的SMILES列表。
        """
        role = ['reactants', 'products'] if role is None else [role]
        return sum([
            self.reaction_adj_list.get(reaction_smiles, {}).get(x, [])
            for x in role
        ], [])

    def sample_multiple_subgraph(self, reactions, hop, max_neighbors=None):
        assert type(reactions) is list, "Order are import for return results"
        assert all(x in self.reaction_adj_list for x in reactions),\
            "Something that is not an reaction is passed as start vertex"

        visited = set()

        Q, head = [], 0
        for i in reactions:
            Q.append(('reaction', i, 0))
            visited.add(('reaction', i))

        while head < len(Q):
            comp, smiles, depth = Q[head]
            head += 1
            if depth == hop * 2 + 1:
                continue
            if comp == 'reaction':
                for son in self.get_reaction_substances(smiles):
                    if ('molecule', son) not in visited:
                        visited.add(('molecule', son))
                        Q.append(('molecule', son, depth + 1))
            else:
                neighbors = self.get_substance_neighbors(smiles)
                if max_neighbors is not None and \
                        len(neighbors) > max_neighbors:
                    neighbors = random.sample(neighbors, max_neighbors)
                for son in neighbors:
                    # neighbor > max do sample
                    if ('reaction', son) not in visited:
                        visited.add(('reaction', son))
                        Q.append(('reaction', son, depth + 1))

        item2id, smiles_list = {}, []
        edge_index, edge_types = [], []
        for comp, smiles in visited:
            if comp == 'molecule':
                continue
            if smiles not in item2id:
                item2id[smiles] = ('reaction', len(item2id))
            this_id = item2id[smiles][1]
            for x in self.get_reaction_substances(smiles, 'reactants'):
                if x not in item2id:
                    item2id[x] = ('molecule', len(item2id))
                that_id = item2id[x][1]
                edge_index.append((this_id, that_id))
                edge_index.append((that_id, this_id))
                edge_types.extend(['reactant'] * 2)

            for x in self.get_reaction_substances(smiles, 'products'):
                if x not in item2id:
                    item2id[x] = ('molecule', len(item2id))
                that_id = item2id[x][1]
                edge_index.append((this_id, that_id))
                edge_index.append((that_id, this_id))
                edge_types.extend(['product'] * 2)

        molecules = [k for k, v in item2id.items() if v[0] == 'molecule']
        molecules.sort(key=lambda x: item2id[x][1])

        molecule_mask = [0] * len(item2id)
        reaction_mask = [0] * len(item2id)
        for k, v in item2id.items():
            molecule_mask[v[1]] = v[0] == 'molecule'
            reaction_mask[v[1]] = v[0] != 'molecule'

        required_ids = [item2id[x][1] for x in reactions]

        return molecules, edge_index, edge_types, \
            molecule_mask, reaction_mask, required_ids

    def __str__(self):
        return f"ChemicalReactionNetwork with {len(self.substance_adj_list)}"\
            + f" substances and {len(self.reaction_adj_list)} reactions"
