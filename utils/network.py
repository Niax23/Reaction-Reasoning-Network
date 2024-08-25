from collections import deque
import torch


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

    def parse_smiles_reaction_to_list(self, smiles_reaction):
        try:
            reactants, products = smiles_reaction.split('>>')
            reactants_list = reactants.split('.')
            products_list = products.split('.')
            unique_reactants_list = list(set(reactants_list))
            unique_products_list = list(set(products_list))
            return unique_reactants_list, unique_products_list
        except ValueError:
            raise ValueError("输入的SMILES序列格式不正确，无法找到 '>>' 分隔符。")

    def add_reaction(self, canonical_rxn):
        """
        添加一个化学反应到网络中。

        参数:
        - canonical_rxn (str): 化学反应的SMILES序列。
        """
        if canonical_rxn in self.reaction_adj_list:
            return

        reactants_list, products_list = \
            self.parse_smiles_reaction_to_list(canonical_rxn)

        self.reaction_adj_list[canonical_rxn] = {
            "reactants": reactants_list,
            "products": products_list
        }

        for reactant in reactants_list:
            if reactant not in self.substance_adj_list:
                self.substance_adj_list[reactant] = []

            self.substance_adj_list[reactant].append({
                "reaction_smiles": canonical_rxn,
                "role": "reactant"
            })

        for product in products_list:
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
            canonical_rxn = reaction_data['canonical_rxn']
            self.add_reaction(canonical_rxn)

    def get_substance_neighbors(self, substance, role=None):
        """
        获取与给定化学物质相关的所有化学反应，并根据角色筛选。
        参数:
        - substance (str): 化学物质的SMILES序列。
        - role (str, optional): 可选参数，用于筛选“reactant”或“product”角色。
        返回:
        - list: 相关化学反应的ID列表，或包含角色信息的列表。
        """
        if role:
            return [
                entry['reaction_smiles'] for entry in
                self.substance_adj_list.get(substance, [])
                if entry['role'] == role
            ]
        return self.substance_adj_list.get(substance, [])

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
        return sum(
            self.reaction_adj_list.get(reaction_smiles, {}).get(x, [])
            for x in role
        )

    def get_nhop_subgraph(self, start_vertex, n):
        """
        获取以某个反应为中心的n跳范围内的子图。
        参数:
        - start_vertex (str): 起始反应的SMILES序列。
        - n (int): 跳数范围。
        返回:
        - tuple: 包含smiles_list, molecule_ids, reaction_ids, edge_index
        """
        if start_vertex not in self.reaction_adj_list:
            raise ValueError(f"起始顶点 {start_vertex} 不是一个反应顶点。")

        queue = deque([(start_vertex, 0)])  # 存储顶点和当前跳数
        visited = set([start_vertex])
        subgraph_vertices = set([start_vertex])
        subgraph_edges = set()  # 使用集合避免重复边

        while queue:
            current_vertex, depth = queue.popleft()
            print("depth: " + str(depth))
            if depth <= n:
                reactants = self.reaction_adj_list[current_vertex]["reactants"]
                products = self.reaction_adj_list[current_vertex]["products"]

                for substance in reactants + products:
                    if substance not in visited:
                        visited.add(substance)
                        subgraph_vertices.add(substance)

                    for entry in self.substance_adj_list[substance]:
                        reaction_smiles = entry["reaction_smiles"]
                        if reaction_smiles in visited:
                            subgraph_edges.add(
                                (substance, reaction_smiles, entry["role"]))
                        elif depth + 1 <= n:
                            queue.append((reaction_smiles, depth + 1))
                            visited.add(reaction_smiles)
                            subgraph_vertices.add(reaction_smiles)
                            subgraph_edges.add(
                                (substance, reaction_smiles, entry["role"]))
        smiles_list = list(subgraph_vertices)
        molecule_ids = [i for i, smiles in enumerate(
            smiles_list) if smiles in self.substance_adj_list]
        reaction_ids = [i for i, smiles in enumerate(
            smiles_list) if smiles in self.reaction_adj_list]

      # 构建 edge_index 和 edge_attr
        edge_index = [[], []]
        edge_attr_list = []
        for source_smiles, target_smiles, role in subgraph_edges:
            source_index = smiles_list.index(source_smiles)
            target_index = smiles_list.index(target_smiles)
            edge_index[0].append(source_index)
            edge_index[1].append(target_index)
            edge_index[1].append(source_index)
            edge_index[0].append(target_index)

            # 构建 edge_attr 对应的特征，这里 role 可以被替换为实际的边特征
            edge_attr_list.append(role)  # 例如，role 可以是 "reactant" 或 "product"
            edge_attr_list.append(role)

        edge_index = torch.tensor(edge_index, dtype=torch.long)  # 转换为张量
        edge_attr = torch.tensor(
            edge_attr_list, dtype=torch.float)  # 你可以根据实际需要调整数据类型

        return {
            "smiles_list": smiles_list,
            "molecule_ids": molecule_ids,
            "reaction_ids": reaction_ids,
            "edge_index": edge_index,
            "edge_attr": edge_attr
        }

    def sample_batch(self, reactions, hop):
        return [self.get_nhop_subgraph(reaction, hop) for reaction in reactions]

    def sample_multiple_subgraph(self, reactions, hop):
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
                for son in self.get_substance_neighbors(smiles):
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
        required_mask = [0] * len(item2id)
        for k, v in item2id.items():
            molecule_mask[v[1]] = v[0] == 'molecule'
            reaction_mask[v[1]] = v[0] != 'molecule'

        for x in reactions:
            required_mask[item2id[x][1]] = True

        return molecules, edge_index, edge_types, \
            molecule_mask, reaction_mask, required_mask

    def __str__(self):
        return f"ChemicalReactionNetwork with {len(self.substance_adj_list)}"\
            + f" substances and {len(self.reaction_adj_list)} reactions"
