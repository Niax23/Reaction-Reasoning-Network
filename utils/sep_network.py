import random


def add_to_dict_list(D, key, value):
    if key not in D:
        D[key] = []
    D[key].append(value)


class SepNetwork:
    def __init__(self, reaction_infos):
        self.adj_infos = {'reaction': {}, 'reactant': {}, 'product': {}}
        for line in reaction_infos:
            cano_rxn = line['canonical_rxn']
            if cano_rxn in self.adj_infos['reaction']:
                continue

            self.adj_infos['reaction'][cano_rxn] = {
                'reactants': line['reactants'],
                'products': line['products'],
                'mapped_reac': line['mapped_reac'],
                'mapped_prod': line['mapped_prod']
            }
            for x in line['reactants']:
                add_to_dict_list(self.adj_infos['reactant'], x, cano_rxn,)

            for x in line['products']:
                add_to_dict_list(self.adj_infos['product'], x, cano_rxn)

    def sample_multiple_subgraph_rxn(self, reactions, hop, max_neighbors=None):
        assert type(reactions) is list, "Order are import for return results"
        assert all(x in self.adj_infos['reaction'] for x in reactions), \
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
                for son in self.adj_infos['reaction'][smiles]['reactants']:
                    if ('reactant', son) not in visited:
                        visited.add(('reactant', son))
                        Q.append(('reactant', son, depth + 1))

                for son in self.adj_infos['reaction'][smiles]['products']:
                    if ('product', son) not in visited:
                        visited.add(('product', son))
                        Q.append(('product', son, depth + 1))

            else:
                neighbors = self.adj_infos[comp][smiles]
                if max_neighbors is not None and \
                        len(neighbors) > max_neighbors:
                    neighbors = random.sample(neighbors, max_neighbors)
                for son in neighbors:
                    # neighbor > max do sample
                    if ('reaction', son) not in visited:
                        visited.add(('reaction', son))
                        Q.append(('reaction', son, depth + 1))

        item2id, edge_index, edge_types = {}, [], []
        reactant_pairs, product_pairs = [], []
        edge_infos, rxn_mapped_infos = [], []
        for comp, smiles in visited:
            if comp != 'reaction':
                continue
            if smiles not in item2id:
                item2id[smiles] = ('reaction', len(item2id))
            this_id = item2id[smiles][1]
            all_prod = []

            for x in self.adj_infos['reaction'][smiles]['products']:
                if x not in item2id:
                    item2id[x] = ('product', len(item2id))
                that_id = item2id[x][1]
                edge_index.append((this_id, that_id))
                edge_index.append((that_id, this_id))
                edge_types.extend(['product'] * 2)
                product_pairs.append((this_id, that_id))

            for x in self.adj_infos['reaction'][smiles]['reactants']:
                if x not in item2id:
                    item2id[x] = ('reactant', len(item2id))
                that_id = item2id[x][1]
                edge_index.append((this_id, that_id))
                edge_index.append((that_id, this_id))
                edge_types.extend(['reactant'] * 2)
                edge_infos.extend([(smiles, x)] * 2)
                reactant_pairs.append((this_id, that_id))

        molecules, molecule_ids, rxn_sms, rxn_ids, mts = [], [], [], [], []

        for k, (rtype, rid) in item2id.items():
            if rtype != 'reaction':
                molecule_ids.append(rid)
                molecules.append(k)
                mts.append(rtype)
            else:
                rxn_ids.append(rid)
                rxn_sms.append(k)
                rxn_mapped_infos.append((
                    self.adj_infos['reaction'][k]['mapped_reac'],
                    self.adj_infos['reaction'][k]['mapped_prod']
                ))

        required_ids = [item2id[x][1] for x in reactions]
        n_node = len(item2id)

        return molecules, mts, molecule_ids, rxn_sms, rxn_mapped_infos, \
            rxn_ids, edge_index, edge_types, edge_infos, required_ids, \
            reactant_pairs, product_pairs, n_node

    def __str__(self):
        return f"ReactionNetwork with {len(self.adj_infos['reactant'])}"\
            + f" reactants, {len(self.adj_infos['product'])} products" \
            + f" and {len(self.reaction_adj_list)} reactions"
