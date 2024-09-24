from rdkit import Chem
import numpy as np


def canonical_smiles(x):
    mol = Chem.MolFromSmiles(x)
    return x if mol is None else Chem.MolToSmiles(mol)


def remove_am(x, canonical=True):
    mol = Chem.MolFromSmiles(x)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    if not canonical:
        return Chem.MolToSmiles(mol, canonical=False)
    else:
        return canonical_smiles(Chem.MolToSmiles(mol))


def get_semi_reaction(mapped_rxn, trans_fn, add_pesudo_node=False):
    def add_reidx(mapper, k):
        if k not in mapper:
            mapper[k] = len(mapper)

    reac, prod = mapped_rxn.split('>>')
    prod_mol = Chem.MolFromSmiles(prod)

    assert all(x.GetAtomMapNum() != 0 for x in prod_mol.GetAtoms()),\
        'Require Atom Mapping for given reaction'

    reac = reac.split('.')
    cano_reac_list = [remove_am(x, True) for x in reac]

    am2belong, idx_remapper = {}, {}
    for idx, rc in enumerate(reac):
        mol = Chem.MolFromSmiles(rc)
        non_zero_am = [x.GetAtomMapNum() for x in mol.GetAtoms()]
        non_zero_am = [x for x in non_zero_am if x != 0]
        am2belong.update({x: idx for x in non_zero_am})
        idx_remapper[idx] = ({x: tdx for tdx, x in enumerate(non_zero_am)})

    prod_graph, prod_am2idx = trans_fn(prod)
    num_bond_feat = prod_graph['edge_feat'].shape[1]
    prod_idx2am = {v: k for k, v in prod_am2idx.items()}

    splited_edges = [[] for _ in range(len(reac))]
    splited_ettr = [[] for _ in range(len(reac))]

    for i in range(prod_graph['edge_index'].shape[1]):
        start = int(prod_graph['edge_index'][0][i])
        end = int(prod_graph['edge_index'][1][i])
        start_am = prod_idx2am[start]
        end_am = prod_idx2am[end]
        start_belong = am2belong[start_am]
        end_belong = am2belong[end_am]

        if start_belong == end_belong:
            splited_edges[start_belong].append((
                idx_remapper[start_belong][start_am],
                idx_remapper[start_belong][end_am]
            ))

            splited_ettr[start_belong].append(prod_graph['edge_feat'][i])

        elif add_pesudo_node:
            add_reidx(idx_remapper[start_belong], start_am)
            add_reidx(idx_remapper[start_belong], end_am)

            add_reidx(idx_remapper[end_belong], start_am)
            add_reidx(idx_remapper[end_belong], end_am)

            splited_edges[start_belong].append((
                idx_remapper[start_belong][start_am],
                idx_remapper[start_belong][end_am]
            ))

            splited_edges[end_belong].append((
                idx_remapper[end_belong][start_am],
                idx_remapper[end_belong][end_am]
            ))

            splited_ettr[start_belong].append(prod_graph['edge_feat'][i])
            splited_ettr[end_belong].append(prod_graph['edge_feat'][i])

    final_graphs = []
    print(mapped_rxn)
    print(idx_remapper)

    for i, p in enumerate(splited_edges):
        node_feat = [0] * len(idx_remapper[i])
        pesudo_mask = [0] * len(idx_remapper[i])

        for k, v in idx_remapper[i].items():
            node_feat[v] = prod_graph['node_feat'][prod_am2idx[k]]
            pesudo_mask[v] = am2belong[k] == i

        final_graphs.append({
            'node_feat': np.stack(node_feat, axis=0),
            'edge_feat': (
                np.stack(splited_ettr[i], axis=0)
                if len(splited_ettr[i]) > 0 else
                np.empty((0, num_bond_feat), dtype=np.int64)
            ),
            'edge_index': (
                np.array(splited_edges[i], dtype=np.int64).T
                if len(splited_edges[i]) > 0 else
                np.empty((2, 0), dtype=np.int64)
            ),
            'num_nodes': len(idx_remapper[i]),
            'pesudo_mask': np.array(pesudo_mask, dtype=bool)
        })

    return cano_reac_list, final_graphs
