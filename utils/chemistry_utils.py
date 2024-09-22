from rdkit import Chem


def canonical_smiles(x):
    mol = Chem.MolFromSmiles(x)
    return x if mol is None else Chem.MolToSmiles(mol)


def remove_am(x, canonical=True):
    mol = Chem.MolFromSmiles(x)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol, canonical=canonical)


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

    am2belong = {}
    for idx, rc in enumerate(rc):
        mol = Chem.MolFromSmiles(rc)
        am2belong.update({
            x.GetAtomMapNum(): idx for x in mol.GetAtoms()
            if x.GetAtomMapNum() != 0
        })

    idx_remapper = {}
    # (atom_map, graph_idx): new_idx, a atom might be in
    # multi graphs as pesudo node when pesudo node added

    prod_graph, prod_am2idx = trans_fn(prod)
    prod_idx2am = {v: k for k, v in prod_am2idx.items()}

    splited_edges = [[] for _ in range(len(reac))]
    splited_ettr = [[] for _ in range(len(reac))]

    for i in range(prod_graph['edge_index'].shape[1]):
        start = int(prod_graph['edge_index'][0][i])
        end = int(prod_graph['edge_index'][1][i])

        start_belong = am2belong[start]
        end_belong = am2belong[end]
        start_am = prod_idx2am[start]
        end_am = prod_idx2am[end]

        if start_belong == end_belong:
            if start_belong not in idx_remapper:
                idx_remapper[start_belong] = {}

            add_reidx(idx_remapper[start_belong], start_am)
            add_reidx(idx_remapper[start_belong], end_am)

            splited_edges[start_belong].append((
                idx_remapper[start_belong][start_am],
                idx_remapper[start_belong][end_am]
            ))

            splited_ettr[start_belong].append(prod_graph['edge_feat'][i])

        elif add_pesudo_node:
            if start_belong not in idx_remapper:
                idx_remapper[start_belong] = {}
            if end_belong not in idx_remapper:
                idx_remapper[end_belong] = {}

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
    for i, p in enumerate(splited_edges):
        node_feat = [0] * len(idx_remapper[i])
        pesudo_mask = [0] * len(idx_remapper[i])

        for k, v in idx_remapper[i].items():
            node_feat[v] = prod_graph['node_feat'][prod_am2idx[k]]
            pesudo_mask[v] = am2belong[k] == i

        final_graphs.append({
            'node_feat': np.stack(node_feat, axis=0),
            'edge_feat': np.stack(splited_ettr[i], axis=0),
            'edge_index': np.array(splited_edges[i], dtype=np.int64).T,
            'num_nodes': len(idx_remapper[i]),
            'pesudo_mask': np.array(pesudo_mask, dtype=bool)
        })

    return cano_reac_list, final_graphs
