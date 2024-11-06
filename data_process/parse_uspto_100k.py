import os
from rdkit import Chem
import pandas
import json
from tqdm import tqdm

import argparse


def split_reac_reag(mapped_rxn):
    reac, prod = mapped_rxn.split('>>')
    prod_mol = Chem.MolFromSmiles(prod)
    prod_am = {x.GetAtomMapNum() for x in prod_mol.GetAtoms()}
    reax, reag = [], []
    for x in reac.split('.'):
        re_mol = Chem.MolFromSmiles(x)
        re_am = {x.GetAtomMapNum() for x in re_mol.GetAtoms()}
        if len(re_am & prod_am) > 0:
            reax.append(x)
        else:
            reag.append(x)
    return reax, reag


def clear_map_number(smi):
    """Clear the atom mapping number of a SMILES sequence"""
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return canonical_smiles(Chem.MolToSmiles(mol))


def canonical_smiles(smi):
    """Canonicalize a SMILES without atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    return smi if mol is None else Chem.MolToSmiles(mol)


def resplit_reag(reac, reag, rxn_with_frag):
    reac_frag, prod = rxn_with_frag.split('>>')
    cntz, um2m = {}, {}
    for x in reag:
        key = clear_map_number(x)
        cntz[key] = cntz.get(key, 0) + 1
    for x in reac:
        key = clear_map_number(x)
        if key not in um2m:
            um2m[key] = []
        um2m[key].append(x)

    reapx, reacx, mreacx = [], [], []
    for x in reac_frag.split('.'):
        pz, ok, cnty = x.split('~'), True, {}
        for y in pz:
            key = clear_map_number(y)
            cnty[key] = cnty.get(key, 0) + 1

        for k, v in cnty.items():
            if cntz.get(k, 0) < v:
                ok = False
                break

        if ok:
            for k, v in cnty.items():
                cntz[k] -= v
            reapx.append(canonical_smiles(x.replace('~', '.')))
        else:
            reacx.append(canonical_smiles(x.replace('~', '.')))
            this_line = []
            for t in pz:
                key = clear_map_number(t)
                if len(um2m.get(key, [])) > 0:
                    this_line.append(um2m[key].pop())
                else:
                    this_line.append(key)
            mreacx.append('.'.join(this_line))

    return mreacx, reacx, reapx


def check(reac, reag, oldx):
    if len(reag) > 0:
        newx = clear_map_number(f'{reac}.{reag}')
    else:
        newx = clear_map_number(reac)
    return newx == clear_map_number(oldx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    out_infos = []
    raw_info = pandas.read_csv(args.data_dir, sep='\t')
    raw_info = raw_info.to_dict('records')
    for idx, ele in enumerate(tqdm(raw_info)):
        mapped_rxn = ele['mapped_rxn']
        old_reac, prod = mapped_rxn.split('>>')
        rxn_with_frag = ele['canonical_rxn_with_fragment_info']
        reac, reag = split_reac_reag(mapped_rxn)
        new_map_reac, new_reac, new_reag = \
            resplit_reag(reac, reag, rxn_with_frag)

        if not check('.'.join(new_map_reac), '.'.join(new_reag), old_reac):
            print('map_rxn', mapped_rxn)
            print('new_reac', new_reac)
            print('new_mapped_reac', new_map_reac)
            print('reag_list', new_reag)
            print('prod', prod)
            exit()

        clear_rxn = canonical_smiles('.'.join(new_reac)) + '>>' + \
            canonical_smiles(ele['products'])

        tline = {
            'new_mapped_rxn': f'{".".join(new_map_reac)}>>{prod}',
            'reac_list': new_reac,
            'reagent_list': new_reag,
            'mapped_reac_list': new_map_reac,
            'products': canonical_smiles(ele['products']),
            'clear_cano_rxn': clear_rxn
        }
        tline.update(ele)
        out_infos.append(tline)
    with open(args.output_dir, 'w') as Fout:
        json.dump(out_infos, Fout, indent=4)