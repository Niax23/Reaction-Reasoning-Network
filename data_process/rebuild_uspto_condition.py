import pandas
import os
import rdkit
from rdkit import Chem
import argparse

from tqdm import tqdm


def canonical_smiles(x):
    mol = Chem.MolFromSmiles(x)
    return Chem.MolToSmiles(mol)


def resplit(moles):
    pos_charge, neg_charge, mlist = [], [], []
    for x in moles.split('.'):
        mol = Chem.MolFromSmiles(x)
        total_charge = sum(t.GetFormalCharge() for t in mol.GetAtoms())
        if total_charge > 0:
            pos_charge.append((x, total_charge))
        elif total_charge < 0:
            neg_charge.append((x, total_charge))
        else:
            mlist.append(x)

    if len(pos_charge) > 0 and len(neg_charge) > 0:
        if len(set(pos_charge)) != 1 or len(set(neg_charge)) != 1:
            return False, []
        elif sum(t[1] for t in pos_charge) +\
                sum(t[1] for t in neg_charge) != 0:
            mlist.extend([t[0] for t in pos_charge])
            mlist.extend([t[0] for t in neg_charge])
        else:
            this_mol = [t[0] for t in pos_charge] + \
                [t[0] for t in neg_charge]
            mlist.append('.'.join(this_mol))
    else:
        mlist.extend([t[0] for t in pos_charge])
        mlist.extend([t[0] for t in neg_charge])

    return True, mlist


def get_main_product(product_list):
    tnum, mx = None, None
    for x in product_list:
        mol = Chem.MolFromSmiles(x)
        if tnum is None or len(mol.GetAtoms()) > tnum:
            tnum, mx = len(mol.GetAtoms()), x
    return mx


def split_equal(reac_list, prod_list):
    cano_reac_list = [canonical_smiles(x) for x in reac_list]
    cano_prod_list = [canonical_smiles(x) for x in prod_list]

    reac_cnter, prod_cnter, shared_list = {}, {}, []

    for x in cano_reac_list:
        reac_cnter[x] = reac_cnter.get(x, 0) + 1

    for x in cano_prod_list:
        prod_cnter[x] = prod_cnter.get(x, 0) + 1

    for k, v in reac_cnter.items():
        if k not in prod_cnter:
            continue
        share_cnt = min(v, prod_cnter[k])
        reac_cnter[k] -= share_cnt
        prod_cnter[k] -= share_cnt
        shared_list.extend([k] * share_cnt)

    new_reac_list = [k for k, v in reac_cnter.items() if v > 0]
    new_prod_list = [k for k, v in prod_cnter.items() if v > 0]

    return new_reac_list, new_prod_list, shared_list


def recheck(reac_list, prod_list, shared_list, rxn):
    reac, prod = rxn.split('>>')
    if canonical_smiles('.'.join(reac_list + shared_list)) != \
            canonical_smiles(reac):
        return False
    if canonical_smiles('.'.join(prod_list + shared_list)) != \
            canonical_smiles(prod_list):
        return False
    return True


def update_mol(x, dt, ix):
    if x == '':
        return
    if x not in dt:
        dt[x] = {'catalyst': 0, 'reagent': 0, 'solvent': 0}
    dt[x][ix] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pandas.read_csv(args.input_file)
    df = df.fillna('')
    out_data, unsplit_rows, nomain_rows, nobel = {}, [], [], []

    substance_to_cat = {}

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        reac, prod = row['canonical_rxn'].split('>>')
        reac_success, reac_list = resplit(reac)
        prod_success, prod_list = resplit(prod)

        if reac_success and prod_success:
            main_product = get_main_product(prod_list)
            new_reac, new_prod, shared_list = split_equal(reac_list, prod_list)
            if main_product in shared_list:
                nomain_rows.append(row)
            else:
                new_rxn = f'{".".join(new_reac)}>>{".".join(new_prod)}'
                data = {
                    'reac_list': new_reac,
                    'prod_list': new_prod,
                    'shared_list': shared_list,
                    'full_reac_list': reac_list,
                    'full_prod_list': prod_list,
                    'old_canonical_rxn': row['canonical_rxn'],
                    'canonical_rxn': new_rxn,
                    'catalyst': row['catalyst'],
                    'reagent1': row['reagent1'],
                    'solvent1': row['solvent1'],
                    'reagent2': row['reagent2'],
                    'solvent2': row['solvent2'],
                }

                update_mol(row['catalyst'], substance_to_cat, 'catalyst')
                update_mol(row['reagent1'], substance_to_cat, 'reagent')
                update_mol(row['reagent2'], substance_to_cat, 'reagent')
                update_mol(row['solvent1'], substance_to_cat, 'solvent')
                update_mol(row['solvent2'], substance_to_cat, 'solvent')

        else:
            unsplit_rows.append(row)

    if len(unsplit_rows) > 0:
        unsplit_df = pandas.DataFrame(unsplit_rows)
        unsplit_path = os.path.join(args.output_dir, 'unsplit.csv')
        unsplit_df.to_csv(unsplit_path, index=False)
