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
            mlist.append(canonical_smiles(x))

    if len(pos_charge) > 0 and len(neg_charge) > 0:
        if len(set(pos_charge)) != 1 and len(set(neg_charge)) != 1:
            return False, []
        elif sum(t[1] for t in pos_charge) +\
                sum(t[1] for t in neg_charge) != 0:
            return False, []
        else:
            pos_charge.sort(key=lambda t: t[1])
            neg_charge.sort(key=lambda t: -t[1])
            point1, point2, cmol, csum = 0, 0, [], 0
            while point1 < len(pos_charge) or point2 < len(neg_charge):
                if csum <= 0:
                    cmol.append(pos_charge[point1][0])
                    csum += pos_charge[point1][1]
                    point1 += 1
                else:
                    cmol.append(neg_charge[point2][0])
                    csum += neg_charge[point2][1]
                    point2 += 1
                if csum == 0:
                    mlist.append(canonical_smiles('.'.join(cmol)))
                    cmol = []
            assert csum == 0 and cmol == [], "Wrong Matching"
    else:
        mlist.extend([canonical_smiles(t[0]) for t in pos_charge])
        mlist.extend([canonical_smiles(t[0]) for t in neg_charge])

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


def getx_belong(dt, x):
    answer, rx = None, None
    for k, v in dt.get(x, {}).items():
        if v != 0 and (answer is None or v > rx):
            answer, rx = k, v
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pandas.read_csv(args.input_file)
    df = df.fillna('')
    out_data, unsplit_rows, nomain_rows = [], [], []

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
                    'old': {
                        'reac_list': reac_list,
                        'prod_list': prod_list,
                        'canonical_rxn': row['canonical_rxn'],
                        'catalyst': row['catalyst1'],
                        'reagent1': row['reagent1'],
                        'solvent1': row['solvent1'],
                        'reagent2': row['reagent2'],
                        'solvent2': row['solvent2'],
                    },
                    'new': {
                        'canonical_rxn': new_rxn,
                        'reac_list': new_reac,
                        'prod_list': new_prod,
                        'shared_list': shared_list,
                    },
                    'source': row['source'],
                    'dataset': row['dataset'],
                    'index': i
                }

                update_mol(row['catalyst1'], substance_to_cat, 'catalyst')
                update_mol(row['reagent1'], substance_to_cat, 'reagent')
                update_mol(row['reagent2'], substance_to_cat, 'reagent')
                update_mol(row['solvent1'], substance_to_cat, 'solvent')
                update_mol(row['solvent2'], substance_to_cat, 'solvent')
                out_data.append(data)

        else:
            unsplit_rows.append(row)

    print('[INFO] unsplited num:', len(unsplit_rows))
    if len(unsplit_rows) > 0:
        unsplit_df = pandas.DataFrame(unsplit_rows)
        unsplit_path = os.path.join(args.output_dir, 'unsplit.csv')
        unsplit_df.to_csv(unsplit_path, index=False)

    print('[INFO] nomain_rows:', len(nomain_rows))
    if len(nomain_rows) > 0:
        nomain_df = pandas.DataFrame(nomain_rows)
        nomain_path = os.path.join(args.output_dir, 'no_reaction.csv')
        nomain_df.to_csv(nomain_path, index=False)

    real_out, nobel = [], []
    for line in tqdm(out_data):
        bingo = False
        catalyst = None if line['old']['catalyst'] == '' \
            else canonical_smiles(line['old']['catalyst'])

        reagent1 = None if line['old']['reagent1'] == ''\
            else canonical_smiles(line['old']['reagent1'])
        reagent2 = None if line['old']['reagent2'] == ''\
            else canonical_smiles(line['old']['reagent2'])

        solvent1 = None if line['old']['solvent1'] == ''\
            else canonical_smiles(line['old']['solvent1'])
        solvent2 = None if line['old']['solvent2'] == ''\
            else canonical_smiles(line['old']['solvent2'])

        for x in line['new']['shared_list']:
            tans = getx_belong(substance_to_cat, x)
            if tans == 'catalyst':
                if x == catalyst or catalyst is None:
                    catalyst = x
                else:
                    bingo = True
                    break
            elif tans == 'reagent':
                if x == reagent1 or reagent1 is None:
                    reagent1 = x
                elif x == reagent2 or reagent2 is None:
                    reagent2 = x
                else:
                    bingo = True
                    break
            elif tans == 'solvent':
                if x == solvent1 or solvent1 is None:
                    solvent1 = x
                elif solvent2 == x or solvent2 is None:
                    solvent2 = x
                else:
                    bingo = True
                    break
            else:
                bingo = True
                break

        if bingo:
            nobel.append(line['index'])
        else:
            line['new'].update({
                'catalyst': '' if catalyst is None else catalyst,
                'reagent1': '' if reagent1 is None else reagent1,
                'reagent2': '' if reagent2 is None else reagent2,
                'solvent1': '' if solvent1 is None else solvent1,
                'solvent2': '' if solvent2 is None else solvent2
            })
            real_out.append(line)

    print('[INFO] no belong:', len(nobel))
    if len(nobel) > 0:
        x_source = df.iloc[nobel]
        nobel_path = os.path.join(args.output_dir, 'nobelong.csv')
        x_source.to_csv(nobel_path, index=False)
