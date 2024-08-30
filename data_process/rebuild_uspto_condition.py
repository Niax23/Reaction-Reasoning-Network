import rdkit
from rdkit import Chem
import argparse


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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', required=True)
	parser.add_argument('--output_file', required=True)
	args = parser.parse_args()