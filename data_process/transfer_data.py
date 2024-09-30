import json
import torch
import argparse
from rdkit import Chem

def canonical_smiles(x):
	return Chem.MolToSmiles(Chem.MolFromSmiles(x))


def cano_rxn(rxn):
    reac, prod = rxn.split('>>')
    return f'{canonical_smiles(reac)}>>{canonical_smiles(prod)}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument(
        '--data_type', required=True, help='the type for data to convert',
        choices=['mole_feature', 'json', 'rxn_feature'],
    )
    parser.add_argument(
        '--dataset',  choices=['uspto_condition', 'uspto_500mt'],
        help='the dataset to process', required=True,
    )
    args = parser.parse_args()
    if args.data_type == 'mole_feature':
        ft = torch.load(args.file)
        ft['smiles2idx'] = {
            canonical_smiles(k): v for k, v in ft['smiles2idx'].items()
        }
        torch.save(ft, args.file)
    elif args.data_type == 'rxn_feature':
        ft = troch.load(args.file)
        ft['smiles2idx'] = {
            (cano_rxn(a), canonical_smiles(b)): v
            for (a, b), v in ft['smiles2idx'].items()
        }
        torch.save(ft, args.file)
    else:
        if args.dataset == 'uspto_condition':
            with open(args.file) as Fin:
                info = json.load(Fin)
            for line in info:
                line['new']['reac_list'] = \
                    [canonical_smiles(x) for x in line['new']['reac_list']]
                line['new']['prod_list'] = \
                    [canonical_smiles(x) for x in line['new']['prod_list']]

            with open(args.file, 'w') as Fout:
                json.dump(info, Fout, indent=4)

        else:
            raise NotImplementedError()
