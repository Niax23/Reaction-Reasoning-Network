import os
import sys
this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(this_folder))


from utils.tokenlizer import smi_tokenizer
from rdkit import Chem
import json
import argparse



metal = {
    'Ag', 'Au', 'Al', 'Ca', 'Co', 'Cr', 'Fe', 'Cu',
    'Hg', 'Ir', 'In', 'K', 'Li', 'Mg', 'Mn', 'Na',
    'Ni', 'Pd', 'Pt', 'Rh', 'Ru', 'Sn', 'Ta', 'Ti',
    'Tl', 'Ce', 'Be', 'V', 'W', 'Zn', 'Zr', 'Sc',
    'Ba', 'Nd', 'Yb', "La", 'Cs', 'Re', 'Pb', 'Dy'
}

halogens = {'F', 'Cl', 'Br', 'I', 'At'}


def is_org(x):
    mol = Chem.MolFromSmiles(x)
    for at in mol.GetAtoms():
        if at.GetSymbol() == 'C':
            if at.GetTotalNumHs() > 0:
                return True
            for nb in at.GetNeighbors():
                if nb.GetSymbol() in halogens:
                    return True
    return False


def if_free_metal(x):
    mol = Chem.MolFromSmiles(x)
    for idx, p in enumerate(mol.GetAtoms()):
        if idx == 1:
            return False
        if p.GetFormalCharge() != 0:
            return False
        if p.GetSymbol() not in metal:
            return False
    return True


def is_Metal_Halides(x):
    mol, has_mat, has_hal = Chem.MolFromSmiles(x), False, False
    for p in mol.GetAtoms():
        sym = p.GetSymbol()
        if sym in halogens:
            has_hal = True
        elif sym in metal:
            has_mat = True
        else:
            return False
    return has_hal and has_mat


def contain_metal_halides(x):
    ha_num = 0
    for px in x.split('.'):
        mol = Chem.MolFromSmiles(px)
        ats = list(mol.GetAtoms())
        if len(ats) == 1 and ats[0].GetSymbol() in halogens:
            ha_num += 1

    for px in x.split('.'):
        mol = Chem.MolFromSmiles(px)
        ats = list(mol.GetAtoms())
        if len(ats) == 1 and ats[0].GetSymbol() in metal:
            if ats[0].GetFormalCharge() <= ha_num:
                return True
    return False


def is_catalyst(x):
    for y in x.split('.'):
        if if_free_metal(y) or is_Metal_Halides(y):
            return True
        mol = Chem.MolFromSmiles(y)
        for at in mol.GetAtoms():
            if not at.IsInRing():
                continue
            sym = at.GetSymbol()
            if sym == 'P' or sym in metal:
                return True
    return contain_metal_halides(x)


def get_rank(x):
    token_len = len(smi_tokenizer(x))
    return (0, token_len) if is_catalyst(x) else (
        (1, token_len) if is_org(x) else (2, token_len)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    all_px = set()

    for model in ['train.json', 'val.json', 'test.json']:
        with open(os.path.join(DATA_DIR, model)) as Fin:
            INFO = json.load(Fin)

        for x in INFO:
            all_px.update(x['reagent_list'])

    sot_px = [(x, get_rank(x)) for x in all_px]
    sot_px.sort(key=lambda x: x[1])

    with open(os.path.join(DATA_DIR, 'all_reagents.json'), 'w') as F:
        json.dump([x[0] for x in sot_px], F, indent=4)

    all_tokens = set()
    for x in all_px:
        all_tokens.update(smi_tokenizer(x))

    with open(os.path.join(DATA_DIR, 'all_tokens.json'), 'w') as F:
        json.dump(list(all_tokens), F, indent=4)
