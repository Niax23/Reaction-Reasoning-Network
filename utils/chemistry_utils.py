from rdkit import Chem


def canonical_smiles(x):
    mol = Chem.MolFromSmiles(x)
    return x if mol is None else Chem.MolToSmiles(mol)
