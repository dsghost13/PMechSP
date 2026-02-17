from rdkit import Chem

def mapped_molecules_equivalent(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1, sanitize=False)
    mol2 = Chem.MolFromSmiles(smiles2, sanitize=False)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES input")

    if not mol1.HasSubstructMatch(mol2):
        return False

    labels_to_check = {10, 20}

    matches = mol1.GetSubstructMatches(mol2, uniquify=False)
    for match in matches:
        equivalent = True

        for i2, i1 in enumerate(match):
            a1 = mol1.GetAtomWithIdx(i1).GetAtomMapNum()
            a2 = mol2.GetAtomWithIdx(i2).GetAtomMapNum()

            if a1 in labels_to_check or a2 in labels_to_check:
                if a1 != a2:
                    equivalent = False
                    break

        if equivalent:
            return True
    return False
#
# s1 = "CC(=O)O[C:20](=[O:21])C.[C-:10]#N"
# s2 = "CC(=O)O[C:20](C)=[O:21].N#[C-:10]"
#
# print(mapped_molecules_equivalent(s1, s2))
#
# def get_lone_pairs(atom):
#     num_valence = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
#     num_bonding = atom.GetFormalCharge() + atom.GetTotalValence()
#     num_lone_pairs = (num_valence - num_bonding) // 2
#     return max(num_lone_pairs, 0)
#
#
# def get_default_valence(atom):
#     valence_list = list(Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum()))
#     num_bonds = atom.GetValence(Chem.ValenceType.EXPLICIT) + atom.GetValence(Chem.ValenceType.IMPLICIT)
#     formal_charge = atom.GetFormalCharge()
#
#     for valence in valence_list:
#         if num_bonds == (valence - formal_charge):
#             return valence
#     return valence_list[0]
#
# def get_valence_full(atom):
#     total_valence = get_lone_pairs(atom) + atom.GetValence(Chem.ValenceType.EXPLICIT) + atom.GetValence(Chem.ValenceType.IMPLICIT)
#     default_valence = get_default_valence(atom)
#     if not total_valence >= default_valence:
#         pass
#     return total_valence >= default_valence
#
# smiles = "C[Al]C(C)C"
# mol = Chem.MolFromSmiles(smiles)
# mol = Chem.AddHs(mol)
# print(Chem.MolToSmiles(mol))
# for atom in mol.GetAtoms():
#     # if atom.GetAtomicNum() != 3:
#     #     break
#     total_valence = atom.GetValence(Chem.ValenceType.EXPLICIT) + atom.GetValence(Chem.ValenceType.IMPLICIT)
#     print("Formal Charge: ", atom.GetFormalCharge())
#     print("Number of Bonds: ", total_valence)
#     print("Valence Full: ", get_valence_full(atom))
#     print("Valence List: ", list(Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())))