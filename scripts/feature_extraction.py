import torch
from rdkit import Chem

def get_atom_features(atom):
    atom.UpdatePropertyCache(strict=False)

    # periodic table values
    atomic_num = atom.GetAtomicNum()
    period = Chem.GetPeriodicTable().GetRow(atomic_num)
    group = PERIODIC_TABLE_GROUPS.get(atomic_num)
    electronegativity = PAULING_ELECTRONEGATIVITIES.get(atomic_num)

    # electronic properties
    lone_pairs = get_lone_pairs(atom)
    is_valence_full = int(get_valence_full(atom))

    # hybridization
    hybridization = atom.GetHybridization()
    is_sp = int(hybridization == Chem.rdchem.HybridizationType.SP)
    is_sp2 = int(hybridization == Chem.rdchem.HybridizationType.SP2)
    is_sp3 = int(hybridization == Chem.rdchem.HybridizationType.SP3)

    return torch.tensor([
        float(atomic_num),
        float(period),
        float(group),
        float(electronegativity),
        float(lone_pairs),
        float(is_valence_full),
        float(is_sp),
        float(is_sp2),
        float(is_sp3)
    ], dtype=torch.float)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ),
        float(bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE),
    ], dtype=torch.float)

def get_lone_pairs(atom):
    num_valence = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
    num_bonding = atom.GetFormalCharge() + atom.GetTotalValence()
    num_lone_pairs = (num_valence - num_bonding) // 2
    return max(num_lone_pairs, 0)

def get_valence_full(atom):
    explicit_valence = atom.GetValence(Chem.ValenceType.EXPLICIT)
    normal_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
    return explicit_valence >= normal_valence

PERIODIC_TABLE_GROUPS = {
    1: 1,    # H
    3: 1,    # Li
    5: 13,   # B
    6: 14,   # C
    7: 15,   # N
    8: 16,   # O
    9: 17,   # F
    11: 1,   # Na
    12: 2,   # Mg
    13: 13,  # Al
    14: 14,  # Si
    15: 15,  # P
    16: 16,  # S
    17: 17,  # Cl
    19: 1,   # K
    20: 2,   # Ca
    21: 3,   # Sc
    22: 4,   # Ti
    24: 6,   # Cr
    25: 7,   # Mn
    26: 8,   # Fe
    27: 9,   # Co
    28: 10,  # Ni
    29: 11,  # Cu
    30: 12,  # Zn
    33: 15,  # As
    35: 17,  # Br
    40: 4,   # Zr
    44: 8,   # Ru
    45: 9,   # Rh
    46: 10,  # Pd
    47: 11,  # Ag
    49: 13,  # In
    50: 14,  # Sn
    53: 17,  # I
    76: 8,   # Os
    77: 9,   # Ir
    78: 10,  # Pt
    79: 11   # Au
}

PAULING_ELECTRONEGATIVITIES = {
    1: 2.20,   # H
    3: 0.98,   # Li
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    11: 0.93,  # Na
    12: 1.31,  # Mg
    13: 1.61,  # Al
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    19: 0.82,  # K
    20: 1.00,  # Ca
    21: 1.36,  # Sc
    22: 1.54,  # Ti
    24: 1.66,  # Cr
    25: 1.55,  # Mn
    26: 1.83,  # Fe
    27: 1.88,  # Co
    28: 1.91,  # Ni
    29: 1.90,  # Cu
    30: 1.65,  # Zn
    33: 2.18,  # As
    35: 2.96,  # Br
    40: 1.33,  # Zr
    44: 2.20,  # Ru
    45: 2.28,  # Rh
    46: 2.20,  # Pd
    47: 1.93,  # Ag
    49: 1.78,  # In
    50: 1.96,  # Sn
    53: 2.66,  # I
    76: 2.20,  # Os
    77: 2.20,  # Ir
    78: 2.28,  # Pt
    79: 2.54   # Au
}