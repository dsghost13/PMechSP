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
    # TODO: is_valence_full = int(get_valence_full(atom))

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
        # TODO: float(is_valence_full),
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
        # TODO: rotatable bonds
    ], dtype=torch.float)

def get_lone_pairs(atom):
    num_valence = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
    num_bonding = atom.GetFormalCharge() + atom.GetTotalValence()
    num_lone_pairs = (num_valence - num_bonding) // 2
    return max(num_lone_pairs, 0)

PERIODIC_TABLE_GROUPS = {
    1: 1,    # H
    2: 18,   # He
    3: 1,    # Li
    4: 2,    # Be
    5: 13,   # B
    6: 14,   # C
    7: 15,   # N
    8: 16,   # O
    9: 17,   # F
    10: 18,  # Ne
    11: 1,   # Na
    12: 2,   # Mg
    13: 13,  # Al
    14: 14,  # Si
    15: 15,  # P
    16: 16,  # S
    17: 17,  # Cl
    18: 18,  # Ar
    19: 1,   # K
    20: 2,   # Ca
    21: 3,   # Sc
    22: 4,   # Ti
    23: 5,   # V
    24: 6,   # Cr
    25: 7,   # Mn
    26: 8,   # Fe
    27: 9,   # Co
    28: 10,  # Ni
    29: 11,  # Cu
    30: 12,  # Zn
    31: 13,  # Ga
    32: 14,  # Ge
    33: 15,  # As
    34: 16,  # Se
    35: 17,  # Br
    36: 18,  # Kr
    37: 1,   # Rb
    38: 2,   # Sr
    39: 3,   # Y
    40: 4,   # Zr
    41: 5,   # Nb
    42: 6,   # Mo
    43: 7,   # Tc
    44: 8,   # Ru
    45: 9,   # Rh
    46: 10,  # Pd
    47: 11,  # Ag
    48: 12,  # Cd
    49: 13,  # In
    50: 14,  # Sn
    51: 15,  # Sb
    52: 16,  # Te
    53: 17,  # I
    54: 18,  # Xe
    55: 1,   # Cs
    56: 2,   # Ba
    57: 3,   # La
    58: 3,   # Ce
    59: 3,   # Pr
    60: 3,   # Nd
    61: 3,   # Pm
    62: 3,   # Sm
    63: 3,   # Eu
    64: 3,   # Gd
    65: 3,   # Tb
    66: 3,   # Dy
    67: 3,   # Ho
    68: 3,   # Er
    69: 3,   # Tm
    70: 3,   # Yb
    71: 3,   # Lu
    72: 4,   # Hf
    73: 5,   # Ta
    74: 6,   # W
    75: 7,   # Re
    76: 8,   # Os
    77: 9,   # Ir
    78: 10,  # Pt
    79: 11,  # Au
    80: 12,  # Hg
    81: 13,  # Tl
    82: 14,  # Pb
    83: 15,  # Bi
    84: 16,  # Po
    85: 17,  # At
    86: 18,  # Rn
    87: 1,   # Fr
    88: 2,   # Ra
    89: 3,   # Ac
    90: 3,   # Th
    91: 3,   # Pa
    92: 3,   # U
    93: 3,   # Np
    94: 3,   # Pu
    95: 3,   # Am
    96: 3,   # Cm
    97: 3,   # Bk
    98: 3,   # Cf
    99: 3,   # Es
    100: 3,  # Fm
    101: 3,  # Md
    102: 3,  # No
    103: 3,  # Lr
    104: 4,  # Rf
    105: 5,  # Db
    106: 6,  # Sg
    107: 7,  # Bh
    108: 8,  # Hs
    109: 9,  # Mt
    110: 10, # Ds
    111: 11, # Rg
    112: 12, # Cn
    113: 13, # Nh
    114: 14, # Fl
    115: 15, # Mc
    116: 16, # Lv
    117: 17, # Ts
    118: 18  # Og
}


PAULING_ELECTRONEGATIVITIES = {
    1: 2.20,   # H
    2: 0.00,   # He
    3: 0.98,   # Li
    4: 1.57,   # Be
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    10: 0.00,  # Ne
    11: 0.93,  # Na
    12: 1.31,  # Mg
    13: 1.61,  # Al
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    18: 0.00,  # Ar
    19: 0.82,  # K
    20: 1.00,  # Ca
    21: 1.36,  # Sc
    22: 1.54,  # Ti
    23: 1.63,  # V
    24: 1.66,  # Cr
    25: 1.55,  # Mn
    26: 1.83,  # Fe
    27: 1.88,  # Co
    28: 1.91,  # Ni
    29: 1.90,  # Cu
    30: 1.65,  # Zn
    31: 1.81,  # Ga
    32: 2.01,  # Ge
    33: 2.18,  # As
    34: 2.55,  # Se
    35: 2.96,  # Br
    36: 3.00,  # Kr
    37: 0.82,  # Rb
    38: 0.95,  # Sr
    39: 1.22,  # Y
    40: 1.33,  # Zr
    41: 1.60,  # Nb
    42: 2.16,  # Mo
    43: 1.90,  # Tc
    44: 2.20,  # Ru
    45: 2.28,  # Rh
    46: 2.20,  # Pd
    47: 1.93,  # Ag
    48: 1.69,  # Cd
    49: 1.78,  # In
    50: 1.96,  # Sn
    51: 2.05,  # Sb
    52: 2.10,  # Te
    53: 2.66,  # I
    54: 2.60,  # Xe
    55: 0.79,  # Cs
    56: 0.89,  # Ba
    57: 1.10,  # La
    58: 1.12,  # Ce
    59: 1.13,  # Pr
    60: 1.14,  # Nd
    61: 1.13,  # Pm
    62: 1.17,  # Sm
    63: 1.20,  # Eu
    64: 1.20,  # Gd
    65: 1.22,  # Tb
    66: 1.23,  # Dy
    67: 1.24,  # Ho
    68: 1.24,  # Er
    69: 1.25,  # Tm
    70: 1.10,  # Yb
    71: 1.27,  # Lu
    72: 1.30,  # Hf
    73: 1.50,  # Ta
    74: 2.36,  # W
    75: 1.90,  # Re
    76: 2.20,  # Os
    77: 2.20,  # Ir
    78: 2.28,  # Pt
    79: 2.54,  # Au
    80: 2.00,  # Hg
    81: 1.62,  # Tl
    82: 2.33,  # Pb
    83: 2.02,  # Bi
    84: 2.00,  # Po
    85: 2.20,  # At
    86: 0.00,  # Rn
    87: 0.70,  # Fr
    88: 0.90,  # Ra
    89: 1.10,  # Ac
    90: 1.30,  # Th
    91: 1.50,  # Pa
    92: 1.38,  # U
    93: 1.36,  # Np
    94: 1.28,  # Pu
    95: 1.13,  # Am
    96: 1.28,  # Cm
    97: 1.30,  # Bk
    98: 1.30,  # Cf
    99: 1.30,  # Es
    100: 1.30, # Fm
    101: 1.30, # Md
    102: 1.30, # No
    103: 1.30, # Lr
    104: 0.00, # Rf
    105: 0.00, # Db
    106: 0.00, # Sg
    107: 0.00, # Bh
    108: 0.00, # Hs
    109: 0.00, # Mt
    110: 0.00, # Ds
    111: 0.00, # Rg
    112: 0.00, # Cn
    113: 0.00, # Nh
    114: 0.00, # Fl
    115: 0.00, # Mc
    116: 0.00, # Lv
    117: 0.00, # Ts
    118: 0.00  # Og
}
