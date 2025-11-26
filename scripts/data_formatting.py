import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset

from scripts.feature_extraction import get_atom_features, get_bond_features

LABEL_MAP = {10: 1, 11: 2, 20: 3, 21: 4}

class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        super().__init__()

        self.graphs = []
        for smiles in smiles_list:
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except:
        return None

    # node features
    x = torch.stack([get_atom_features(atom) for atom in mol.GetAtoms()])

    # node labels
    labels = [LABEL_MAP.get(atom.GetAtomMapNum(), 0) for atom in mol.GetAtoms()]
    num_nodes = len(labels)

    y = torch.zeros((num_nodes, 5), dtype=torch.float)
    for i, label in enumerate(labels):
        y[i, label] = 1.0

    # undirected edges with edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        f = get_bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(f)
        edge_attr.append(f)

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 8), dtype=torch.float)  # 6 = bond feature dim
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)

    # graph object
    return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)