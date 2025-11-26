import torch
from rdkit import Chem
from torch_geometric.utils import subgraph

from scripts.data_formatting import LABEL_MAP

ATOM_MAP = {value: key for key, value in LABEL_MAP.items()}

def split_batch_by_molecule(logits, batch):
    mol_preds = []
    mol_true = []

    # atom-level truth labels
    true = batch.y.argmax(dim=1)

    for i in range(batch.num_graphs):
        node_idx = (batch.batch == i).nonzero(as_tuple=True)[0]

        mol_logits = logits[node_idx]
        mol_x = batch.x[node_idx]
        mol_edge_index, _ = subgraph(
            node_idx,
            batch.edge_index,
            relabel_nodes=True,
            num_nodes=batch.x.size(0)
        )

        # molecule-level predictions and truth labels
        mol_preds.append(decode(mol_logits, mol_x, mol_edge_index))
        mol_true.append(true[node_idx])

    return mol_preds, mol_true

def decode(logits, x, edge_index):
    num_nodes = logits.size(0)
    preds = torch.zeros(num_nodes, dtype=torch.long)
    prob = torch.softmax(logits, dim=1)

    # label scores
    scores_10 = prob[:, 1].argsort(descending=True)
    scores_20 = prob[:, 3].argsort(descending=True)
    scores_21 = prob[:, 4]
    scores_11 = prob[:, 2]
    scores_0 = prob[:, 0]

    # best scores for 10 and 20
    best_idx_10 = scores_10[0].item()
    best_idx_20 = scores_20[0].item()

    # determine if 10 or 20 should be labelled first
    if prob[best_idx_10, 1] >= prob[best_idx_20, 3]:
        first_class, first_scores = 1, scores_10
        second_class, second_scores = 3, scores_20
    else:
        first_class, first_scores = 3, scores_20
        second_class, second_scores = 1, scores_10

    # label first atom
    first_idx = first_scores[0].item()
    preds[first_idx] = first_class

    # label second atom with collision avoidance
    second_idx = None
    for idx in second_scores.tolist():
        if preds[idx] == 0:
            preds[idx] = second_class
            second_idx = idx
            break

    # find index of 20
    idx_10, idx_20 = None, None
    if first_class == 1:
        idx_10, idx_20 = first_idx, second_idx
    elif second_class == 1:
        idx_10, idx_20 = second_idx, first_idx

    # label 21 if 20 already has full valence and below group 2
    if (idx_20 is not None) and (x[idx_20, 5] == 1.0):
        neighbors = edge_index[1][edge_index[0] == idx_20].tolist()

        best_idx_21 = None
        best_score = float('-inf')

        for n in neighbors:
            if preds[n] == 0:
                if not x[idx_20, 1] <= 2:
                    if scores_21[n] <= scores_0[n]:
                        continue

                if scores_21[n] > best_score:
                    best_score = scores_21[n]
                    best_idx_21 = n

        if best_idx_21 is not None:
            preds[best_idx_21] = 4

    # label 11 if 10 has no lone pairs
    if (idx_10 is not None) and (x[idx_10, 4] == 0.0):
        neighbors = edge_index[1][edge_index[0] == idx_10].tolist()

        best_idx_11 = None
        best_score = float('-inf')
        for n in neighbors:
            if (preds[n] == 0) and (scores_11[n] > best_score):
                best_score = scores_11[n]
                best_idx_11 = n

        if best_idx_11 is not None:
            preds[best_idx_11] = 2

    return preds

def get_prediction_smiles(labels, smiles):
    predicted_smiles = []

    for labels, smi in zip(labels, smiles):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            predicted_smiles.append("")
            continue

        for atom, label in zip(mol.GetAtoms(), labels.tolist()):
            atom.SetAtomMapNum(ATOM_MAP.get(label, 0))

        predicted_smiles.append(Chem.MolToSmiles(mol))

    return predicted_smiles