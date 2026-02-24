import torch
from rdkit import Chem
from torch_geometric.utils import subgraph

from data_formatting import LABEL_MAP

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
        preds = decode(mol_logits)
        mol_preds.append(preds)
        mol_true.append(true[node_idx])

    return mol_preds, mol_true

def decode(logits):
    num_nodes = logits.size(0)
    preds = torch.zeros(num_nodes, dtype=torch.long)
    prob = torch.softmax(logits, dim=1)

    # label scores
    scores_10 = prob[:, 1].argsort(descending=True)
    scores_20 = prob[:, 2].argsort(descending=True)

    # best scores for 10 and 20
    best_idx_10 = scores_10[0].item()
    best_idx_20 = scores_20[0].item()

    # determine if 10 or 20 should be labelled first
    if prob[best_idx_10, 1] >= prob[best_idx_20,2]:
        first_class, first_scores = 1, scores_10
        second_class, second_scores = 2, scores_20
    else:
        first_class, first_scores = 2, scores_20
        second_class, second_scores = 1, scores_10

    # label first atom
    first_idx = first_scores[0].item()
    preds[first_idx] = first_class

    # label second atom with collision avoidance
    for idx in second_scores.tolist():
        if preds[idx] == 0:
            preds[idx] = second_class
            break

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