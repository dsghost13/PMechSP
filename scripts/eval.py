import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data_formatting import SmilesDataset
from downstream import get_prediction_smiles, split_batch_by_molecule
from nn_models import GINE
from test import mapped_molecules_equivalent

MODEL_PATH = '/data/homezvol0/petertl2/PMechSP/models/mayr/mayr_159.pt'
OUTPUT_CSV = '/data/homezvol0/petertl2/PMechSP/results/mayr_test_eval.csv'

# raw dataset
df = pd.read_csv('/data/homezvol0/petertl2/PMechSP/datasets/4k_Clayden.csv')
smiles_list = df['SMILES Labelled'].tolist()

test_dataset = SmilesDataset(smiles_list)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

## 70/15/15 train/test/val split
#train_smiles, test_smiles = train_test_split(smiles_list, test_size=0.15, random_state=42)
#train_smiles, val_smiles = train_test_split(train_smiles, test_size=0.1765, random_state=42)
#
## dataset objects
#train_dataset = SmilesDataset(train_smiles)
#test_dataset  = SmilesDataset(test_smiles)
#val_dataset   = SmilesDataset(val_smiles)
#
## dataloaders
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
#val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# model instance
model = GINE(input_dim=8, hidden_dim=128, output_dim=3, edge_dim=6, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# load best model
model.load_state_dict(torch.load(MODEL_PATH))

smiles_original = []
smiles_predicted = []
smiles_matched = []

# test evaluation
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(out, batch.y.argmax(dim=1))
        test_loss += loss.item()

        preds, true = split_batch_by_molecule(out, batch)

        smiles_original.extend(batch.smiles)
        smiles_predicted.extend(get_prediction_smiles(preds, batch.smiles))

        batch_smiles_true = batch.smiles
        batch_smiles_pred = get_prediction_smiles(preds, batch.smiles)

        for smi_p, smi_t in zip(batch_smiles_pred, batch_smiles_true):
            try:
                if mapped_molecules_equivalent(smi_p, smi_t):
                    smiles_matched.append(True)
                    correct += 1
                else:
                    smiles_matched.append(False)
            except:
                smiles_matched.append(False)
            total += 1

avg_test_loss = test_loss / len(test_loader)
test_acc = correct / total

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# saves test predictions to .csv file
test_df = pd.DataFrame({
    'SMILES Original': smiles_original,
    'SMILES Predicted': smiles_predicted,
    'SMILES Matched': smiles_matched,
})
test_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}.")