import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from scripts.data_formatting import SmilesDataset
from scripts.downstream import split_batch_by_molecule, get_prediction_smiles
#from scripts.nn_models import <MODEL>

#---------------------------------------------------------------------------------------------------------------

# raw dataset
df = pd.read_csv('../datasets/<dataset>.csv')
smiles_list = df['SMILES Labelled'].tolist()

#---------------------------------------------------------------------------------------------------------------

# 70/15/15 train/test/val split
train_smiles, test_smiles = train_test_split(smiles_list, test_size=0.15, random_state=42)
train_smiles, val_smiles = train_test_split(train_smiles, test_size=0.1765, random_state=42)

# dataset objects
train_dataset = SmilesDataset(train_smiles)
test_dataset  = SmilesDataset(test_smiles)
val_dataset   = SmilesDataset(val_smiles)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

#---------------------------------------------------------------------------------------------------------------

# class weights based on train split
labels = []
for batch in train_loader:
    labels.append(batch.y.argmax(dim=1))
labels = torch.cat(labels)
class_counts = torch.bincount(labels)
class_weights = labels.size(0) / (len(class_counts) * class_counts.float())

# model instance
#model = <MODEL>(input_dim=9, hidden_dim=128, output_dim=5, edge_dim=8, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
criterion = torch.nn.CrossEntropyLoss()

#---------------------------------------------------------------------------------------------------------------

counter = 0
patience = 15
best_val_loss = float('inf')
num_epochs = 1000

for epoch in range(num_epochs):
    # training
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(out, batch.y.argmax(dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y.argmax(dim=1))
            val_loss += loss.item()

            preds, true = split_batch_by_molecule(out, batch)
            for p, t in zip(preds, true):
                if torch.equal(p, t):
                    correct += 1
                total += 1
    avg_val_loss = val_loss / len(val_loader)

    # validation accuracy
    val_acc = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # early stopping
    if avg_val_loss < best_val_loss:
        counter = 0
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"../models/<model>/<model>_{epoch+1}.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"Training stopped at epoch {epoch+1}.")
            break

#---------------------------------------------------------------------------------------------------------------

MODEL_PATH = '../models/<model>/<model>_<#>.pt'
OUTPUT_CSV = '../results/<model>_test_eval.csv'

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
        for p, t in zip(preds, true):
            if torch.equal(p[t != 2], t[t != 2]):
                correct += 1
            total += 1

        smiles_original.extend(batch.smiles)
        smiles_predicted.extend(get_prediction_smiles(preds, batch.smiles))

avg_test_loss = test_loss / len(test_loader)
test_acc = correct / total

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# saves test predictions to .csv file
test_df = pd.DataFrame({
    'SMILES Original': smiles_original,
    'SMILES Predicted': smiles_predicted
    #'SMILES Matched': smiles_matched
})
test_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}.")