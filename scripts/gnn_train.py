# File: scripts/gnn_train.py
"""
Train a simple GCN using:
 - ../outputs/embeddings.npy
 - ../outputs/adjacency.npy  (if not present, builds from edges.csv)
 - ../outputs/nodes.csv
Outputs:
 - ../outputs/gnn_model.pth
 - ../outputs/gnn_preds.json (node_id -> risk_prob)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as M
import pandas as pd

OUT_DIR = os.path.join("..", "outputs")
EMB_FILE = os.path.join(OUT_DIR, "embeddings.npy")
ADJ_FILE = os.path.join(OUT_DIR, "adjacency.npy")
EDGES_FILE = os.path.join(OUT_DIR, "edges.csv")
NODEIDS_FILE = os.path.join(OUT_DIR, "node_ids.json")
NODES_CSV = os.path.join(OUT_DIR, "nodes.csv")
OUT_PTH = os.path.join(OUT_DIR, "gnn_model.pth")
OUT_PRED = os.path.join(OUT_DIR, "gnn_preds.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load embeddings
emb = np.load(EMB_FILE)
N, D = emb.shape

# load nodes csv (contains label)
nodes_df = pd.read_csv(NODES_CSV, encoding="utf-8")
# ensure node_index ordering
nodes_df = nodes_df.sort_values("node_index").reset_index(drop=True)
labels = nodes_df["label"].fillna(-9).astype(int).values

# load or build adjacency
if os.path.exists(ADJ_FILE):
    adj = np.load(ADJ_FILE)
else:
    # build adjacency from edges.csv if adjacency.npy not present
    adj = np.zeros((N, N), dtype=float)
    if os.path.exists(EDGES_FILE):
        import csv
        with open(EDGES_FILE, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                src = row["src"]; dst = row["dst"]; w = float(row.get("weight", 1.0))
                try:
                    i = nodes_df[nodes_df["chunk_id"] == src]["node_index"].values[0]
                    j = nodes_df[nodes_df["chunk_id"] == dst]["node_index"].values[0]
                except Exception:
                    continue
                adj[i, j] += w
        adj = (adj + adj.T) / 2.0
    else:
        raise FileNotFoundError("Neither adjacency.npy nor edges.csv found. Run build_graph_edges.py first.")

# save adjacency & node_ids for downstream (if not exist)
if not os.path.exists(ADJ_FILE):
    np.save(ADJ_FILE, adj)

if os.path.exists(NODEIDS_FILE):
    with open(NODEIDS_FILE, "r", encoding="utf-8") as f:
        node_ids = json.load(f)
else:
    node_ids = nodes_df["chunk_id"].tolist()
    with open(NODEIDS_FILE, "w", encoding="utf-8") as f:
        json.dump(node_ids, f, ensure_ascii=False, indent=2)

# normalize adjacency (symmetric normalization)
A = torch.from_numpy(adj).float().to(device)
deg = A.sum(dim=1) + 1e-8
D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt

# build feature matrix: embeddings + label one-hot (categorical)
unique_labels = sorted(nodes_df['label'].unique())
label_map_for_ohe = {label: i for i, label in enumerate(unique_labels)}
num_types = len(unique_labels)
node_types_mapped = nodes_df['label'].map(label_map_for_ohe)
one_hot_types = F.one_hot(torch.tensor(node_types_mapped.values), num_classes=num_types).float()
X_emb = torch.tensor(emb, dtype=torch.float).to(device)
base_x = torch.cat([X_emb, one_hot_types.to(device)], dim=1)
feature_dim = base_x.shape[1]

print("N nodes:", N, "embedding_dim:", D, "num_types:", num_types, "feature_dim:", feature_dim)

# labels: use weak supervision where label in {0,1,2,...}
train_mask = (nodes_df['label'] >= 0)  # labels >=0 are considered supervised (customize if needed)
train_idx = np.where(train_mask.values)[0]
if len(train_idx) == 0:
    print("Warning: No supervised labels found. Model will not be trained in supervised mode.")
y = torch.tensor(np.where(nodes_df['label'].values == 1, 1, 0)).long().to(device)

# Define simple GCN
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
    def forward(self, x, adj):
        x = adj @ x
        x = self.lin(x)
        return x

class Net(nn.Module):
    def __init__(self, in_features, hidden=128):
        super().__init__()
        self.g1 = SimpleGCNLayer(in_features, hidden)
        self.g2 = SimpleGCNLayer(hidden, 2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x, adj):
        x = F.relu(self.g1(x, adj))
        x = self.dropout(x)
        x = self.g2(x, adj)
        return x

model = Net(in_features=feature_dim, hidden=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 200
X = base_x
for epoch in range(1, EPOCHS + 1):
    model.train()
    opt.zero_grad()
    out = model(X, A_norm)
    if len(train_idx) > 0:
        loss = loss_fn(out[train_idx], y[train_idx])
        loss.backward()
        opt.step()
    else:
        loss = torch.tensor(0.0)

    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            logits = model(X, A_norm).cpu()
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
            if len(train_idx) > 0:
                pred = (probs[train_idx] > 0.5).astype(int)
                true = nodes_df['label'].values[train_idx]
                # convert true to binary (1 vs not-1)
                true_bin = (true == 1).astype(int)
                acc = M.accuracy_score(true_bin, pred)
            else:
                acc = 0.0
        print(f"Epoch {epoch} loss={float(loss):.4f} train_acc={acc:.4f}")

# save model
torch.save(model.state_dict(), OUT_PTH)
print("Model saved to", OUT_PTH)

# save predictions
model.eval()
with torch.no_grad():
    logits = model(X, A_norm).cpu()
    probs = F.softmax(logits, dim=1)[:, 1].numpy()

pred_map = {node_ids[i]: float(probs[i]) for i in range(N)}
with open(OUT_PRED, "w", encoding="utf-8") as f:
    json.dump(pred_map, f, ensure_ascii=False, indent=2)
print("Predictions saved to", OUT_PRED)
