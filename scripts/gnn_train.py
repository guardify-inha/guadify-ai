"""
gnn_train.py
- Pure PyTorch version: Train a simple GCN-like model
- Save model and node-level risk probabilities (gnn_preds.json)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import os
import numpy as np
import sklearn.metrics as M

# -------------------------------
# 1. 준비 (샘플 데이터/마스크)
# -------------------------------

num_nodes = 10
num_features = 5
num_classes = 2

# 랜덤 노드 feature
x = torch.randn((num_nodes, num_features), dtype=torch.float)

# 인접 행렬 (undirected)
adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,0),(0,2),(1,3)]
for i,j in edges:
    adj[i,j] = 1
    adj[j,i] = 1  # 무향 그래프

# 라벨
y = torch.randint(0, num_classes, (num_nodes,))

# 학습/테스트 마스크
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[:6] = True
test_mask = ~train_mask

# nodes_df: chunk_id 포함
nodes_df = pd.DataFrame({"chunk_id": [f"chunk_{i}" for i in range(num_nodes)]})

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 출력 경로
MODEL_PTH = os.path.join("..", "outputs", "gnn_model.pth")
OUT_PRED = os.path.join("..", "outputs", "gnn_preds.json")
os.makedirs(os.path.dirname(MODEL_PTH), exist_ok=True)

# -------------------------------
# 2. Simple GCN layer
# -------------------------------
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # Degree normalization
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(1) + 1e-6))
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        x = adj_norm @ x
        x = self.linear(x)
        return x

# -------------------------------
# 3. GCN Model
# -------------------------------
class Net(nn.Module):
    def __init__(self, in_features, hidden_features=16, num_classes=2):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(in_features, hidden_features)
        self.gcn2 = SimpleGCNLayer(hidden_features, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return x

# -------------------------------
# 4. 학습
# -------------------------------
model = Net(num_features).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

x = x.to(device)
adj = adj.to(device)
y = y.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)

for epoch in range(1, 201):
    model.train()
    opt.zero_grad()
    out = model(x, adj)
    if train_mask.sum() > 0:
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()
    else:
        loss = torch.tensor(0.0)

    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        logits = out.detach().cpu()
        pred = logits[test_mask].argmax(1).numpy()
        true = y[test_mask].cpu().numpy()
        acc = M.accuracy_score(true, pred) if len(true) > 0 else 0.0
        print(f"Epoch {epoch} loss={loss.item():.4f} test_acc={acc:.4f}")

# -------------------------------
# 5. 모델 저장
# -------------------------------
torch.save(model.state_dict(), MODEL_PTH)
print("Model saved:", MODEL_PTH)

# -------------------------------
# 6. 노드별 확률 예측
# -------------------------------
model.eval()
with torch.no_grad():
    logits = model(x, adj).cpu()
    probs = F.softmax(logits, dim=1)[:,1].numpy()

pred_map = {cid: float(probs[i]) for i, cid in enumerate(nodes_df["chunk_id"].tolist())}
with open(OUT_PRED, "w", encoding="utf-8") as f:
    json.dump(pred_map, f, ensure_ascii=False, indent=2)

print("GNN predictions saved:", OUT_PRED)
