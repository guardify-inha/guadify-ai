"""
gnn_train.py
- Loads real graph data (nodes, embeddings, edges) to train a GCN model.
- Node features are a combination of text embeddings and node types.
- Trains the model to classify nodes as 'fair' (0) or 'unfair' (1).
- Saves the trained model weights and final predictions for all nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import os
import numpy as np
import sklearn.metrics as M
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. 데이터 로딩 및 경로 설정
# -------------------------------
print("✅ 1. Loading real data...")
OUT_DIR = os.path.join("..", "outputs")
NODES_FILE = os.path.join(OUT_DIR, "nodes.csv")
EMBED_FILE = os.path.join(OUT_DIR, "embeddings.npy")
EDGES_FILE = os.path.join(OUT_DIR, "edges.csv")

# 출력 경로
MODEL_PTH = os.path.join(OUT_DIR, "gnn_model.pth")
OUT_PRED = os.path.join(OUT_DIR, "gnn_preds.json")
os.makedirs(os.path.dirname(MODEL_PTH), exist_ok=True)

# 데이터 로드
nodes_df = pd.read_csv(NODES_FILE)
edges_df = pd.read_csv(EDGES_FILE)
all_embeddings = np.load(EMBED_FILE)

num_nodes = len(nodes_df)
print(f"   - Loaded {num_nodes} nodes and {len(edges_df)} edges.")

# chunk_id를 노드 인덱스로 변환하기 위한 딕셔너리 생성
chunk_id_to_idx = {row['chunk_id']: row['node_index'] for _, row in nodes_df.iterrows()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   - Using device: {device}")

# -------------------------------
# 2. 🧠 노드 특징(x) 생성 (임베딩 + 타입 정보)
# -------------------------------
print("✅ 2. Creating node features (x)...")
# 노드 타입(label)을 원-핫 인코딩
unique_labels = sorted(nodes_df['label'].unique())
label_map_for_ohe = {label: i for i, label in enumerate(unique_labels)}
node_types_mapped = nodes_df['label'].map(label_map_for_ohe)
one_hot_types = F.one_hot(torch.tensor(node_types_mapped.values), num_classes=len(unique_labels))

# 텍스트 임베딩과 원-핫 인코딩된 타입 정보를 결합
x_embeddings = torch.tensor(all_embeddings, dtype=torch.float)
x_types = one_hot_types.float()
x = torch.cat([x_embeddings, x_types], dim=1)
print(f"   - Node feature matrix 'x' created with shape: {x.shape}")

# -------------------------------
# 3. 🔗 인접 행렬(adj) 생성
# -------------------------------
print("✅ 3. Creating adjacency matrix (adj)...")
adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
for _, row in edges_df.iterrows():
    src_id = row['src']
    dst_id = row['dst']
    if src_id in chunk_id_to_idx and dst_id in chunk_id_to_idx:
        src_idx = chunk_id_to_idx[src_id]
        dst_idx = chunk_id_to_idx[dst_id]
        adj[src_idx, dst_idx] = 1.0
        adj[dst_idx, src_idx] = 1.0 # 무향 그래프
print("   - Adjacency matrix created.")

# -------------------------------
# 4. 🎯 정답 라벨(y) 및 학습/검증 마스크 생성
# -------------------------------
print("✅ 4. Creating labels (y) and masks...")
# GNN이 예측할 클래스: 0 (공정), 1 (불공정)
y = torch.full((num_nodes,), -1, dtype=torch.long) # 기본값 -1 (라벨 없음)
# reference(1), structured_reference(2) => 불공정(1)
y[nodes_df[nodes_df['label'].isin([1, 2])].index] = 1
# standard(0) => 공정(0)
y[nodes_df[nodes_df['label'] == 0].index] = 0

# 학습에 사용할 노드 인덱스 (라벨이 있는 노드만)
train_val_indices = torch.where(y != -1)[0]
train_val_labels = y[train_val_indices]

# 학습/검증 데이터 분리 (80% 학습, 20% 검증)
train_indices, val_indices, _, _ = train_test_split(
    train_val_indices,
    train_val_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_val_labels # 클래스 비율 유지
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask[val_indices] = True
print(f"   - Training nodes: {train_mask.sum().item()}, Validation nodes: {val_mask.sum().item()}")

# -------------------------------
# 5. GCN 모델 정의 (기존 코드와 동일)
# -------------------------------
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(1) + 1e-6))
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        x = adj_norm @ x
        x = self.linear(x)
        return x

class Net(nn.Module):
    def __init__(self, in_features, hidden_features=64, num_classes=2):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(in_features, hidden_features)
        self.gcn2 = SimpleGCNLayer(hidden_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return x

# -------------------------------
# 6. 학습 실행
# -------------------------------
print("✅ 6. Starting model training...")
# 모델 입력 크기를 실제 x의 차원으로 수정
model = Net(in_features=x.shape[1], num_classes=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

x = x.to(device)
adj = adj.to(device)
y = y.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)

best_val_acc = 0
for epoch in range(1, 201):
    model.train()
    opt.zero_grad()
    out = model(x, adj)
    loss = loss_fn(out[train_mask], y[train_mask])
    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            val_logits = logits[val_mask]
            pred = val_logits.argmax(1)
            true = y[val_mask]
            acc = M.accuracy_score(true.cpu(), pred.cpu())
            if acc > best_val_acc:
                best_val_acc = acc
                # 최고의 성능일 때 모델 저장
                torch.save(model.state_dict(), MODEL_PTH)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f} | 🚀 Model Saved!")
            else:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")

print("   - Training finished.")
# -------------------------------
# 7. 전체 노드에 대한 예측 결과 저장
# -------------------------------
print("✅ 7. Saving final predictions for all nodes...")
# 최고의 성능을 보인 모델 불러오기
model.load_state_dict(torch.load(MODEL_PTH))
model.eval()
with torch.no_grad():
    final_logits = model(x, adj).cpu()
    # '불공정' 클래스(1)에 대한 확률 계산
    final_probs = F.softmax(final_logits, dim=1)[:, 1].numpy()

pred_map = {cid: float(prob) for cid, prob in zip(nodes_df["chunk_id"], final_probs)}

with open(OUT_PRED, "w", encoding="utf-8") as f:
    json.dump(pred_map, f, ensure_ascii=False, indent=2)

print(f"   - Model saved to: {MODEL_PTH}")
print(f"   - GNN predictions saved to: {OUT_PRED}")
print("🎉 All tasks completed.")