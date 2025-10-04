"""
Train a GNN model on the contract graph and output predictions.
"""

import torch
import json
import sklearn.metrics as M

# ⚠️ 주의: 아래 변수들은 외부에서 준비되어 있어야 합니다
# - data: PyG Data 객체
# - train_mask, test_mask: 학습/테스트 마스크 (torch.bool)
# - nodes_df: node 메타데이터 (pandas DataFrame, chunk_id 포함)
# - Net: GNN 모델 클래스
# - device: torch.device
# - MODEL_PTH, OUT_PRED: 저장 경로 (string)

# 모델 초기화
model = Net(data.num_node_features).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

best_val = 0.0
for epoch in range(1, 201):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)

    if train_mask.sum() > 0:
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()
    else:
        loss = torch.tensor(0.0)

    # 평가
    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        logits = model(data.x, data.edge_index).detach().cpu()

        if test_mask.sum() > 0:
            pred = logits[test_mask].argmax(dim=1).numpy()
            true = data.y[test_mask].cpu().numpy()
            acc = M.accuracy_score(true, pred) if len(true) > 0 else 0.0
        else:
            acc = 0.0

        print(f"Epoch {epoch} loss={loss.item():.4f} test_acc={acc:.4f}")

# 모델 저장
torch.save(model.state_dict(), MODEL_PTH)
print("Model saved:", MODEL_PTH)

# 노드별 확률 예측
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index).cpu()
    # class 1 확률
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()

# chunk_id → 확률 매핑
pred_map = {cid: float(probs[i]) for i, cid in enumerate(nodes_df["chunk_id"].tolist())}
with open(OUT_PRED, "w", encoding="utf-8") as f:
    json.dump(pred_map, f, ensure_ascii=False, indent=2)

print("GNN predictions saved:", OUT_PRED)
