"""
query_pipeline.py
- Replaces the old query_pipeline.py.
- This is the final step: GNN Inference.
- Loads the pre-trained GCN model and the full graph data.
- Takes candidate clauses (from query_and_extract.py) as input.
- Dynamically adds the new clauses to the graph, runs the GNN model,
  and predicts their 'unfairness' probability.
- Saves the final sorted results to a JSON file and a readable summary.
"""
import torch
import torch.nn.functional as F
import pandas as pd
import json
import os
import numpy as np
import argparse
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# 1. ⚙️ 설정 및 모델/데이터 로드
# -------------------------------
print("✅ 1. Loading models and base graph data...")

# 경로 설정
OUT_DIR = os.path.join("..", "outputs")
MODEL_PTH = os.path.join(OUT_DIR, "gnn_model.pth")
NODES_FILE = os.path.join(OUT_DIR, "nodes.csv")
EMBED_FILE = os.path.join(OUT_DIR, "embeddings.npy")
EDGES_FILE = os.path.join(OUT_DIR, "edges.csv")
INDEX_FILE = os.path.join(OUT_DIR, "faiss.index")
RESULT_JSON = os.path.join(OUT_DIR, "final_results.json")
SUMMARY_TXT = os.path.join(OUT_DIR, "summary.txt")

# 임베딩 모델
EMBED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# GNN 추론 하이퍼파라미터
SIMILARITY_TOP_K = 5 # 새로운 조항을 기존 그래프와 연결할 때 찾을 유사 노드 수

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   - Using device: {device}")

# GNN 모델 클래스 (gnn_train.py와 동일하게 정의해야 모델 로드 가능)
class SimpleGCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
    def forward(self, x, adj):
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(1) + 1e-6))
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        x = adj_norm @ x
        x = self.linear(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, in_features, hidden_features=64, num_classes=2):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(in_features, hidden_features)
        self.gcn2 = SimpleGCNLayer(hidden_features, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return x

# 모델 및 데이터 로드
try:
    # 베이스 그래프 데이터
    nodes_df = pd.read_csv(NODES_FILE)
    edges_df = pd.read_csv(EDGES_FILE)
    base_embeddings = np.load(EMBED_FILE)
    # 임베딩 및 FAISS 모델
    embed_model = SentenceTransformer(EMBED_MODEL)
    faiss_index = faiss.read_index(INDEX_FILE)
except FileNotFoundError as e:
    print(f"Error: Required file not found. {e}")
    print("Please run embed_and_index.py, build_graph.py, and gnn_train.py first.")
    exit(1)

num_base_nodes = len(nodes_df)

# -------------------------------
# 2. 🧠 추론을 위한 그래프 데이터 준비
# -------------------------------
# gnn_train.py와 동일한 방식으로 베이스 그래프의 특징(x)과 인접 행렬(adj)을 준비합니다.

# 특징(x) 생성
unique_labels = sorted(nodes_df['label'].unique())
label_map_for_ohe = {label: i for i, label in enumerate(unique_labels)}
num_types = len(unique_labels)
node_types_mapped = nodes_df['label'].map(label_map_for_ohe)
one_hot_types = F.one_hot(torch.tensor(node_types_mapped.values), num_classes=num_types)
x_embeddings = torch.tensor(base_embeddings, dtype=torch.float)
x_types = one_hot_types.float()
base_x = torch.cat([x_embeddings, x_types], dim=1)
feature_dim = base_x.shape[1]

# 인접 행렬(adj) 생성
chunk_id_to_idx = {row['chunk_id']: row['node_index'] for _, row in nodes_df.iterrows()}
base_adj = torch.zeros((num_base_nodes, num_base_nodes), dtype=torch.float)
for _, row in edges_df.iterrows():
    if row['src'] in chunk_id_to_idx and row['dst'] in chunk_id_to_idx:
        src_idx, dst_idx = chunk_id_to_idx[row['src']], chunk_id_to_idx[row['dst']]
        base_adj[src_idx, dst_idx] = 1.0
        base_adj[dst_idx, src_idx] = 1.0

print("✅ 2. Base graph features and adjacency matrix are ready.")

# -------------------------------
# 3. 🚀 GNN 추론 실행 함수
# -------------------------------
def run_gnn_inference(candidate_clauses):
    """후보 조항들을 기존 그래프에 동적으로 추가하고 GNN 추론을 실행합니다."""
    print("✅ 3. Starting GNN inference for new clauses...")
    if not candidate_clauses:
        print("   - No candidate clauses to analyze.")
        return []

    num_new_nodes = len(candidate_clauses)
    
    # 1. 새로운 노드(후보 조항)의 특징 벡터 생성
    new_texts = [c['text'] for c in candidate_clauses]
    new_embeddings = embed_model.encode(new_texts, convert_to_tensor=False).astype("float32")
    
    # 'user' 타입(-1)에 대한 원-핫 벡터 생성
    user_type_idx = label_map_for_ohe.get(-1)
    if user_type_idx is None:
        raise ValueError("'user' type (label: -1) not found in training data labels.")
    user_type_ohe = F.one_hot(torch.tensor([user_type_idx] * num_new_nodes), num_classes=num_types).float()
    
    new_x = torch.cat([torch.tensor(new_embeddings), user_type_ohe], dim=1)

    # 2. 새로운 노드를 기존 그래프와 연결 (FAISS 사용)
    faiss.normalize_L2(new_embeddings)
    distances, indices = faiss_index.search(new_embeddings, SIMILARITY_TOP_K)
    
    new_edges = []
    for i in range(num_new_nodes): # i는 새로운 노드의 상대적 인덱스 (0, 1, 2...)
        new_node_abs_idx = num_base_nodes + i # 전체 그래프에서의 실제 인덱스
        for j in range(SIMILARITY_TOP_K):
            base_node_idx = indices[i][j]
            if base_node_idx >= 0:
                new_edges.append((new_node_abs_idx, base_node_idx))

    # 3. 확장된 그래프(Expanded Graph) 생성
    total_nodes = num_base_nodes + num_new_nodes
    expanded_x = torch.cat([base_x, new_x], dim=0)
    expanded_adj = F.pad(base_adj, (0, num_new_nodes, 0, num_new_nodes))
    for u, v in new_edges:
        expanded_adj[u, v] = 1.0
        expanded_adj[v, u] = 1.0
    
    print(f"   - Expanded graph created with {total_nodes} total nodes.")

    # 4. GNN 모델 로드 및 추론 실행
    model = Net(in_features=feature_dim, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(expanded_x.to(device), expanded_adj.to(device))
        probs = F.softmax(logits, dim=1)
        # 새로운 노드에 대한 '불공정'(클래스 1) 확률만 추출
        unfair_probs = probs[num_base_nodes:, 1].cpu().numpy()

    # 5. 결과 정리
    results = []
    for i, clause in enumerate(candidate_clauses):
        results.append({
            "article": clause.get("article", "N/A"),
            "text": clause["text"],
            "unfair_prob_gnn": round(float(unfair_probs[i]), 4)
        })
        
    # 확률 높은 순으로 정렬
    results.sort(key=lambda x: x['unfair_prob_gnn'], reverse=True)
    print("   - Inference finished.")
    return results

# -------------------------------
# 4. 📊 결과 저장 및 출력
# -------------------------------
def save_results(results):
    """추론 결과를 JSON과 TXT 파일로 저장합니다."""
    print("✅ 4. Saving results...")
    
    # JSON 저장
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"   - Full results saved to {RESULT_JSON}")

    # 요약 TXT 저장
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("==== GNN 불공정 약관 조항 분석 결과 ====\n\n")
        if not results:
            f.write("분석된 위험 후보 조항이 없습니다.\n")
        for res in results:
            prob_percent = res['unfair_prob_gnn'] * 100
            f.write(f"🚨 [위험도: {prob_percent:.2f}%] - {res['article']}\n")
            f.write(f"   - 원문: {res['text'][:200].strip()}...\n")
            f.write("-" * 60 + "\n")
    print(f"   - Summary saved to {SUMMARY_TXT}")

# -------------------------------
# 5. 🎬 메인 실행 블록
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN inference on candidate clauses.")
    parser.add_argument(
        "candidate_file",
        help="Path to the candidate clauses file (e.g., ../outputs/candidate_clauses.jsonl)"
    )
    args = parser.parse_args()

    try:
        with open(args.candidate_file, "r", encoding="utf-8") as f:
            clauses_to_analyze = [json.loads(line) for line in f]
        
        final_results = run_gnn_inference(clauses_to_analyze)
        save_results(final_results)
        print("\n🎉 All tasks completed.")

    except FileNotFoundError:
        print(f"Error: Candidate file not found at {args.candidate_file}")
        print("Please run query_and_extract.py first to generate the candidate clauses.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")