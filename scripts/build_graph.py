"""
Builds graph edges from embeddings and metadata.
Edges include:
- Adjacent chunks in same document
- Semantic similarity (FAISS top-k)
- Keyword co-occurrence
Outputs: edges.csv
"""

import os
import pickle
import numpy as np
import faiss
import csv

# 입력 / 출력 경로
META_FILE = os.path.join("..", "outputs", "faiss_meta.pkl")
EMB_FILE = os.path.join("..", "outputs", "embeddings.npy")
EDGES_CSV = os.path.join("..", "outputs", "edges.csv")

# 파라미터
TOP_K = 5          # 유사도 검색 시 top-k
SIM_THRESH = 0.6   # 유사도 threshold
KEYWORDS = ["해지", "손해배상", "면책", "위약금", "지체", "해약", "환불"]

# 1. 메타/임베딩 로드
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

embs = np.load(EMB_FILE)

# 2. id 매핑
ids = [m["chunk_id"] for m in meta]
id_to_idx = {cid: i for i, cid in enumerate(ids)}

edges = []

# 3. 같은 문서 내 인접 청크 연결
for i, m in enumerate(meta):
    if i + 1 < len(meta) and meta[i + 1]["source_file"] == m["source_file"]:
        edges.append((m["chunk_id"], meta[i + 1]["chunk_id"], 1.0, "adjacent"))
        edges.append((meta[i + 1]["chunk_id"], m["chunk_id"], 1.0, "adjacent"))

# 4. FAISS 기반 semantic similarity
embs_norm = embs.copy()
faiss.normalize_L2(embs_norm)
index = faiss.IndexFlatIP(embs_norm.shape[1])
index.add(embs_norm)

D, I = index.search(embs_norm, TOP_K + 1)  # 자기 자신 포함하므로 +1
for i_row, row in enumerate(I):
    src = ids[i_row]
    for j_pos, j in enumerate(row[1:]):  # 첫 번째는 self → 스킵
        sim = float(D[i_row, j_pos + 1])
        dst = ids[j]
        if sim >= SIM_THRESH:
            edges.append((src, dst, sim, "faiss_sim"))

# 5. 키워드 기반 연결
for i, m in enumerate(meta):
    text_i = m.get("text", "")
    for j in range(i + 1, len(meta)):
        text_j = meta[j].get("text", "")
        for kw in KEYWORDS:
            if kw in text_i and kw in text_j:
                edges.append((m["chunk_id"], meta[j]["chunk_id"], 0.8, "keyword"))
                edges.append((meta[j]["chunk_id"], m["chunk_id"], 0.8, "keyword"))
                break  # 한 키워드만 매칭해도 연결

# 6. edges.csv 저장
with open(EDGES_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["src", "dst", "weight", "type"])
    for e in edges:
        w.writerow(e)

print(f"Edges written: {len(edges)} -> {EDGES_CSV}")
