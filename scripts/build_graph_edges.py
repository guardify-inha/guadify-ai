# File: scripts/build_graph_edges.py
"""
Builds graph edges from embeddings and metadata.
Edges include:
- Adjacent chunks in same document
- Semantic similarity (FAISS top-k)
- Keyword co-occurrence
Outputs: edges.csv, adjacency.npy (symmetric), node_ids.json
"""

import os
import pickle
import numpy as np
import faiss
import csv
import json

OUT_DIR = os.path.join("..", "outputs")
META_FILE = os.path.join(OUT_DIR, "faiss_meta.pkl")
EMB_FILE = os.path.join(OUT_DIR, "embeddings.npy")
EDGES_CSV = os.path.join(OUT_DIR, "edges.csv")
ADJ_FILE = os.path.join(OUT_DIR, "adjacency.npy")
NODEIDS_FILE = os.path.join(OUT_DIR, "node_ids.json")

# 파라미터
TOP_K = 6          # 유사도 검색 시 top-k
SIM_THRESH = 0.45  # 유사도 threshold (inner product on normalized vectors ~ cosine)
KEYWORDS = ["해지", "손해배상", "면책", "위약금", "지체", "해약", "환불"]

# 1. 메타/임베딩 로드
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

embs = np.load(EMB_FILE)  # already normalized in embed step

N = embs.shape[0]
ids = [m.get("chunk_id") or f"chunk_{i}" for i, m in enumerate(meta)]
id_to_idx = {cid: i for i, cid in enumerate(ids)}

edges = []

# 2. 같은 문서 내 인접 청크 연결 (bidirectional)
for i, m in enumerate(meta):
    # look forward for next chunk in same source_file
    for j in range(i+1, min(i+4, N)):  # only check next few to limit edges
        if meta[j].get("source_file") == m.get("source_file"):
            edges.append((m["chunk_id"], meta[j]["chunk_id"], 1.0, "adjacent"))
            edges.append((meta[j]["chunk_id"], m["chunk_id"], 1.0, "adjacent"))
            break

# 3. FAISS 기반 semantic similarity
# ensure normalized
embs_norm = embs.copy().astype("float32")
faiss.normalize_L2(embs_norm)
index = faiss.IndexFlatIP(embs_norm.shape[1])
index.add(embs_norm)

D, I = index.search(embs_norm, TOP_K + 1)  # self included
for i_row, row in enumerate(I):
    src = ids[i_row]
    for j_pos, j in enumerate(row[1:]):  # skip self
        sim = float(D[i_row, j_pos + 1])
        dst = ids[j]
        if sim >= SIM_THRESH:
            edges.append((src, dst, sim, "faiss_sim"))

# 4. 키워드 기반 연결
for i, m in enumerate(meta):
    text_i = (m.get("text") or "").lower()
    for j in range(i + 1, N):
        text_j = (meta[j].get("text") or "").lower()
        for kw in KEYWORDS:
            if kw in text_i and kw in text_j:
                edges.append((m["chunk_id"], meta[j]["chunk_id"], 0.8, "keyword"))
                edges.append((meta[j]["chunk_id"], m["chunk_id"], 0.8, "keyword"))
                break

# 5. Save edges.csv
with open(EDGES_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["src", "dst", "weight", "type"])
    for e in edges:
        w.writerow(e)

print(f"Edges written: {len(edges)} -> {EDGES_CSV}")

# 6. Build adjacency matrix (N x N) using summed weights, and save numpy
adj = np.zeros((N, N), dtype=float)
for src, dst, w, _t in edges:
    i = id_to_idx.get(src)
    j = id_to_idx.get(dst)
    if i is None or j is None:
        continue
    adj[i, j] += float(w)

# Make symmetric (undirected)
adj = (adj + adj.T) / 2.0

# optional: sparsify tiny weights
adj[adj < 1e-6] = 0.0

np.save(ADJ_FILE, adj)
with open(NODEIDS_FILE, "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False, indent=2)

print(f"Adjacency saved: {ADJ_FILE}")
print(f"Node ids saved: {NODEIDS_FILE}")
