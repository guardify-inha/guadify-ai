"""
Creates embeddings, saves embeddings.npy, faiss.index, and meta pickle.
Also emits nodes.csv (node metadata + label placeholder) for graph building.
"""

import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import csv

# 입력 / 출력 경로
CHUNKS_FILE = os.path.join("..", "outputs", "chunks.jsonl")
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = os.path.join("..", "outputs", "faiss.index")
META_FILE = os.path.join("..", "outputs", "faiss_meta.pkl")
EMB_FILE = os.path.join("..", "outputs", "embeddings.npy")
NODES_CSV = os.path.join("..", "outputs", "nodes.csv")

# 임베딩 모델 로드
model = SentenceTransformer(EMBED_MODEL)

texts = []
meta = []

# 1. 청크 로드
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        meta.append({
            "chunk_id": obj.get("chunk_id"),
            "source_file": obj.get("source_file"),
            "source_tag": obj.get("source_tag", "unknown"),
            "page": obj.get("page"),
            "chunk_idx": obj.get("chunk_idx"),
            "level": obj.get("level", "unknown"),
            "text": obj.get("text"),
        })

# 2. 임베딩 생성
print(f"Embedding {len(texts)} chunks using model {EMBED_MODEL} ...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype("float32")

# 3. 임베딩 저장
np.save(EMB_FILE, embeddings)

# 4. FAISS 인덱스 생성
d = embeddings.shape[1]
faiss.normalize_L2(embeddings)  # cosine 유사도 기반 검색용
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# 5. 인덱스 & 메타 저장
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(meta, f)

# 6. nodes.csv 저장 (GNN 학습용)
with open(NODES_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "chunk_id", "source_file", "source_tag", "page",
        "chunk_idx", "level", "label", "text"
    ])
    for m in meta:
        tag = m.get("source_tag")
        # label heuristic: reference = positive(1), standard = negative(0), others unknown(-1)
        if tag == "reference":
            label = 1
        elif tag == "standard":
            label = 0
        else:
            label = -1
        writer.writerow([
            m.get("chunk_id"),
            m.get("source_file"),
            tag,
            m.get("page"),
            m.get("chunk_idx"),
            m.get("level"),
            label,
            m.get("text", "").replace("\n", " ")[:1000]  # CSV용 줄바꿈 제거
        ])

print("Index saved:", INDEX_FILE)
print("Meta saved:", META_FILE)
print("Embeddings saved:", EMB_FILE)
print("Nodes CSV saved:", NODES_CSV)
