# scripts/embed_and_index.py
import json, os, pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

CHUNKS_FILE = "../outputs/chunks.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "../outputs/faiss.index"
META_FILE = "../outputs/faiss_meta.pkl"

# 1. 모델 로드
model = SentenceTransformer(EMBED_MODEL)

texts = []
meta = []

# 2. 청크 파일 읽기
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        meta.append({
            "chunk_id": obj.get("chunk_id"),
            "source_file": obj.get("source_file"),
            "source_tag": obj.get("source_tag", "unknown"),  # 추가
            "page": obj.get("page"),
            "chunk_idx": obj.get("chunk_idx"),
            "text": obj.get("text"),
        })

print(f"Embedding {len(texts)} chunks using model {EMBED_MODEL} ...")

# 3. 임베딩 생성
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype("float32")

# 4. FAISS 인덱스 생성 및 추가
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 5. 저장
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(meta, f)

print("Index saved:", INDEX_FILE)
print("Meta saved:", META_FILE)
