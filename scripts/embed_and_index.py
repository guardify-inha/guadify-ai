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

model = SentenceTransformer(EMBED_MODEL)
texts = []
meta = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        meta.append({"chunk_id": obj["chunk_id"], "source_file": obj["source_file"], "chunk_idx": obj["chunk_idx"], "text": obj["text"]})

print("Embedding", len(texts), "chunks ...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype("float32")

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(meta, f)

print("Index saved:", INDEX_FILE, "Meta saved:", META_FILE)
