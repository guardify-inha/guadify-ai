"""
Creates embeddings for:
 - ../outputs/chunks.jsonl (all chunk types produced by extract_and_chunk_gpt.py)
 - ../outputs/structured_reference.jsonl (GPT-structured JSON)
Outputs:
 - ../outputs/embeddings.npy
 - ../outputs/faiss.index
 - ../outputs/faiss_meta.pkl
 - ../outputs/nodes.csv   (GNN node metadata + label)
Notes:
 - Uses SentenceTransformer for Korean legal embeddings.
 - Normalizes embeddings and builds FAISS IndexFlatIP (cosine similarity).
"""

import os
import json
import pickle
import csv
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --------- config ----------
OUT_DIR = os.path.join("..", "outputs")
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
STRUCTURED_FILE = os.path.join(OUT_DIR, "structured_reference.jsonl")

# Embedding model (change if you prefer another legal-specific model)
EMBED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# Output artifact paths
EMB_FILE = os.path.join(OUT_DIR, "embeddings.npy")
INDEX_FILE = os.path.join(OUT_DIR, "faiss.index")
META_FILE = os.path.join(OUT_DIR, "faiss_meta.pkl")
NODES_CSV = os.path.join(OUT_DIR, "nodes.csv")
LABEL_MAP_FILE = os.path.join(OUT_DIR, "label_map.json")

# batch size for encoding
BATCH_SIZE = 64

# label mapping for GNN (you can adjust as desired)
LABEL_MAP = {
    "reference": 1,           # 원문 시정자료 청크 (source reference)
    "structured_reference": 2,# GPT가 정리한 불공정조항 JSON
    "standard": 0,            # 표준약관 (공정 기준)
    "law": -2,                # 법령 문장 (판단 근거)
    "user": -1,               # 사용자 업로드 약관
    "unknown": -9
}

# --------- helper functions ----------
def build_structured_embed_text(struct_obj):
    """
    For a structured_reference JSON object (from GPT), build a single text string to embed.
    Prefer to combine 불공정조항 + 시정이유 + 관련법조항 + 출처 for richer embedding.
    """
    pieces = []
    for k in ["불공정조항", "시정이유", "관련법조항", "출처"]:
        v = struct_obj.get(k) or struct_obj.get(k.replace("불공정", "unfair")) or ""
        if isinstance(v, str) and v.strip():
            pieces.append(v.strip())
    return " || ".join(pieces) if pieces else struct_obj.get("text", "")

# --------- main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = []   # list[str] for embedding
    meta = []    # list[dict] meta parallel to texts

    # 1) Load chunks.jsonl
    if os.path.exists(CHUNKS_FILE):
        print(f"Loading chunks from {CHUNKS_FILE} ...")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                source_tag = obj.get("source_tag", "unknown")
                level = obj.get("level", "unknown")
                if source_tag == "law":
                    title = obj.get("article_title") or ""
                    t = (title + " " + (obj.get("text") or "")).strip()
                elif source_tag == "standard":
                    title = obj.get("article_title") or ""
                    t = (title + " " + (obj.get("text") or "")).strip()
                else:
                    t = obj.get("text") or ""

                if not t or not str(t).strip():
                    continue

                texts.append(t)
                meta.append({
                    "chunk_id": obj.get("chunk_id") or f"chunk_{len(meta)}",
                    "source_file": obj.get("source_file"),
                    "source_tag": source_tag,
                    "page": obj.get("page"),
                    "chunk_idx": obj.get("chunk_idx"),
                    "level": level,
                    "text": t,
                })
    else:
        print(f"Warning: {CHUNKS_FILE} not found. No chunk embeddings will be created from it.")

    # 2) Load structured_reference.jsonl and add as separate entries
    if os.path.exists(STRUCTURED_FILE):
        print(f"Loading structured references from {STRUCTURED_FILE} ...")
        with open(STRUCTURED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = build_structured_embed_text(obj)
                if not t or not str(t).strip():
                    continue
                texts.append(t)
                meta.append({
                    "chunk_id": obj.get("chunk_id") or str(uuid.uuid4()),
                    "source_file": obj.get("source_file"),
                    "source_tag": "structured_reference",
                    "page": None,
                    "chunk_idx": None,
                    "level": "structured_reference",
                    "text": t,
                    "structured_fields": {
                        "불공정조항": obj.get("불공정조항"),
                        "시정이유": obj.get("시정이유"),
                        "관련법조항": obj.get("관련법조항"),
                        "출처": obj.get("출처")
                    }
                })
    else:
        print(f"Note: {STRUCTURED_FILE} not found. No structured_reference embeddings will be created.")

    n_texts = len(texts)
    if n_texts == 0:
        print("No texts to embed. Exiting.")
        return

    print(f"Total texts to embed: {n_texts}")

    # 3) Create embeddings in batches
    print("Encoding embeddings ...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # 4) Normalize embeddings (for cosine similarity using IndexFlatIP)
    print("Normalizing embeddings ...")
    faiss.normalize_L2(embeddings)

    # 5) Save numpy array
    print(f"Saving embeddings to {EMB_FILE} ...")
    np.save(EMB_FILE, embeddings)

    # 6) Build FAISS index (Inner Product on normalized vectors => cosine)
    d = embeddings.shape[1]
    print(f"Building FAISS index (dim={d}) ...")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS index saved to {INDEX_FILE}")

    # 7) Save meta
    print(f"Saving meta to {META_FILE} ...")
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

    # 8) Write nodes.csv for GNN
    print(f"Writing nodes CSV to {NODES_CSV} ...")
    with open(NODES_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_index", "chunk_id", "source_file", "source_tag", "page",
            "chunk_idx", "level", "label", "text"
        ])
        for idx, m in enumerate(meta):
            tag = m.get("source_tag", "unknown")
            label = LABEL_MAP.get(tag, LABEL_MAP["unknown"])
            txt = (m.get("text") or "").replace("\n", " ").strip()
            writer.writerow([
                idx,
                m.get("chunk_id"),
                m.get("source_file"),
                tag,
                m.get("page"),
                m.get("chunk_idx"),
                m.get("level"),
                label,
                txt[:2000]
            ])

    # 9) Save LABEL_MAP so inference can reuse ordering/values
    try:
        with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)
        print(f"Saved label map to {LABEL_MAP_FILE}")
    except Exception as e:
        print("Warning: unable to save label_map.json:", e)

    print("All artifacts saved:")
    print(" -", EMB_FILE)
    print(" -", INDEX_FILE)
    print(" -", META_FILE)
    print(" -", NODES_CSV)
    print(" -", LABEL_MAP_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
