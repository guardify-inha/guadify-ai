# File: scripts/query_and_extract.py
"""
Extract candidate clauses for GNN analysis:
 - split user contract into clauses
 - remove clauses that are too similar to 'standard' clauses (FAISS)
 - optional: call LLM to prune obviously safe clauses
Outputs:
 - ../outputs/candidate_clauses.jsonl
"""

import os
import re
import json
import argparse
import pickle
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ----------------- 설정 -----------------
OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 800
OPENAI_TEMPERATURE = 0.0

OUT_DIR = os.path.join("..", "outputs")
INDEX_FILE = os.path.join(OUT_DIR, "faiss.index")
META_FILE = os.path.join(OUT_DIR, "faiss_meta.pkl")
CANDIDATES_FILE = os.path.join(OUT_DIR, "candidate_clauses.jsonl")

EMBED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
SIMILARITY_THRESHOLD = 0.75  # 표준 약관과의 유사도 기준

ARTICLE_RE = re.compile(r'(제\s*\d+\s*조[^\n\r]*)')

# ----------------- helpers -----------------
def split_by_article(text):
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [{"article": "전체", "text": text.strip()}]
    results = []
    for i, m in enumerate(matches):
        start_idx = m.start()
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(text)
        article_title = m.group(1).strip()
        body = text[start_idx:end_idx].strip()
        results.append({"article": article_title, "text": body})
    return results

print("Loading embedding model and FAISS index...")
embed_model = SentenceTransformer(EMBED_MODEL)
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        faiss_meta = pickle.load(f)
else:
    raise FileNotFoundError("FAISS index/meta not found. Run embed_and_index.py first.")

def search_standard_similarity(query_text, top_k=3):
    q_emb = embed_model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, indices = faiss_index.search(q_emb, top_k)
    max_sim = 0.0
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        rec = faiss_meta[idx]
        if rec.get("source_tag") == "standard":
            max_sim = max(max_sim, float(dist))
    return max_sim

def call_llm_to_prune_clauses(chunks, api_key=None):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set; skipping LLM pruning.")
        return []
    openai.api_key = api_key

    assembled = [f"[chunk{i}] {c.get('text')}" for i, c in enumerate(chunks)]
    system_msg = (
        "You are a meticulous Korean legal assistant. Identify clauses that are "
        "absolutely standard and safe to ignore. Return JSON: {\"safe_chunk_indices\": [int,...]}"
    )
    user_msg = "Identify safe clauses from:\n\n" + "\n\n".join(assembled)
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS
        )
        content = resp.choices[0].message.content
        # parse JSON substring
        try:
            obj = json.loads(content)
        except Exception:
            m = re.search(r"(\{.*\})", content, flags=re.S)
            if m:
                obj = json.loads(m.group(1))
            else:
                return []
        safe_indices = obj.get("safe_chunk_indices", [])
        if isinstance(safe_indices, list):
            return safe_indices
        return []
    except Exception as e:
        print("LLM pruning error:", e)
        return []

# ----------------- main pipeline -----------------
def extract_candidate_clauses(text, use_llm_filter=True, save_path=CANDIDATES_FILE):
    print("Starting clause extraction pipeline...")
    clauses = split_by_article(text)
    print(f" - {len(clauses)} clauses found")

    candidates_after_faiss = []
    for c in clauses:
        max_sim = search_standard_similarity(c["text"])
        if max_sim >= SIMILARITY_THRESHOLD:
            continue
        candidates_after_faiss.append(c)
    print(f" - {len(candidates_after_faiss)} remain after FAISS filter")

    if use_llm_filter and candidates_after_faiss:
        safe_indices = call_llm_to_prune_clauses(candidates_after_faiss)
        print(f" - LLM pruned {len(safe_indices)} clauses as safe")
        final_candidates = [c for i, c in enumerate(candidates_after_faiss) if i not in safe_indices]
    else:
        final_candidates = candidates_after_faiss

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for clause in final_candidates:
                f.write(json.dumps(clause, ensure_ascii=False) + "\n")
        print("Saved candidates to", save_path)

    return final_candidates

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="contract text file")
    parser.add_argument("--no-llm", action="store_true", help="skip LLM pruning")
    args = parser.parse_args()
    with open(args.file, "r", encoding="utf-8") as f:
        txt = f.read()
    extract_candidate_clauses(txt, use_llm_filter=not args.no_llm)
