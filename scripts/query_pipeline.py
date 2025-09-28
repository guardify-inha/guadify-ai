# scripts/query_pipeline.py
import os, json, pickle, re, sys
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import openai

# ----------------- 설정 -----------------
INDEX_FILE = "../outputs/faiss.index"
META_FILE = "../outputs/faiss_meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
CANDIDATE_MAX = 20  # LLM 후보 최대 개수

# load FAISS index & meta
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

model = SentenceTransformer(EMBED_MODEL)

# ----------------- OpenAI API -----------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("Error: OPENAI_API_KEY not set")
    sys.exit(1)
openai.api_key = OPENAI_KEY

# ----------------- 리트리버 -----------------
def retrieve_similar(text, top_k=5):
    emb = model.encode([text]).astype("float32")
    D, I = index.search(emb, top_k)
    results = []
    for idx in I[0]:
        results.append(meta[idx])
    return results

# ----------------- LLM 후보 추출 -----------------
def extract_candidates_llm(full_text, max_candidates=CANDIDATE_MAX):
    prompt = f"""
You are a Korean legal assistant.
From the following contract text, extract up to {max_candidates} clauses that may be unfair to consumers.
Return as valid JSON array with objects like: {{"clause": "..."}}
Make sure the output is valid JSON parsable by Python json.loads.

Contract text:
\"\"\"{full_text}\"\"\""""
    
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000
    )
    text = resp.choices[0].message.content.strip()

    # JSON 블록 제거 가능 (```json ... ```)
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()
    
    try:
        data = json.loads(text)
        return [item["clause"] for item in data if "clause" in item]
    except Exception as e:
        print("Error parsing JSON from LLM:", e)
        print("LLM raw response:", text)
        return []

# ----------------- LLM 최종 판단 -----------------
def ask_llm_for_risk_clause(clause_text, retrieved_examples):
    prompt = f"""
You are a legal assistant. Evaluate the risk that the following contract clause is an unfair term for consumers.

Clause:
\"\"\"{clause_text}\"\"\"

Relevant prior examples (short):
"""
    for ex in retrieved_examples:
        prompt += f"- From {ex['source_file']} (chunk {ex['chunk_idx']}): {ex['text'][:200].replace('\\n',' ')}...\n"

    prompt += """
Answer in valid JSON: {"risk_percent": int(0-100), "risk_level": "low/medium/high", "reason": "...", "suggested_law_codes": ["..."]}
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=400
    )
    text = resp.choices[0].message.content.strip()

    # JSON 확인용 (실패해도 그대로 리턴)
    try:
        json.loads(text)
    except Exception:
        print("Warning: LLM returned non-JSON. Raw:", text)
    return text

# ----------------- 메인 -----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_pipeline.py path/to/contract.txt")
        sys.exit(1)

    txt = open(sys.argv[1], encoding="utf-8").read()
    candidates = extract_candidates_llm(txt)

    if not candidates:
        print("No candidate clauses found.")
        sys.exit(0)

    for c in candidates:
        retrieved = retrieve_similar(c, top_k=5)
        llm_response = ask_llm_for_risk_clause(c, retrieved)
        print("CANDIDATE:", c[:200])
        print("LLM RESPONSE:", llm_response)
        print("-"*80)
