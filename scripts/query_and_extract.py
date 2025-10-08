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
OPENAI_MAX_TOKENS = 1500
OPENAI_TEMPERATURE = 0.0

INDEX_FILE = os.path.join("..", "outputs", "faiss.index")
META_FILE = os.path.join("..", "outputs", "faiss_meta.pkl")
EMBED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # 공개 한국어 법률 SBERT

SIMILARITY_THRESHOLD = 0.70  # cosine similarity 기준

ARTICLE_RE = re.compile(r'(제\s*\d+\s*조[^\n\r]*)')

# ----------------- 조항 단위 분리 -----------------
def split_by_article(text):
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [{"article": None, "text": text.strip()}]

    results = []
    for i, m in enumerate(matches):
        start_idx = m.start()
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(text)
        article_title = m.group(1).strip()
        body = text[start_idx:end_idx].strip()
        results.append({"article": article_title, "text": body})
    return results

# ----------------- FAISS 로드 및 유사도 검색 -----------------
embed_model = SentenceTransformer(EMBED_MODEL)

def load_faiss():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError("FAISS index or meta not found. Run embed_and_index.py first.")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def search_standard_similarity(query_text, top_k=5):
    index, meta = load_faiss()
    q_emb = embed_model.encode([query_text]).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        rec = meta[idx]
        if rec.get("source_tag") != "standard":
            continue
        results.append({"text": rec["text"], "score": float(dist)})
    return results

# ----------------- LLM 후보 추출 -----------------
def call_openai_filter(chunks, api_key=None):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")
    openai.api_key = api_key

    assembled = []
    total_len = 0
    MAX_PROMPT_CHARS = 15000
    for i, c in enumerate(chunks):
        piece = f"[chunk{i}] ({c.get('article')}) {c.get('text')}"
        if total_len + len(piece) > MAX_PROMPT_CHARS:
            break
        assembled.append(piece)
        total_len += len(piece)

    system_msg = (
        "You are a Korean legal assistant. "
        "Input은 계약 조항 후보입니다.\n"
        "모든 조항 중 잠재적 위험이 있는 조항은 최대한 포함하세요. "
        "공정한 조항이라도 위험 가능성이 조금이라도 있으면 후보로 남겨야 합니다.\n"
        "Return ONLY valid JSON array. Each element must have:\n"
        " - article: related article title\n"
        " - excerpt: short excerpt (<=300 chars)\n"
        " - reason: why it may be unfair\n"
    )

    user_msg = "Analyze the following clauses. Include all clauses with any potential risk, even if minor:\n\n" + "\n\n---\n\n".join(assembled)

    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS
    )
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\[.*\])", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {"error": "failed_to_parse_json", "raw": text}
        return {"error": "no_json_found", "raw": text}

# ----------------- 메인 분석 -----------------
def analyze_contract(text, openai_call=True, save_path=os.path.join("..", "outputs", "query_results.json")):
    clauses = split_by_article(text)
    filtered = []
    for c in clauses:
        sims = search_standard_similarity(c["text"], top_k=3)
        if sims and max([s["score"] for s in sims]) >= SIMILARITY_THRESHOLD:
            continue
        filtered.append(c)

    llm_result = None
    llm_filtered_count = 0
    if openai_call and filtered:
        llm_result = call_openai_filter(filtered)
        if isinstance(llm_result, list):
            llm_filtered_count = len(filtered) - len(llm_result)

    result = {
        "timestamp": datetime.now().isoformat(),
        "total_articles": len(clauses),
        "after_standard_filter": len(filtered),
        "llm_filtered_count": llm_filtered_count,
        "final_candidates": llm_result
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("==== 분석 결과 ====")
    print(f"총 조항 수: {len(clauses)}")
    print(f"표준약관 필터 이후 남은 조항: {len(filtered)}")
    print(f"LLM으로 제외된 조항 수: {llm_filtered_count}")
    print("잠재적 위험 조항:")
    if isinstance(llm_result, list):
        for r in llm_result:
            print(f"- {r.get('article')}: {r.get('excerpt')} ({r.get('reason')})")
    else:
        print(llm_result)

    return result

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="분석할 계약서(텍스트) 파일 경로", required=True)
    parser.add_argument("--openai", action="store_true", help="OpenAI 호출 여부")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    analyze_contract(text, openai_call=args.openai)
