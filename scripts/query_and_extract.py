# scripts/query_and_extract.py
import os, re, json, argparse, pickle
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ----------------- 설정 -----------------
OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 1500
OPENAI_TEMPERATURE = 0.0

INDEX_FILE = "../outputs/faiss.index"
META_FILE = "../outputs/faiss_meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

MAX_PROMPT_CHARS = 15000
SIMILARITY_THRESHOLD = 0.4   # 표준약관과 "비슷하다"고 보는 기준 (거리 기준, 조정 필요)

ARTICLE_RE = re.compile(r'(제\s*\d+\s*조[^\n]*)')

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

# ----------------- 벡터 검색 -----------------
embed_model = SentenceTransformer(EMBED_MODEL)

def load_faiss():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def search_standard_similarity(query_text, top_k=1):
    """표준약관(source_tag == 'standard') 중 유사한 것 찾기"""
    index, meta = load_faiss()
    q_emb = embed_model.encode([query_text]).astype("float32")
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

# ----------------- LLM 호출 -----------------
def call_openai_filter(chunks, api_key=None):
    """청크 리스트를 받아 LLM에게 잠재적 위험 후보를 최대한 반환"""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")

    client = openai.OpenAI(api_key=api_key)

    assembled = []
    total_len = 0
    for i, c in enumerate(chunks):
        piece = f"[chunk{i}] ({c['article']}) {c['text']}"
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

    resp = client.chat.completions.create(
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

# ----------------- 메인 로직 -----------------
def analyze_contract(text, openai_call=True, output_dir="../outputs"):
    # 1. 계약서에서 조항 단위 분리
    clauses = split_by_article(text)

    # 2. 표준약관과 유사한 건 제외
    filtered = []
    for c in clauses:
        sims = search_standard_similarity(c["text"], top_k=1)
        if sims and sims[0]["score"] < SIMILARITY_THRESHOLD:
            continue
        filtered.append(c)

    # 3. LLM으로 잠재적 위험 후보 최대한 추출
    llm_result = None
    llm_filtered_count = 0
    if openai_call and filtered:
        llm_result = call_openai_filter(filtered)
        if isinstance(llm_result, list):
            llm_filtered_count = len(filtered) - len(llm_result)

    # 타임스탬프 폴더 생성 (초단위까지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"analysis_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_articles": len(clauses),
        "after_standard_filter": len(filtered),
        "llm_filtered_count": llm_filtered_count,
        "final_candidates": llm_result
    }

    # 저장
    save_path = os.path.join(result_dir, "query_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"📁 결과 저장 위치: {result_dir}")

    # 보기 좋게 출력
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

# ----------------- 실행용 CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="분석할 계약서(텍스트) 파일 경로", required=True)
    parser.add_argument("--openai", action="store_true", help="OpenAI 호출 여부")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    analyze_contract(text, openai_call=args.openai)
