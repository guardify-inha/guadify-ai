"""
query_and_extract.py (Updated)
- Acts as a pre-processor to extract candidate clauses for GNN analysis.
- Step 1: Splits user's contract into articles/clauses.
- Step 2: Filters out clauses that are highly similar to known 'standard' clauses using FAISS.
- Step 3 (Optional): Uses an LLM not to find unfair clauses, but to *prune* obviously safe/standard clauses from the remaining candidates.
- Output: A simple list of candidate clauses (`candidate_clauses.jsonl`) to be fed into the GNN inference script.
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
OPENAI_MAX_TOKENS = 1500
OPENAI_TEMPERATURE = 0.0

# 경로 설정
OUT_DIR = os.path.join("..", "outputs")
INDEX_FILE = os.path.join(OUT_DIR, "faiss.index")
META_FILE = os.path.join(OUT_DIR, "faiss_meta.pkl")
CANDIDATES_FILE = os.path.join(OUT_DIR, "candidate_clauses.jsonl") # 최종 후보 저장 파일

# 임베딩 모델
EMBED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
SIMILARITY_THRESHOLD = 0.75 # 표준 약관과의 유사도 기준 (조금 더 엄격하게 설정 가능)

# 정규 표현식
ARTICLE_RE = re.compile(r'(제\s*\d+\s*조[^\n\r]*)')

# ----------------- 헬퍼 함수 (기존과 거의 동일) -----------------

def split_by_article(text):
    """조항 단위로 텍스트를 분리합니다."""
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

# FAISS 관련 객체는 한 번만 로드하도록 전역으로 관리
print("Loading embedding model and FAISS index...")
embed_model = SentenceTransformer(EMBED_MODEL)
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        faiss_meta = pickle.load(f)
else:
    raise FileNotFoundError("FAISS index/meta not found. Run embed_and_index.py first.")

def search_standard_similarity(query_text, top_k=1):
    """FAISS를 사용해 표준 약관과의 최고 유사도를 검색합니다."""
    q_emb = embed_model.encode([query_text], convert_to_tensor=False).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, indices = faiss_index.search(q_emb, top_k)
    
    max_sim = 0.0
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0: continue
        rec = faiss_meta[idx]
        # 'standard' 태그를 가진 노드와의 유사도만 확인
        if rec.get("source_tag") == "standard":
            max_sim = max(max_sim, float(dist))
    return max_sim

# ----------------- 🎯 LLM 필터 (역할 변경) -----------------

def call_llm_to_prune_clauses(chunks, api_key=None):
    """
    LLM을 호출하여 명백히 안전하거나 표준적인 조항을 식별합니다.
    불공정성을 찾는게 아니라, '더 이상 분석할 필요가 없는' 조항의 인덱스를 반환합니다.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")
    openai.api_key = api_key

    assembled = [f"[chunk{i}] {c.get('text')}" for i, c in enumerate(chunks)]
    
    system_msg = (
        "You are a meticulous Korean legal assistant. Your task is to identify contract clauses "
        "that are **absolutely standard, common, and have no potential for unfairness or ambiguity**."
        "You are a filter to remove obviously safe clauses before a more detailed analysis."
        "Return ONLY a valid JSON object with a single key 'safe_chunk_indices', "
        "containing a list of integer indices for the chunks that are safe to ignore."
        "\nExample: {\"safe_chunk_indices\": [2, 5, 8]}"
        "\nIf no clauses are clearly safe, return an empty list: {\"safe_chunk_indices\": []}"
    )

    user_msg = "Please identify the safe clauses from the following list:\n\n" + "\n".join(assembled)
    
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            response_format={"type": "json_object"} # JSON 출력 모드 활용
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        safe_indices = data.get("safe_chunk_indices", [])
        if isinstance(safe_indices, list):
            return safe_indices
        return []
    except Exception as e:
        print(f"Error calling LLM or parsing response: {e}")
        return [] # 오류 발생 시, 안전한 조항이 없는 것으로 간주

# ----------------- 🚀 메인 분석 파이프라인 -----------------

def extract_candidate_clauses(text, use_llm_filter=True, save_path=CANDIDATES_FILE):
    """GNN 분석을 위한 최종 후보 조항 목록을 추출합니다."""
    print("🚀 Starting clause extraction pipeline...")
    
    # 1. 텍스트를 조항 단위로 분리
    clauses = split_by_article(text)
    print(f"   - Step 1: Split into {len(clauses)} clauses.")

    # 2. 표준 약관과 유사한 조항 필터링 (FAISS)
    candidates_after_faiss = []
    for c in clauses:
        max_sim = search_standard_similarity(c["text"])
        if max_sim >= SIMILARITY_THRESHOLD:
            continue
        candidates_after_faiss.append(c)
    print(f"   - Step 2: {len(candidates_after_faiss)} clauses remaining after FAISS similarity filter.")

    # 3. 명백히 안전한 조항 필터링 (LLM)
    if use_llm_filter and candidates_after_faiss:
        print("   - Step 3: Using LLM to prune obviously safe clauses...")
        safe_indices = call_llm_to_prune_clauses(candidates_after_faiss)
        print(f"     - LLM identified {len(safe_indices)} clauses as safe.")
        final_candidates = [
            c for i, c in enumerate(candidates_after_faiss) if i not in safe_indices
        ]
    else:
        final_candidates = candidates_after_faiss
    
    print(f"   - Final: {len(final_candidates)} candidate clauses remain for GNN analysis.")
    
    # 4. 최종 후보 목록을 파일로 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for clause in final_candidates:
                f.write(json.dumps(clause, ensure_ascii=False) + "\n")
        print(f"✅ Pipeline finished. Candidates saved to {save_path}")

    return final_candidates

# ----------------- CLI 실행 -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract candidate clauses from a contract for GNN analysis.")
    parser.add_argument("--file", help="Path to the contract text file to analyze.", required=True)
    parser.add_argument("--no-llm", action="store_true", help="Disable the LLM filtering step.")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        contract_text = f.read()
    
    extract_candidate_clauses(contract_text, use_llm_filter=not args.no_llm)