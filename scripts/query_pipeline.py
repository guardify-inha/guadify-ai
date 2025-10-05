# scripts/query_pipeline.py
import os, json, pickle, sys
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import openai

# ----------------- 설정 -----------------
INDEX_FILE = "../outputs/faiss.index"
META_FILE = "../outputs/faiss_meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
CANDIDATE_MAX = 20

LAW_DIR = "../law"
TEMP_DIR = "../temp"
REFERENCE_FILE = "../data/outputs/chunks.jsonl"  # <- 변경

# ----------------- FAISS & 모델 로드 -----------------
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)
model = SentenceTransformer(EMBED_MODEL)

# ----------------- OpenAI -----------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("Error: OPENAI_API_KEY not set")
    sys.exit(1)
openai.api_key = OPENAI_KEY

# ----------------- reference 예시 로드 (chunks.jsonl) -----------------
def load_reference_examples():
    examples = []
    if not os.path.exists(REFERENCE_FILE):
        print(f"Warning: {REFERENCE_FILE} not found")
        return examples
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("source_tag") == "contract":
                examples.append({
                    "source_file": obj.get("source_file", "unknown"),
                    "chunk_idx": idx,
                    "text": obj.get("text", "")
                })
    return examples

REFERENCE_EXAMPLES = load_reference_examples()

# ----------------- 유사도 검색 -----------------
def retrieve_similar(text, top_k=5):
    emb = model.encode([text]).astype("float32")
    D, I = index.search(emb, top_k)
    results = []
    for idx in I[0]:
        results.append(meta[idx])
    return results

# ----------------- 후보 조항 추출 -----------------
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
    # 코드 블록 제거
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    try:
        data = json.loads(text)
        return [item["clause"] for item in data if "clause" in item]
    except Exception as e:
        print("Error parsing JSON from LLM:", e)
        print("LLM raw response:", text)
        return []

# ----------------- 최종 위험 판단 -----------------
def ask_llm_for_risk_clause(clause_text, retrieved_examples, reference_examples):
    prompt = f"""
You are a legal assistant. Evaluate the risk that the following contract clause is an unfair term.

Clause:
\"\"\"{clause_text}\"\"\" 

Consider the following:
1. Prior law/guidelines in order of priority: 
   - 약관심사지침.pdf
   - Other laws in {LAW_DIR}
   - 시행령 in {TEMP_DIR} (today focus on 금융)
2. Previously judged unfair examples from reference folder:
"""
    for ex in reference_examples:
        if ex['text'] in clause_text:
            prompt += f"- Match with {ex['source_file']} line {ex['chunk_idx']}: {ex['text'][:150]}...\n"

    prompt += """
Answer in valid JSON with:
{
  "risk_percent": int(0-100),
  "risk_level": "low/medium/high",
  "reason": "...",
  "suggested_law_codes": ["..."]
}
Use only the provided data. Do not hallucinate.
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=400
    )
    text = resp.choices[0].message.content.strip()
    # 코드 블록 제거
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        print("Warning: LLM returned non-JSON. Raw:", text)
        return {"risk_percent":0, "risk_level":"low", "reason":"failed parsing","suggested_law_codes":[]}

# ----------------- 메인 -----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_pipeline.py path/to/query_results.json")
        sys.exit(1)

    query_results_path = sys.argv[1]
    results = json.load(open(query_results_path, encoding="utf-8"))
    final_candidates = results.get("final_candidates", [])

    if not final_candidates:
        print("No candidate clauses found.")
        sys.exit(0)

    all_results = []
    summary_lines = []

    for c in final_candidates:
        clause_text = c.get("excerpt") or c.get("text") or c
        clause_identifier = c.get("law_article") or c.get("article") or "Unknown Article"
        retrieved = retrieve_similar(clause_text, top_k=5)
        llm_response = ask_llm_for_risk_clause(clause_text, retrieved, REFERENCE_EXAMPLES)
        
        # 1) JSON 형태로 저장
        record = {
            "article": clause_identifier,
            "clause_text": clause_text,
            "risk_eval": llm_response
        }
        all_results.append(record)

        # 2) 사람이 읽기 좋은 텍스트 요약
        summary_lines.append(
            f"조항: {clause_identifier}\n"
            f"문장 원문:\n{clause_text}\n"
            f"리스크: {llm_response.get('risk_percent', 0)}% / {llm_response.get('risk_level','low')}\n"
            f"참고 법/판례: {', '.join(llm_response.get('suggested_law_codes', []))}\n"
            f"설명: {llm_response.get('reason','')}\n"
            f"{'-'*80}\n"
        )

    # query_results.json과 같은 폴더에 저장
    result_dir = os.path.dirname(query_results_path)
    
    # 전체 JSON 저장
    with open(os.path.join(result_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 사람이 읽기 좋은 요약 저장
    with open(os.path.join(result_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    print(f"📁 결과 저장 위치: {result_dir}")
    print("Evaluation done. JSON saved to all_results.json, summary saved to summary.txt")
