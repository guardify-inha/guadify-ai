"""
query_pipeline.py
- GNN preds와 LLM 평가를 결합
- retrieve_similar / ask_llm_for_risk_clause 정의
- JSON/summary 저장
"""

import os
import sys
import json
import random

# -------------------------------
# 1. 인자 체크
# -------------------------------
if len(sys.argv) < 3:
    print("Usage: python query_pipeline.py path/to/query_results.json path/to/gnn_preds.json")
    sys.exit(1)

query_json_path = sys.argv[1]
gnn_pred_path = sys.argv[2]

# -------------------------------
# 2. 입력 파일 로드
# -------------------------------
results = json.load(open(query_json_path, encoding="utf-8"))
gnn_preds = json.load(open(gnn_pred_path, encoding="utf-8"))

final_candidates = results.get("final_candidates", [])
if not final_candidates:
    print("No candidate clauses found.")
    sys.exit(0)

# -------------------------------
# 3. Mock / 단순 함수 정의
# -------------------------------

def retrieve_similar(clause_text, top_k=5):
    """
    테스트용: 유사 조항 검색 시뮬레이션
    - 실제 구현 시 FAISS나 벡터 검색 사용
    """
    return [f"Similar clause {i}" for i in range(top_k)]

def ask_llm_for_risk_clause(clause_text, retrieved, REFERENCE_EXAMPLES=None):
    """
    테스트용: LLM 리스크 평가 시뮬레이션
    - 실제 구현 시 OpenAI API 또는 LLM 호출
    """
    risk_percent = random.randint(0, 100)
    reason = f"Mock reason: {len(clause_text)} chars, {len(retrieved)} retrieved"
    return {"risk_percent": risk_percent, "reason": reason}

# -------------------------------
# 4. 가중치
# -------------------------------
ALPHA = 0.7
BETA = 0.3

# -------------------------------
# 5. 결과 처리
# -------------------------------
all_results = []
summary_lines = []

for idx, c in enumerate(final_candidates):
    clause_text = c.get("excerpt") or c.get("text") or str(c)
    clause_identifier = c.get("law_article") or c.get("article") or "Unknown Article"

    # (1) 유사 조항 검색
    retrieved = retrieve_similar(clause_text)

    # (2) LLM 평가
    llm_response = ask_llm_for_risk_clause(clause_text, retrieved)

    # (3) GNN 위험도
    gnn_prob = None
    cid = f"chunk_{idx}"  # 테스트용: 순서 기반 chunk_id
    if cid in gnn_preds:
        gnn_prob = float(gnn_preds[cid])

    # (4) 최종 점수 결합
    llm_score = llm_response.get("risk_percent", 0) / 100.0
    final_score = llm_score if gnn_prob is None else ALPHA * gnn_prob + BETA * llm_score

    # 결과 기록
    record = {
        "article": clause_identifier,
        "clause_text": clause_text,
        "llm_eval": llm_response,
        "gnn_prob": gnn_prob,
        "final_score": round(float(final_score), 4),
    }
    all_results.append(record)

    summary_lines.append(
        f"조항: {clause_identifier}\n"
        f"문장 원문:\n{clause_text}\n"
        f"최종 스코어: {round(100 * final_score)}%\n"
        f"LLM 리스크: {llm_response.get('risk_percent', 0)}%\n"
        f"GNN 확률: {gnn_prob}\n"
        f"설명: {llm_response.get('reason', '')}\n"
        + "-"*80 + "\n"
    )

# -------------------------------
# 6. 결과 저장
# -------------------------------
os.makedirs(os.path.join("..", "outputs"), exist_ok=True)
with open(os.path.join("..", "outputs", "all_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
with open(os.path.join("..", "outputs", "summary.txt"), "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

print("Evaluation done. JSON saved to outputs/all_results.json, summary saved to outputs/summary.txt")
