"""
Query pipeline with GNN risk score:
- Load LLM + GNN results
- Retrieve similar clauses
- Combine risk scores
- Save JSON and summary
"""

import os
import sys
import json

# ⚠️ 주의: 아래 함수/변수는 외부에서 정의되어 있어야 함
# - retrieve_similar(clause_text, top_k=5)
# - ask_llm_for_risk_clause(clause_text, retrieved, REFERENCE_EXAMPLES)
# - REFERENCE_EXAMPLES, model, index, meta, gnn_preds, ALPHA, BETA
#   (gnn_preds는 {chunk_id: 확률} dict, ALPHA/BETA는 가중치)

if len(sys.argv) < 2:
    print("Usage: python query_pipeline.py path/to/query_results.json")
    sys.exit(1)

# 1. 입력 JSON 로드
results = json.load(open(sys.argv[1], encoding="utf-8"))
final_candidates = results.get("final_candidates", [])

if not final_candidates:
    print("No candidate clauses found.")
    sys.exit(0)

all_results = []
summary_lines = []

# 2. 후보 조항 처리
for c in final_candidates:
    clause_text = c.get("excerpt") or c.get("text") or str(c)
    clause_identifier = c.get("law_article") or c.get("article") or "Unknown Article"

    # (1) 유사 조항 검색
    retrieved = retrieve_similar(clause_text, top_k=5)

    # (2) LLM 리스크 판별
    llm_response = ask_llm_for_risk_clause(clause_text, retrieved, REFERENCE_EXAMPLES)

    # (3) GNN 위험도 찾기 (FAISS recent chunk 기준)
    q_emb = model.encode([clause_text]).astype("float32")
    Dq, Iq = index.search(q_emb, 1)
    gnn_prob = None
    if Iq[0][0] >= 0:
        nearest_meta = meta[Iq[0][0]]
        cid = nearest_meta.get("chunk_id")
        if cid and cid in gnn_preds:
            gnn_prob = float(gnn_preds[cid])

    # (4) 점수 결합
    llm_score = llm_response.get("risk_percent", 0) / 100.0
    if gnn_prob is None:
        final_score = llm_score
    else:
        final_score = ALPHA * gnn_prob + BETA * llm_score

    record = {
        "article": clause_identifier,
        "clause_text": clause_text,
        "llm_eval": llm_response,    # LLM 결과(reason, risk_percent 등)
        "gnn_prob": gnn_prob,        # GNN 예측 위험도 (0~1)
        "final_score": round(float(final_score), 4),  # 결합 점수
    }
    all_results.append(record)

    summary_lines.append(
        f"조항: {clause_identifier}\n"
        f"문장 원문:\n{clause_text}\n"
        f"최종 스코어: {round(100 * final_score)}%\n"
        f"LLM 리스크: {llm_response.get('risk_percent', 0)}%\n"
        f"GNN 확률: {gnn_prob}\n"
        f"설명: {llm_response.get('reason', '')}\n"
        f"{'-' * 80}\n"
    )

# 5. 결과 저장
os.makedirs(os.path.join("..", "outputs"), exist_ok=True)

with open(os.path.join("..", "outputs", "all_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

with open(os.path.join("..", "outputs", "summary.txt"), "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

print("Evaluation done. JSON saved to outputs/all_results.json, summary saved to outputs/summary.txt")
