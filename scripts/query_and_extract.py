# scripts/query_and_extract.py
import os
import re
import json
import openai  # pip install openai>=1.0.0

# OpenAI 설정
OPENAI_MODEL = "gpt-4o"   # 필요한 모델로 교체 가능
OPENAI_MAX_TOKENS = 1500
OPENAI_TEMPERATURE = 0.0

# 키워드 패턴 (선택적으로 사용 가능)
KEYWORDS = re.compile(r"(환불|환급|중도해지|위약금|손해배상|원상회복|승인취소|할부|지체|지연|면책|환불기한)")

# 안전 장치: 한 번에 LLM에 넣을 최대 문자 길이
MAX_PROMPT_CHARS = 15000

# ----------------- 유틸 함수 -----------------
def extract_candidates_by_keyword(text):
    """키워드가 포함된 문장만 골라냄 (빠른 후보 필터링용, 옵션)"""
    sents = re.split(r'(?<=[\.\?\!\n])\s+', text)
    candidates = [s.strip() for s in sents if s and KEYWORDS.search(s)]
    return candidates


def call_openai_extract(chunks, api_key=None, model=OPENAI_MODEL,
                        max_tokens=OPENAI_MAX_TOKENS, temp=OPENAI_TEMPERATURE):
    """청크 리스트를 받아 LLM에게 불공정 조항 뽑아달라고 요청"""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")

    client = openai.OpenAI(api_key=api_key)

    # 전체 텍스트를 LLM에 넘기기 전에 길이 제한 적용
    assembled = []
    total_len = 0
    for i, c in enumerate(chunks):
        piece = f"[chunk{i}] {c}"
        if total_len + len(piece) > MAX_PROMPT_CHARS:
            break
        assembled.append(piece)
        total_len += len(piece)

    system_msg = (
        "You are a Korean legal assistant that finds potentially unfair or consumer-unfriendly contract clauses.\n"
        "Return ONLY valid JSON array. Each element must have:\n"
        " - excerpt: short excerpt (<=300 chars)\n"
        " - reason: 1-line explanation why suspicious\n"
        " - suggested_keywords: list of 1~4 keywords (strings)\n"
        " - source: chunk index or note\n"
        "Maximum 10 items."
    )

    user_msg = "Analyze the following contract text (Korean). Identify up to 10 clauses that may be unfair to consumers.\n\n" + "\n\n---\n\n".join(assembled)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=temp,
        max_tokens=max_tokens
    )
    text = resp.choices[0].message.content.strip()

    # JSON 파싱
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


# ----------------- 메인 함수 -----------------
def analyze_text_with_llm(text, use_keyword=False, openai_call=True):
    chunks = []
    if use_keyword:
        chunks = extract_candidates_by_keyword(text)
    else:
        chunks = re.split(r'(?<=[\.\?\!\n])\s+', text)

    llm_result = None
    if openai_call and chunks:
        print(f"Calling OpenAI on {len(chunks)} chunks (truncated if too large)...")
        llm_result = call_openai_extract(chunks)

    return {
        "chunks_count": len(chunks),
        "llm_result": llm_result
    }


# ----------------- 실행용 CLI -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="분석할 계약서(텍스트) 파일 경로", required=True)
    parser.add_argument("--openai", action="store_true", help="OpenAI 호출 여부")
    parser.add_argument("--keyword", action="store_true", help="키워드 필터링 사용 여부")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    out = analyze_text_with_llm(text, use_keyword=args.keyword, openai_call=args.openai)
    print(json.dumps(out, ensure_ascii=False, indent=2))
