# File: scripts/extract_and_chunk.py
"""
Enhanced chunk extractor + GPT structuring (batch)
- reference → clause 단위로 쪼갠 후 MAX_CHARS_FOR_GPT 단위로 묶어서 GPT 요청
- law → 문장 단위
- standard → 제n조 단위
Outputs:
  1. ../outputs/chunks.jsonl
  2. ../outputs/structured_reference.jsonl
"""

import os
import json
import uuid
import re
import time
import fitz  # PyMuPDF
from tqdm import tqdm
import kss
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- 초기 설정 --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_ROOT = os.path.join("..", "data")
CONTRACTS_REF_DIR = os.path.join(DATA_ROOT, "contracts", "reference")
LAW_DIR = os.path.join(DATA_ROOT, "law")
STANDARD_DIR = os.path.join(DATA_ROOT, "standard")

OUT_DIR = os.path.join("..", "outputs")
OUT_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
STRUCTURED_FILE = os.path.join(OUT_DIR, "structured_reference.jsonl")

CLAUSE_START_RE = re.compile(r"^\s*(\d+\)|[가-힣]\.)\s*", re.MULTILINE)
ARTICLE_RE = re.compile(r"(제\s*\d+\s*조[^\n\r]*)", flags=re.MULTILINE)

MAX_CHARS_FOR_GPT = 3000  # GPT 호출 글자 단위

# -------------------- 유틸 --------------------
def split_sentences_kss(text):
    try:
        return [s.strip() for s in kss.split_sentences(text) if s.strip()]
    except Exception:
        return [text]

def split_by_clause(text):
    text = text.replace("\r\n", "\n")
    matches = list(CLAUSE_START_RE.finditer(text))
    if not matches:
        return [text]
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def split_by_article_with_titles(text):
    text = text.replace("\r\n", "\n")
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [{"title": None, "body": text}]
    results = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start_idx = m.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start_idx:end_idx].strip()
        results.append({"title": title, "body": body})
    return results

# -------------------- GPT 구조화 --------------------
def call_gpt_structure(text, max_retries=3):
    text_to_send = text[:MAX_CHARS_FOR_GPT]
    prompt = f"""
다음은 공정거래위원회의 약관 시정 보도자료 일부입니다.
아래 텍스트에서 불공정조항, 시정이유, 관련법조항, 출처를 JSON으로 구조화하세요.

출력 형식 (반드시 이 형태로만):
{{
  "불공정조항": "...",
  "시정이유": "...",
  "관련법조항": "...",
  "출처": "..."
}}

텍스트:
{text_to_send}
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty GPT response")
            content = re.sub(r"^```json|```$", "", content.strip())
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ GPT 구조화 실패 (시도 {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    return None

# -------------------- PDF 처리 --------------------
def process_pdf_file(path, source_tag, structured_out=None):
    doc = fitz.open(path)
    fname = os.path.basename(path)
    chunk_idx = 0

    if source_tag == "reference":
        full_text = "\n".join([p.get_text() for p in doc if p.get_text()])
        clauses = split_by_clause(full_text)
        batch = ""
        for clause in clauses:
            if len(batch + clause) > MAX_CHARS_FOR_GPT:
                chunk_id = str(uuid.uuid4())
                rec = {
                    "chunk_id": chunk_id,
                    "source_file": fname,
                    "source_tag": source_tag,
                    "level": "clause_batch",
                    "chunk_idx": chunk_idx,
                    "text": batch,
                }
                chunk_idx += 1
                yield rec
                if structured_out:
                    structured = call_gpt_structure(batch)
                    if structured:
                        structured["chunk_id"] = chunk_id
                        structured["source_file"] = fname
                        structured_out.write(json.dumps(structured, ensure_ascii=False) + "\n")
                batch = ""
            batch += clause + "\n"
        if batch:
            chunk_id = str(uuid.uuid4())
            rec = {
                "chunk_id": chunk_id,
                "source_file": fname,
                "source_tag": source_tag,
                "level": "clause_batch",
                "chunk_idx": chunk_idx,
                "text": batch,
            }
            yield rec
            if structured_out:
                structured = call_gpt_structure(batch)
                if structured:
                    structured["chunk_id"] = chunk_id
                    structured["source_file"] = fname
                    structured_out.write(json.dumps(structured, ensure_ascii=False) + "\n")
            chunk_idx += 1

    elif source_tag == "law":
        full_text = "\n".join([p.get_text() for p in doc if p.get_text()])
        for art in split_by_article_with_titles(full_text):
            title = art["title"]
            for s in split_sentences_kss(art["body"]):
                yield {
                    "chunk_id": str(uuid.uuid4()),
                    "source_file": fname,
                    "source_tag": source_tag,
                    "level": "law_sentence",
                    "article_title": title,
                    "text": s,
                }

    elif source_tag == "standard":
        full_text = "\n".join([p.get_text() for p in doc if p.get_text()])
        for art in split_by_article_with_titles(full_text):
            yield {
                "chunk_id": str(uuid.uuid4()),
                "source_file": fname,
                "source_tag": source_tag,
                "level": "standard_article",
                "article_title": art["title"],
                "text": art["body"],
            }

# -------------------- 메인 --------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    file_list = []

    for d, tag in [
        (CONTRACTS_REF_DIR, "reference"),
        (LAW_DIR, "law"),
        (STANDARD_DIR, "standard"),
    ]:
        if os.path.isdir(d):
            file_list += [
                (os.path.join(d, f), tag)
                for f in os.listdir(d)
                if f.lower().endswith(".pdf")
            ]

    if not file_list:
        print("❌ No PDF files found.")
        return

    with open(OUT_FILE, "w", encoding="utf-8") as outf, \
         open(STRUCTURED_FILE, "w", encoding="utf-8") as structured_out:
        for path, tag in tqdm(file_list, desc="Processing PDFs"):
            try:
                for rec in process_pdf_file(path, tag, structured_out):
                    outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")

    print(f"✅ Done. Chunks → {OUT_FILE}")
    print(f"✅ Structured reference → {STRUCTURED_FILE}")

if __name__ == "__main__":
    main()
