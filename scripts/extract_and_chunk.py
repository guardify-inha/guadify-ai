# File: scripts/extract_and_chunk.py
"""
Enhanced chunk extractor
- reference → CSV 기반 직접 구조화
- law → 문장 단위
- standard → 제n조 단위
- Mecab backend 강제 사용
- 이미 처리된 파일은 건너뜀
Outputs:
  1. ../outputs/chunks.jsonl
  2. ../outputs/structured_reference.jsonl
"""

import os
import json
import uuid
import re
import csv
import fitz  # PyMuPDF
from tqdm import tqdm
import kss


# -------------------- 경로 설정 --------------------
DATA_ROOT = os.path.join("..", "data")
CONTRACTS_REF_DIR = os.path.join(DATA_ROOT, "contracts", "reference")
LAW_DIR = os.path.join(DATA_ROOT, "law")
STANDARD_DIR = os.path.join(DATA_ROOT, "standard")

OUT_DIR = os.path.join("..", "outputs")
OUT_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
STRUCTURED_FILE = os.path.join(OUT_DIR, "structured_reference.jsonl")

CLAUSE_START_RE = re.compile(r"^\s*(\d+\)|[가-힣]\.)\s*", re.MULTILINE)
ARTICLE_RE = re.compile(r"(제\s*\d+\s*조[^\n\r]*)", flags=re.MULTILINE)

# -------------------- 유틸 --------------------
def split_sentences_kss(text):
    try:
        # Mecab backend 사용
        return [s.strip() for s in kss.split_sentences(text, backend="mecab") if s.strip()]
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

# -------------------- CSV 기반 Reference 처리 --------------------
def process_reference_csv(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 필드: ID, 파일명, 대주제, 소주제, 불공정약관원문, 시정요청사유, 근거조항, 수정후약관조항
            rec = {
                "chunk_id": str(uuid.uuid4()),
                "source_file": row.get("파일명"),
                "source_tag": "reference",
                "level": "reference_clause",
                "ID": row.get("ID"),
                "대주제": row.get("대주제"),
                "소주제": row.get("소주제"),
                "불공정약관원문": row.get("불공정약관원문"),
                "시정요청사유": row.get("시정요청사유"),
                "근거조항": row.get("근거조항"),
                "수정후약관조항": row.get("수정후약관조항"),
            }
            yield rec

# -------------------- PDF 처리 --------------------
def process_pdf_file(path, source_tag):
    doc = fitz.open(path)
    fname = os.path.basename(path)

    full_text = "\n".join([p.get_text() for p in doc if p.get_text()])
    if not full_text.strip():
        print(f"⚠️ {fname}: PDF에서 텍스트를 추출하지 못했습니다.")
        return

    if source_tag == "law":
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
        arts = split_by_article_with_titles(full_text)
        print(f"{fname}: found {len(arts)} standard articles")
        for art in arts:
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

    # 이미 처리된 파일 확인
    processed_files = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_files.add(obj.get("source_file"))
                except:
                    continue

    # CSV reference 처리
    for f in os.listdir(CONTRACTS_REF_DIR):
        if f.lower().endswith(".csv") and f not in processed_files:
            file_list.append((os.path.join(CONTRACTS_REF_DIR, f), "reference"))

    # PDF law / standard 처리
    for d, tag in [(LAW_DIR, "law"), (STANDARD_DIR, "standard")]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".pdf") and f not in processed_files:
                    file_list.append((os.path.join(d, f), tag))

    if not file_list:
        print("❌ 처리할 파일이 없습니다. 이미 모두 완료됨.")
        return

    with open(OUT_FILE, "a", encoding="utf-8") as outf, \
         open(STRUCTURED_FILE, "a", encoding="utf-8") as structured_out:
        for path, tag in tqdm(file_list, desc="Processing Files"):
            try:
                if tag == "reference":
                    for rec in process_reference_csv(path):
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        structured_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    for rec in process_pdf_file(path, tag):
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n"
                        )
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")

    print(f"✅ Done. Chunks → {OUT_FILE}")
    print(f"✅ Structured reference → {STRUCTURED_FILE}")

if __name__ == "__main__":
    main()
