# scripts/extract_and_chunk.py
import os
import json
import uuid
import re
import fitz  # pymupdf
from tqdm import tqdm

# 데이터 루트 (상대경로)
DATA_ROOT = "../data"
# 처리할 하위 폴더들 (필요시 추가)
CONTRACTS_DIR = os.path.join(DATA_ROOT, "contracts", "reference")
LAW_DIR = os.path.join(DATA_ROOT, "law")
STANDARD_DIR = os.path.join(DATA_ROOT, "standard")

OUT_FILE = "../outputs/chunks.jsonl"
CHUNK_SIZE = 1200    # 문자 기준 (약 250~600 tokens)
CHUNK_OVERLAP = 200
MAX_PAGE_CHARS = 200_000   # 페이지당 최대 글자 수 제한 (필요시 조정)

# 조항(제n조) 기준 분할 정규식 (한글 법조문 제목 패턴)
ARTICLE_RE = re.compile(r'(제\s*\d+\s*조[^\n\r]*)', flags=re.MULTILINE)

def split_text_simple(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """기존 방식: 문자 기준으로 split (판결문 등)"""
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        new_start = end - overlap
        if new_start <= start:
            new_start = start + size
        start = new_start
    return chunks

def split_by_article(text, max_chunk_size=CHUNK_SIZE):
    """
    법령/표준문서용: '제n조' 단위로 분리.
    - 각 조항을 하나의 청크로 만듬.
    - 만약 조항 텍스트가 너무 길면 내부에서 부분 분할(part) 처리.
    """
    text = text.replace("\r\n", "\n")
    # 찾은 조항 제목의 시작 인덱스 목록
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        # 조항 패턴이 없으면 전체 텍스트를 simple split으로 처리
        return split_text_simple(text, size=max_chunk_size, overlap=CHUNK_OVERLAP)

    chunks = []
    for i, m in enumerate(matches):
        start_idx = m.start()
        # 다음 매치의 시작 또는 문서 끝까지가 이 조항의 범위
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(text)
        article_text = text[start_idx:end_idx].strip()
        # 조항 텍스트 전처리
        article_text = re.sub(r'\n{2,}', '\n', article_text).strip()

        # 만약 조항 하나가 너무 길면 내부 분할 (part)
        if len(article_text) <= max_chunk_size:
            chunks.append(article_text)
        else:
            # 분할하되 문맥 유지를 위해 오버랩 적용
            start = 0
            part_idx = 0
            while start < len(article_text):
                end = min(len(article_text), start + max_chunk_size)
                sub = article_text[start:end].strip()
                if sub:
                    # 파트 표시: 원래 조항 텍스트 앞부분을 붙여 id 매칭 용도로 유지
                    chunks.append(f"{sub}\n[PART {part_idx}]")
                part_idx += 1
                new_start = end - CHUNK_OVERLAP
                if new_start <= start:
                    new_start = start + max_chunk_size
                start = new_start
    return chunks

def process_pdf_file(path, source_tag):
    """
    source_tag : 'contract' | 'law' | 'standard'
    """
    doc = fitz.open(path)
    file_chunks = []
    fname = os.path.basename(path)
    for page_num, page in enumerate(doc, start=1):
        txt = page.get_text()
        if not txt:
            continue

        # 너무 큰 페이지는 truncate (또는 skip) 처리
        if len(txt) > MAX_PAGE_CHARS:
            print(f"⚠️  Truncating large page {page_num} in {fname} ({len(txt)} chars)")
            txt = txt[:MAX_PAGE_CHARS]

        if source_tag in ("law", "standard"):
            # 조항 단위 분리 (문서 전체 기준으로 분리해 조항 단위 보장)
            # 법/표준은 페이지 단위가 아닐 수 있으므로 전체 페이지 텍스트를 모아서 처리
            file_chunks.append(("PAGE_TEXT", page_num, txt))
        else:
            # 계약서/판결문 계열: 기존 문자 기준 청킹
            chunks = split_text_simple(txt)
            for i, c in enumerate(chunks):
                yield {
                    "chunk_id": str(uuid.uuid4()),
                    "source_file": fname,
                    "source_tag": source_tag,
                    "page": page_num,
                    "chunk_idx": i,
                    "text": c
                }

    # 법/표준 처리: 파일 전체를 합쳐서 조항 단위로 분리
    if source_tag in ("law", "standard") and file_chunks:
        # 합치기: 페이지 순으로
        full_text = "\n".join([p[2] for p in file_chunks])
        articles = split_by_article(full_text)
        for i, art in enumerate(articles):
            yield {
                "chunk_id": str(uuid.uuid4()),
                "source_file": fname,
                "source_tag": source_tag,
                "page": None,
                "chunk_idx": i,
                "text": art
            }

def main():
    os.makedirs("../outputs", exist_ok=True)
    out_path = OUT_FILE

    # gather files to process from the three locations (if exist)
    file_list = []
    if os.path.isdir(CONTRACTS_DIR):
        for f in os.listdir(CONTRACTS_DIR):
            if f.lower().endswith(".pdf"):
                file_list.append((os.path.join(CONTRACTS_DIR, f), "contract"))
    if os.path.isdir(LAW_DIR):
        for f in os.listdir(LAW_DIR):
            if f.lower().endswith(".pdf"):
                file_list.append((os.path.join(LAW_DIR, f), "law"))
    if os.path.isdir(STANDARD_DIR):
        for f in os.listdir(STANDARD_DIR):
            if f.lower().endswith(".pdf"):
                file_list.append((os.path.join(STANDARD_DIR, f), "standard"))

    if not file_list:
        print("No PDF files found in configured directories. Check DATA_ROOT and subfolders.")
        return

    with open(out_path, "w", encoding="utf-8") as outf:
        for path, tag in tqdm(file_list, desc="Processing PDFs"):
            try:
                for rec in process_pdf_file(path, tag):
                    outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print("Done. Chunks written to", out_path)

if __name__ == "__main__":
    main()
