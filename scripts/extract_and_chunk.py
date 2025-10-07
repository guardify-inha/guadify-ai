# File: scripts/extract_and_chunk.py
"""
Improved chunk extractor that preserves hierarchy levels and records metadata required for GNN nodes.
Now uses sentence-level splitting for finer-grained nodes.
Produces ../outputs/chunks.jsonl
"""
import os
import json
import uuid
import re
import fitz  # pymupdf
from tqdm import tqdm

# 데이터 루트 (상대경로)
DATA_ROOT = os.path.join("..", "data")
# 처리할 하위 폴더들
CONTRACTS_REF_DIR = os.path.join(DATA_ROOT, "contracts", "reference")
LAW_DIR = os.path.join(DATA_ROOT, "law")
STANDARD_DIR = os.path.join(DATA_ROOT, "standard")

OUT_FILE = os.path.join("..", "outputs", "chunks.jsonl")
CHUNK_SIZE = 1200  # 문자 기준 (보조)
CHUNK_OVERLAP = 200
MAX_PAGE_CHARS = 200_000

# 조항(제n조) 기준 분할 정규식 (한글 법조문 제목 패턴)
ARTICLE_RE = re.compile(r'(제\s*\d+\s*조[^\n\r]*)', flags=re.MULTILINE)

# 간단 한국어 문장 분리 (완벽하지 않음 — 실험/튜닝 필요)
# 문장끝표현: 마침표, 물음표, 느낌표, 줄임표, 그리고 한국어 종결어미 "다.", "합니다.", "한다." 계열을 포괄하려 시도
SENT_END = re.compile(r'(?<=\S)([.。!?！？…]+|다\.|합니다\.|한다\.|습니다\.|습니다$)(\s+|$)')

def split_text_simple(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """기존 방식: 문자 기준으로 split (보조용)"""
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

def split_sentences_korean(text):
    """
    간단 룰 기반 한국어 문장 분리:
    - SENT_END 패턴으로 구분하되, 분리 후 각 문장을 트리밍
    - 패턴이 완벽하지 않으니 실제 수행 후 샘플로 검증 필요
    """
    if not text:
        return []
    # 통일된 개행 처리
    text = text.replace("\r\n", "\n").strip()
    parts = []
    last = 0
    for m in SENT_END.finditer(text):
        end_pos = m.end()
        sent = text[last:end_pos].strip()
        if sent:
            parts.append(sent)
        last = end_pos
    # 남은 꼬리
    if last < len(text):
        tail = text[last:].strip()
        if tail:
            parts.append(tail)
    # fallback
    if not parts:
        return [text]
    return parts

def split_by_article_with_titles(text, max_chunk_size=CHUNK_SIZE):
    """
    법령/표준문서용: '제n조' 단위로 분리하고 각 조항의 제목(매칭 그룹)을 반환.
    Returns a list of dicts: {"title": title_or_None, "body": text}
    """
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
        body = re.sub(r'\n{2,}', '\n', body).strip()
        results.append({"title": title, "body": body})
    return results

def process_pdf_file(path, source_tag):
    """PDF 파일을 열어 청크(문장) 단위로 텍스트를 yield"""
    doc = fitz.open(path)
    fname = os.path.basename(path)

    # chunk_idx는 각 파일에서 연속으로 증가시키는 인덱스
    chunk_idx_global = 0

    # law/standard는 문서 전체를 모아서 '제n조' 단위로 처리
    if source_tag in ("law", "standard"):
        # 모아서 처리
        pages_text = []
        for page_num, page in enumerate(doc, start=1):
            txt = page.get_text()
            if not txt:
                continue
            if len(txt) > MAX_PAGE_CHARS:
                print(f"⚠️  Truncating large page {page_num} in {fname} ({len(txt)} chars)")
                txt = txt[:MAX_PAGE_CHARS]
            pages_text.append(txt)
        full_text = "\n".join(pages_text)
        if not full_text.strip():
            return

        articles = split_by_article_with_titles(full_text)
        for art_idx, art in enumerate(articles):
            title = art.get("title")
            body = art.get("body", "")
            # 문장 단위로 분리해서 각 문장 하나의 chunk로 만듦
            sents = split_sentences_korean(body)
            for si, s in enumerate(sents):
                yield {
                    "chunk_id": str(uuid.uuid4()),
                    "source_file": fname,
                    "source_tag": source_tag,
                    "page": None,
                    "chunk_idx": chunk_idx_global,
                    "level": "article_sentence",
                    "article_title": title,
                    "text": s
                }
                chunk_idx_global += 1
    else:
        # reference / user 같은 계약서류: 페이지별로 문장단위 분할
        for page_num, page in enumerate(doc, start=1):
            txt = page.get_text()
            if not txt:
                continue
            if len(txt) > MAX_PAGE_CHARS:
                print(f"⚠️  Truncating large page {page_num} in {fname} ({len(txt)} chars)")
                txt = txt[:MAX_PAGE_CHARS]
            # 페이지 텍스트를 문장 단위로 분리
            sents = split_sentences_korean(txt)
            for si, s in enumerate(sents):
                yield {
                    "chunk_id": str(uuid.uuid4()),
                    "source_file": fname,
                    "source_tag": source_tag,
                    "page": page_num,
                    "chunk_idx": chunk_idx_global,
                    "level": "sentence",
                    "article_title": None,
                    "text": s
                }
                chunk_idx_global += 1

def main():
    os.makedirs(os.path.join("..", "outputs"), exist_ok=True)
    out_path = OUT_FILE

    file_list = []

    if os.path.isdir(CONTRACTS_REF_DIR):
        for f in os.listdir(CONTRACTS_REF_DIR):
            if f.lower().endswith(".pdf"):
                file_list.append((os.path.join(CONTRACTS_REF_DIR, f), "reference"))

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