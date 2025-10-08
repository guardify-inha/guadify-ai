# File: scripts/extract_and_chunk_v2.py
"""
새로운 데이터 구조에 맞는 청킹 스크립트
- 법령 데이터 (최고 우선순위)
- 표준약관 데이터 (중간 우선순위)  
- 보도자료 데이터 (최저 우선순위, 엑셀에서 전처리된 데이터)
"""
import os
import json
import uuid
import re
import fitz  # pymupdf
from tqdm import tqdm

# 현재 스크립트의 위치를 기준으로 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 데이터 루트
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
# 처리할 하위 폴더들
LAW_DIR = os.path.join(DATA_ROOT, "law")
STANDARD_DIR = os.path.join(DATA_ROOT, "standard")
PROCESSED_REFERENCE_FILE = os.path.join(DATA_ROOT, "processed", "processed_reference_cases.jsonl")

OUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "chunks_v2.jsonl")

# 디버깅용 경로 확인
print(f"DEBUG: SCRIPT_DIR = {SCRIPT_DIR}")
print(f"DEBUG: PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DEBUG: DATA_ROOT = {DATA_ROOT}")
print(f"DEBUG: LAW_DIR = {LAW_DIR}")
print(f"DEBUG: STANDARD_DIR = {STANDARD_DIR}")
print(f"DEBUG: PROCESSED_REFERENCE_FILE = {PROCESSED_REFERENCE_FILE}")
print(f"DEBUG: LAW_DIR exists = {os.path.exists(LAW_DIR)}")
print(f"DEBUG: STANDARD_DIR exists = {os.path.exists(STANDARD_DIR)}")
print(f"DEBUG: PROCESSED_REFERENCE_FILE exists = {os.path.exists(PROCESSED_REFERENCE_FILE)}")
CHUNK_SIZE = 1200  # 문자 기준
CHUNK_OVERLAP = 200
MAX_PAGE_CHARS = 200_000

# 조항(제n조) 기준 분할 정규식 (한글 법조문 제목 패턴)
ARTICLE_RE = re.compile(r'(제\s*\d+\s*조[^\n\r]*)', flags=re.MULTILINE)

# 간단 한국어 문장 분리
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
    """간단 룰 기반 한국어 문장 분리"""
    if not text:
        return []
    text = text.replace("\r\n", "\n").strip()
    parts = []
    last = 0
    for m in SENT_END.finditer(text):
        end_pos = m.end()
        sent = text[last:end_pos].strip()
        if sent:
            parts.append(sent)
        last = end_pos
    if last < len(text):
        tail = text[last:].strip()
        if tail:
            parts.append(tail)
    if not parts:
        return [text]
    return parts

def split_by_article_with_titles(text, max_chunk_size=CHUNK_SIZE):
    """법령/표준문서용: '제n조' 단위로 분리하고 각 조항의 제목을 반환"""
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

def process_pdf_file(path, source_tag, priority):
    """PDF 파일을 열어 청크(문장) 단위로 텍스트를 yield"""
    doc = fitz.open(path)
    fname = os.path.basename(path)
    chunk_idx_global = 0

    # law/standard는 문서 전체를 모아서 '제n조' 단위로 처리
    if source_tag in ("law", "standard"):
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
                    "text": s,
                    "priority": priority,
                    "document_type": "law" if source_tag == "law" else "standard"
                }
                chunk_idx_global += 1
    else:
        # 기타 문서: 페이지별로 문장단위 분할
        for page_num, page in enumerate(doc, start=1):
            txt = page.get_text()
            if not txt:
                continue
            if len(txt) > MAX_PAGE_CHARS:
                print(f"⚠️  Truncating large page {page_num} in {fname} ({len(txt)} chars)")
                txt = txt[:MAX_PAGE_CHARS]
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
                    "text": s,
                    "priority": priority,
                    "document_type": source_tag
                }
                chunk_idx_global += 1

def process_reference_cases():
    """전처리된 보도자료 사례들을 처리"""
    if not os.path.exists(PROCESSED_REFERENCE_FILE):
        print(f"⚠️  전처리된 보도자료 파일을 찾을 수 없습니다: {PROCESSED_REFERENCE_FILE}")
        return

    with open(PROCESSED_REFERENCE_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                case_data = json.loads(line.strip())
                
                # 구조화된 텍스트를 문장 단위로 분할
                structured_text = case_data.get('structured_text', '')
                sents = split_sentences_korean(structured_text)
                
                for si, sentence in enumerate(sents):
                    if len(sentence.strip()) < 10:  # 너무 짧은 문장 제외
                        continue
                        
                    yield {
                        "chunk_id": str(uuid.uuid4()),
                        "source_file": case_data.get('file_name', f'case_{line_num}'),
                        "source_tag": "reference",
                        "page": None,
                        "chunk_idx": si,
                        "level": "case_sentence",
                        "article_title": case_data.get('sub_topic', ''),
                        "text": sentence,
                        "priority": case_data.get('priority', 0.3),
                        "document_type": "reference",
                        "case_id": case_data.get('case_id', f'case_{line_num}'),
                        "sub_topic": case_data.get('sub_topic', ''),
                        "unfair_clause": case_data.get('unfair_clause', ''),
                        "correction_reason": case_data.get('correction_reason', ''),
                        "legal_basis": case_data.get('legal_basis', '')
                    }
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 (라인 {line_num}): {e}")
                continue
            except Exception as e:
                print(f"보도자료 처리 오류 (라인 {line_num}): {e}")
                continue

def main():
    os.makedirs(os.path.join("..", "outputs"), exist_ok=True)
    out_path = OUT_FILE

    print("🚀 새로운 데이터 구조로 청킹 시작")
    print("📊 우선순위: 법령(1.0) > 표준약관(0.8) >= 보도자료(0.3)")

    with open(out_path, "w", encoding="utf-8") as outf:
        total_chunks = 0
        
        # 1. 법령 데이터 처리 (최고 우선순위)
        if os.path.isdir(LAW_DIR):
            print("📚 법령 데이터 처리 중...")
            law_files = [f for f in os.listdir(LAW_DIR) if f.lower().endswith(".pdf")]
            for f in tqdm(law_files, desc="법령 파일"):
                try:
                    for rec in process_pdf_file(os.path.join(LAW_DIR, f), "law", 1.0):
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_chunks += 1
                except Exception as e:
                    print(f"법령 파일 처리 오류 {f}: {e}")
        else:
            print("⚠️  법령 디렉토리를 찾을 수 없습니다.")

        # 2. 표준약관 데이터 처리 (중간 우선순위)
        if os.path.isdir(STANDARD_DIR):
            print("📋 표준약관 데이터 처리 중...")
            standard_files = [f for f in os.listdir(STANDARD_DIR) if f.lower().endswith(".pdf")]
            for f in tqdm(standard_files, desc="표준약관 파일"):
                try:
                    for rec in process_pdf_file(os.path.join(STANDARD_DIR, f), "standard", 0.8):
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_chunks += 1
                except Exception as e:
                    print(f"표준약관 파일 처리 오류 {f}: {e}")
        else:
            print("⚠️  표준약관 디렉토리를 찾을 수 없습니다.")

        # 3. 보도자료 데이터 처리 (최저 우선순위)
        print("📰 보도자료 데이터 처리 중...")
        for rec in tqdm(process_reference_cases(), desc="보도자료 사례"):
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_chunks += 1

    print(f"✅ 청킹 완료: {total_chunks}개 청크 생성")
    print(f"📁 저장 위치: {out_path}")

if __name__ == "__main__":
    main()
