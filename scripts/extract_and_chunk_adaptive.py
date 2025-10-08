"""
적응형 청킹을 사용한 개선된 텍스트 추출 및 청킹 스크립트

기존 extract_and_chunk.py를 적응형 청킹 모듈을 사용하도록 개선한 버전입니다.
"""

import os
import json
import fitz  # pymupdf
from tqdm import tqdm
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.chunking import chunk_document, DocumentType

# 데이터 루트 (상대경로)
DATA_ROOT = "data"
# 처리할 하위 폴더들
CONTRACTS_DIR = os.path.join(DATA_ROOT, "contracts", "reference")
LAW_DIR = os.path.join(DATA_ROOT, "law")
STANDARD_DIR = os.path.join(DATA_ROOT, "standard")
PRECEDENT_DIR = os.path.join(DATA_ROOT, "precedent")  # 새로 추가

OUT_FILE = "outputs/chunks_adaptive.jsonl"
MAX_PAGE_CHARS = 200_000   # 페이지당 최대 글자 수 제한

def process_pdf_file_adaptive(path, source_tag):
    """
    적응형 청킹을 사용하여 PDF 파일을 처리합니다.
    
    Args:
        path: PDF 파일 경로
        source_tag: 문서 유형 ('contract', 'law', 'standard', 'precedent')
    """
    doc = fitz.open(path)
    fname = os.path.basename(path)
    
    print(f"📄 처리 중: {fname} (유형: {source_tag})")
    
    for page_num, page in enumerate(doc, start=1):
        txt = page.get_text()
        if not txt:
            continue

        # 너무 큰 페이지는 truncate 처리
        if len(txt) > MAX_PAGE_CHARS:
            print(f"⚠️  큰 페이지 {page_num} 잘라내기: {fname} ({len(txt)} 문자)")
            txt = txt[:MAX_PAGE_CHARS]

        # 적응형 청킹 적용
        try:
            chunks = chunk_document(
                text=txt,
                document_type=source_tag,
                source_file=fname,
                page=page_num
            )
            
            for chunk in chunks:
                # 기존 형식과 호환되도록 메타데이터 추가
                chunk_data = {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "source_tag": source_tag,
                    "page": chunk["page"],
                    "chunk_idx": chunk.get("chunk_identifier", "unknown"),
                    "text": chunk["text"],
                    "length": chunk["length"],
                    "chunking_strategy": "adaptive"  # 새로운 필드
                }
                yield chunk_data
                
        except Exception as e:
            print(f"❌ 청킹 오류 {fname} 페이지 {page_num}: {e}")
            # 오류 발생 시 기본 청킹으로 폴백
            yield {
                "chunk_id": str(uuid.uuid4()),
                "source_file": fname,
                "source_tag": source_tag,
                "page": page_num,
                "chunk_idx": 0,
                "text": txt[:1000],  # 첫 1000자만
                "length": min(len(txt), 1000),
                "chunking_strategy": "fallback"
            }

def main():
    """메인 실행 함수"""
    import uuid
    
    os.makedirs("outputs", exist_ok=True)
    out_path = OUT_FILE

    # 처리할 파일 목록 수집
    file_list = []
    
    # 계약서
    if os.path.isdir(CONTRACTS_DIR):
        for f in os.listdir(CONTRACTS_DIR):
            if f.lower().endswith(".pdf"):
                file_list.append((os.path.join(CONTRACTS_DIR, f), "contract"))
    
    # 법률 (temp 디렉토리 제외)
    if os.path.isdir(LAW_DIR):
        for f in os.listdir(LAW_DIR):
            if f.lower().endswith(".pdf") and f != "temp":
                file_list.append((os.path.join(LAW_DIR, f), "law"))
    
    # 표준
    if os.path.isdir(STANDARD_DIR):
        for f in os.listdir(STANDARD_DIR):
            if f.lower().endswith(".pdf"):
                file_list.append((os.path.join(STANDARD_DIR, f), "standard"))
    
    # 판례 (새로 추가)
    if os.path.isdir(PRECEDENT_DIR):
        for f in os.listdir(PRECEDENT_DIR):
            if f.lower().endswith(".pdf"):
                file_list.append((os.path.join(PRECEDENT_DIR, f), "precedent"))

    if not file_list:
        print("❌ 설정된 디렉토리에서 PDF 파일을 찾을 수 없습니다.")
        print(f"   확인할 디렉토리: {DATA_ROOT}")
        return

    print(f"🚀 {len(file_list)}개 PDF 파일 처리 시작...")
    print("📊 적응형 청킹 전략 적용:")
    print("   - 계약서: 조항 단위 + 문장 경계 고려")
    print("   - 법률: 조항 단위 + 하위 단위(항, 호) 고려")
    print("   - 표준: 조항 단위")
    print("   - 판례: 문장 단위 + 의미적 경계 고려")

    total_chunks = 0
    with open(out_path, "w", encoding="utf-8") as outf:
        for path, tag in tqdm(file_list, desc="📄 PDF 처리"):
            try:
                for rec in process_pdf_file_adaptive(path, tag):
                    outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1
            except Exception as e:
                print(f"❌ 파일 처리 오류 {path}: {e}")

    print(f"✅ 완료! 청크 파일 저장: {out_path}")
    print(f"📊 총 {total_chunks}개 청크 생성")

if __name__ == "__main__":
    main()
