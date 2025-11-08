"""텍스트 분할 유틸리티"""
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


def split_contract_by_clauses(text: str) -> List[str]:
    """
    계약서를 조항 단위로 분리
    
    예: "제1조 (목적)", "제2조 (정의)" 등의 패턴으로 분리
    """
    # 조항 패턴: "제N조", "제N조 (제목)", "제N장", "제N절" 등
    clause_pattern = r'(제\d+조\s*(?:\([^)]+\))?|제\d+장|제\d+절)'
    
    # 조항 시작 위치 찾기
    matches = list(re.finditer(clause_pattern, text))
    
    if not matches:
        # 조항 패턴이 없으면 문단 단위로 분리
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    clauses = []
    for i, match in enumerate(matches):
        start = match.start()
        # 다음 조항 시작 전까지 또는 텍스트 끝까지
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_text = text[start:end].strip()
        if clause_text:
            clauses.append(clause_text)
    
    return clauses


def create_chunk_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> TextSplitter:
    """법률 문서 청킹을 위한 TextSplitter 생성"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )



