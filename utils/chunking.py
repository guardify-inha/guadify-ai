"""
적응형 청킹(Adaptive Chunking) 전략 모듈

문서 유형에 따라 최적화된 텍스트 청킹 전략을 제공합니다.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """문서 유형 열거형"""
    CONTRACT = "contract"
    LAW = "law"
    STANDARD = "standard"
    PRECEDENT = "precedent"  # 판례
    REGULATION = "regulation"  # 규정/시행령

class ChunkingStrategy:
    """청킹 전략 클래스"""
    
    def __init__(self, 
                 chunk_size: int = 1200,
                 chunk_overlap: int = 200,
                 preserve_semantic_boundaries: bool = True,
                 use_sentence_boundaries: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_semantic_boundaries = preserve_semantic_boundaries
        self.use_sentence_boundaries = use_sentence_boundaries

class AdaptiveChunker:
    """적응형 청킹 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 문서 유형별 전략 설정
        self.strategies = {
            DocumentType.CONTRACT: ChunkingStrategy(
                chunk_size=1000,
                chunk_overlap=150,
                preserve_semantic_boundaries=True,
                use_sentence_boundaries=True
            ),
            DocumentType.LAW: ChunkingStrategy(
                chunk_size=1500,
                chunk_overlap=200,
                preserve_semantic_boundaries=True,
                use_sentence_boundaries=False  # 조항 단위가 우선
            ),
            DocumentType.STANDARD: ChunkingStrategy(
                chunk_size=1200,
                chunk_overlap=200,
                preserve_semantic_boundaries=True,
                use_sentence_boundaries=False
            ),
            DocumentType.PRECEDENT: ChunkingStrategy(
                chunk_size=800,
                chunk_overlap=100,
                preserve_semantic_boundaries=True,
                use_sentence_boundaries=True
            ),
            DocumentType.REGULATION: ChunkingStrategy(
                chunk_size=1400,
                chunk_overlap=180,
                preserve_semantic_boundaries=True,
                use_sentence_boundaries=False
            )
        }
        
        # 정규표현식 패턴들
        self.patterns = {
            'article': re.compile(r'(제\s*\d+\s*조[^\n\r]*)', flags=re.MULTILINE),
            'paragraph': re.compile(r'(제\s*\d+\s*항[^\n\r]*)', flags=re.MULTILINE),
            'subparagraph': re.compile(r'(제\s*\d+\s*호[^\n\r]*)', flags=re.MULTILINE),
            'sentence': re.compile(r'[.!?]\s+'),
            'clause': re.compile(r'(제\s*\d+\s*절[^\n\r]*)', flags=re.MULTILINE),
            'contract_section': re.compile(r'(제\s*\d+\s*장[^\n\r]*)', flags=re.MULTILINE),
            'contract_article': re.compile(r'(제\s*\d+\s*조[^\n\r]*)', flags=re.MULTILINE),
        }

    def chunk_text(self, 
                   text: str, 
                   document_type: DocumentType,
                   source_file: str = "unknown",
                   page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        텍스트를 문서 유형에 따라 적응적으로 청킹합니다.
        
        Args:
            text: 청킹할 텍스트
            document_type: 문서 유형
            source_file: 소스 파일명
            page: 페이지 번호 (선택사항)
            
        Returns:
            청크 정보가 담긴 딕셔너리 리스트
        """
        self.logger.info(f"문서 유형 '{document_type.value}'에 대한 청킹 시작")
        
        # 문서 유형별 청킹 전략 선택
        if document_type == DocumentType.CONTRACT:
            return self._chunk_contract(text, source_file, page)
        elif document_type == DocumentType.LAW:
            return self._chunk_law(text, source_file, page)
        elif document_type == DocumentType.STANDARD:
            return self._chunk_standard(text, source_file, page)
        elif document_type == DocumentType.PRECEDENT:
            return self._chunk_precedent(text, source_file, page)
        elif document_type == DocumentType.REGULATION:
            return self._chunk_regulation(text, source_file, page)
        else:
            # 기본 청킹
            return self._chunk_default(text, source_file, page)

    def _chunk_contract(self, text: str, source_file: str, page: Optional[int]) -> List[Dict[str, Any]]:
        """계약서 청킹: 조항 단위 + 문장 경계 고려"""
        strategy = self.strategies[DocumentType.CONTRACT]
        chunks = []
        
        # 1단계: 조항 단위로 분할
        article_matches = list(self.patterns['contract_article'].finditer(text))
        
        if article_matches:
            # 조항이 있는 경우
            for i, match in enumerate(article_matches):
                start_idx = match.start()
                end_idx = article_matches[i+1].start() if i+1 < len(article_matches) else len(text)
                article_text = text[start_idx:end_idx].strip()
                
                # 조항이 너무 길면 문장 단위로 재분할
                if len(article_text) > strategy.chunk_size:
                    sub_chunks = self._split_by_sentences(article_text, strategy)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append(self._create_chunk_metadata(
                            sub_chunk, source_file, page, f"article_{i}_part_{j}"
                        ))
                else:
                    chunks.append(self._create_chunk_metadata(
                        article_text, source_file, page, f"article_{i}"
                    ))
        else:
            # 조항이 없는 경우 문장 단위로 분할
            chunks = self._split_by_sentences(text, strategy)
            for i, chunk in enumerate(chunks):
                chunks[i] = self._create_chunk_metadata(
                    chunk, source_file, page, f"section_{i}"
                )
        
        return chunks

    def _chunk_law(self, text: str, source_file: str, page: Optional[int]) -> List[Dict[str, Any]]:
        """법률 청킹: 조항 단위 + 하위 단위(항, 호) 고려"""
        strategy = self.strategies[DocumentType.LAW]
        chunks = []
        
        # 조항 단위로 분할
        article_matches = list(self.patterns['article'].finditer(text))
        
        if article_matches:
            for i, match in enumerate(article_matches):
                start_idx = match.start()
                end_idx = article_matches[i+1].start() if i+1 < len(article_matches) else len(text)
                article_text = text[start_idx:end_idx].strip()
                
                # 조항이 너무 길면 항 단위로 재분할
                if len(article_text) > strategy.chunk_size:
                    sub_chunks = self._split_by_paragraphs(article_text, strategy)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append(self._create_chunk_metadata(
                            sub_chunk, source_file, page, f"article_{i}_paragraph_{j}"
                        ))
                else:
                    chunks.append(self._create_chunk_metadata(
                        article_text, source_file, page, f"article_{i}"
                    ))
        else:
            # 조항이 없는 경우 기본 분할
            chunks = self._split_by_sentences(text, strategy)
            for i, chunk in enumerate(chunks):
                chunks[i] = self._create_chunk_metadata(
                    chunk, source_file, page, f"section_{i}"
                )
        
        return chunks

    def _chunk_standard(self, text: str, source_file: str, page: Optional[int]) -> List[Dict[str, Any]]:
        """표준 문서 청킹: 조항 단위"""
        return self._chunk_law(text, source_file, page)  # 법률과 동일한 전략

    def _chunk_precedent(self, text: str, source_file: str, page: Optional[int]) -> List[Dict[str, Any]]:
        """판례 청킹: 사건별/쟁점별 분할"""
        strategy = self.strategies[DocumentType.PRECEDENT]
        chunks = []
        
        # 판례의 경우 문장 단위 분할이 더 적합
        chunks = self._split_by_sentences(text, strategy)
        for i, chunk in enumerate(chunks):
            chunks[i] = self._create_chunk_metadata(
                chunk, source_file, page, f"precedent_section_{i}"
            )
        
        return chunks

    def _chunk_regulation(self, text: str, source_file: str, page: Optional[int]) -> List[Dict[str, Any]]:
        """규정/시행령 청킹: 조항 단위"""
        return self._chunk_law(text, source_file, page)  # 법률과 동일한 전략

    def _chunk_default(self, text: str, source_file: str, page: Optional[int]) -> List[Dict[str, Any]]:
        """기본 청킹: 고정 크기"""
        strategy = ChunkingStrategy()
        chunks = []
        
        start = 0
        length = len(text)
        chunk_idx = 0
        
        while start < length:
            end = min(length, start + strategy.chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(self._create_chunk_metadata(
                    chunk, source_file, page, f"chunk_{chunk_idx}"
                ))
                chunk_idx += 1
            
            new_start = end - strategy.chunk_overlap
            if new_start <= start:
                new_start = start + strategy.chunk_size
            start = new_start
        
        return chunks

    def _split_by_sentences(self, text: str, strategy: ChunkingStrategy) -> List[str]:
        """문장 단위로 분할"""
        if not strategy.use_sentence_boundaries:
            return [text]
        
        sentences = self.patterns['sentence'].split(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= strategy.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _split_by_paragraphs(self, text: str, strategy: ChunkingStrategy) -> List[str]:
        """항 단위로 분할"""
        paragraph_matches = list(self.patterns['paragraph'].finditer(text))
        
        if not paragraph_matches:
            return self._split_by_sentences(text, strategy)
        
        chunks = []
        for i, match in enumerate(paragraph_matches):
            start_idx = match.start()
            end_idx = paragraph_matches[i+1].start() if i+1 < len(paragraph_matches) else len(text)
            paragraph_text = text[start_idx:end_idx].strip()
            
            if len(paragraph_text) <= strategy.chunk_size:
                chunks.append(paragraph_text)
            else:
                # 항이 너무 길면 문장 단위로 재분할
                sub_chunks = self._split_by_sentences(paragraph_text, strategy)
                chunks.extend(sub_chunks)
        
        return chunks

    def _create_chunk_metadata(self, 
                              text: str, 
                              source_file: str, 
                              page: Optional[int], 
                              chunk_identifier: str) -> Dict[str, Any]:
        """청크 메타데이터 생성"""
        return {
            "chunk_id": str(uuid.uuid4()),
            "source_file": source_file,
            "page": page,
            "chunk_identifier": chunk_identifier,
            "text": text,
            "length": len(text)
        }

# 전역 인스턴스
adaptive_chunker = AdaptiveChunker()

def chunk_document(text: str, 
                   document_type: str, 
                   source_file: str = "unknown",
                   page: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    문서를 적응적으로 청킹하는 메인 함수
    
    Args:
        text: 청킹할 텍스트
        document_type: 문서 유형 ('contract', 'law', 'standard', 'precedent', 'regulation')
        source_file: 소스 파일명
        page: 페이지 번호
        
    Returns:
        청크 정보가 담긴 딕셔너리 리스트
    """
    try:
        doc_type = DocumentType(document_type)
        return adaptive_chunker.chunk_text(text, doc_type, source_file, page)
    except ValueError:
        logger.warning(f"알 수 없는 문서 유형: {document_type}, 기본 청킹 사용")
        return adaptive_chunker._chunk_default(text, source_file, page)



