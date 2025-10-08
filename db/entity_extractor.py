"""
법률 엔티티 추출기

법률 문서에서 조항, 법령, 개념 등의 엔티티를 추출하고 관계를 파악
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """엔티티 타입"""
    ARTICLE = "Article"          # 조항 (제1조, 제2조 등)
    LAW = "Law"                  # 법령명
    CONCEPT = "Concept"          # 법률 개념
    CLAUSE = "Clause"            # 약관 조항
    REFERENCE = "Reference"      # 참조 (제1조 제2항 등)
    PENALTY = "Penalty"          # 벌칙/처벌
    PROCEDURE = "Procedure"      # 절차/방법

@dataclass
class Entity:
    """추출된 엔티티"""
    text: str                    # 원본 텍스트
    entity_type: EntityType      # 엔티티 타입
    start_pos: int              # 시작 위치
    end_pos: int                # 끝 위치
    properties: Dict[str, Any]  # 추가 속성
    source_file: str            # 출처 파일
    chunk_id: str               # 청크 ID

@dataclass
class Relationship:
    """엔티티 간 관계"""
    from_entity: Entity         # 시작 엔티티
    to_entity: Entity           # 끝 엔티티
    relation_type: str          # 관계 타입
    confidence: float           # 신뢰도 (0.0 ~ 1.0)
    context: str                # 관계가 발견된 문맥

class LegalEntityExtractor:
    """법률 엔티티 추출기"""
    
    def __init__(self):
        """초기화 및 정규식 패턴 설정"""
        self._setup_patterns()
    
    def _setup_patterns(self):
        """정규식 패턴 설정"""
        # 조항 패턴 (제1조, 제2조 제1항 등)
        self.article_pattern = re.compile(
            r'(제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?)',
            re.MULTILINE
        )
        
        # 법령명 패턴
        self.law_pattern = re.compile(
            r'([가-힣]+법(?:률)?|약관\s*규제에\s*관한\s*법률|금융소비자\s*보호에\s*관한\s*법률)',
            re.MULTILINE
        )
        
        # 참조 패턴 (제1조 제2항에 의하면, 제3조에 따라 등)
        self.reference_pattern = re.compile(
            r'(제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?에\s*(?:의하면|따라|의하여))',
            re.MULTILINE
        )
        
        # 벌칙/처벌 패턴
        self.penalty_pattern = re.compile(
            r'(벌금|징역|과태료|과징금|처벌|벌칙)',
            re.MULTILINE
        )
        
        # 절차/방법 패턴
        self.procedure_pattern = re.compile(
            r'(신고|신청|통지|고지|승인|허가|등록|변경|해지|철회)',
            re.MULTILINE
        )
        
        # 금액 패턴
        self.amount_pattern = re.compile(
            r'(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:원|만원|억원|천원))',
            re.MULTILINE
        )
        
        # 기간 패턴
        self.period_pattern = re.compile(
            r'(\d+\s*(?:일|개월|년|주|시간|분|초))',
            re.MULTILINE
        )
    
    def extract_entities(self, text: str, source_file: str, chunk_id: str) -> List[Entity]:
        """
        텍스트에서 법률 엔티티 추출
        
        Args:
            text: 추출할 텍스트
            source_file: 출처 파일명
            chunk_id: 청크 ID
            
        Returns:
            추출된 엔티티 리스트
        """
        entities = []
        
        # 조항 추출
        entities.extend(self._extract_articles(text, source_file, chunk_id))
        
        # 법령명 추출
        entities.extend(self._extract_laws(text, source_file, chunk_id))
        
        # 참조 추출
        entities.extend(self._extract_references(text, source_file, chunk_id))
        
        # 벌칙/처벌 추출
        entities.extend(self._extract_penalties(text, source_file, chunk_id))
        
        # 절차/방법 추출
        entities.extend(self._extract_procedures(text, source_file, chunk_id))
        
        # 중복 제거 및 정렬
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_pos)
        
        return entities
    
    def _extract_articles(self, text: str, source_file: str, chunk_id: str) -> List[Entity]:
        """조항 엔티티 추출"""
        entities = []
        for match in self.article_pattern.finditer(text):
            article_text = match.group(1)
            entities.append(Entity(
                text=article_text,
                entity_type=EntityType.ARTICLE,
                start_pos=match.start(),
                end_pos=match.end(),
                properties={
                    "article_number": self._extract_article_number(article_text),
                    "has_paragraph": "항" in article_text,
                    "has_item": "호" in article_text
                },
                source_file=source_file,
                chunk_id=chunk_id
            ))
        return entities
    
    def _extract_laws(self, text: str, source_file: str, chunk_id: str) -> List[Entity]:
        """법령명 엔티티 추출"""
        entities = []
        for match in self.law_pattern.finditer(text):
            law_text = match.group(1)
            entities.append(Entity(
                text=law_text,
                entity_type=EntityType.LAW,
                start_pos=match.start(),
                end_pos=match.end(),
                properties={
                    "law_type": "법률" if "법률" in law_text else "법",
                    "is_contract_law": "약관" in law_text,
                    "is_consumer_law": "소비자" in law_text
                },
                source_file=source_file,
                chunk_id=chunk_id
            ))
        return entities
    
    def _extract_references(self, text: str, source_file: str, chunk_id: str) -> List[Entity]:
        """참조 엔티티 추출"""
        entities = []
        for match in self.reference_pattern.finditer(text):
            ref_text = match.group(1)
            entities.append(Entity(
                text=ref_text,
                entity_type=EntityType.REFERENCE,
                start_pos=match.start(),
                end_pos=match.end(),
                properties={
                    "reference_type": "의하면" if "의하면" in ref_text else "따라",
                    "referenced_article": self._extract_article_number(ref_text)
                },
                source_file=source_file,
                chunk_id=chunk_id
            ))
        return entities
    
    def _extract_penalties(self, text: str, source_file: str, chunk_id: str) -> List[Entity]:
        """벌칙/처벌 엔티티 추출"""
        entities = []
        for match in self.penalty_pattern.finditer(text):
            penalty_text = match.group(1)
            entities.append(Entity(
                text=penalty_text,
                entity_type=EntityType.PENALTY,
                start_pos=match.start(),
                end_pos=match.end(),
                properties={
                    "penalty_type": penalty_text,
                    "severity": self._get_penalty_severity(penalty_text)
                },
                source_file=source_file,
                chunk_id=chunk_id
            ))
        return entities
    
    def _extract_procedures(self, text: str, source_file: str, chunk_id: str) -> List[Entity]:
        """절차/방법 엔티티 추출"""
        entities = []
        for match in self.procedure_pattern.finditer(text):
            procedure_text = match.group(1)
            entities.append(Entity(
                text=procedure_text,
                entity_type=EntityType.PROCEDURE,
                start_pos=match.start(),
                end_pos=match.end(),
                properties={
                    "procedure_type": procedure_text,
                    "is_mandatory": procedure_text in ["신고", "신청", "통지", "고지"]
                },
                source_file=source_file,
                chunk_id=chunk_id
            ))
        return entities
    
    def _extract_article_number(self, text: str) -> int:
        """조항 번호 추출"""
        match = re.search(r'제\s*(\d+)\s*조', text)
        return int(match.group(1)) if match else 0
    
    def _get_penalty_severity(self, penalty_text: str) -> str:
        """벌칙 심각도 분류"""
        if penalty_text in ["징역", "벌금"]:
            return "high"
        elif penalty_text in ["과태료", "과징금"]:
            return "medium"
        else:
            return "low"
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """중복 엔티티 제거"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.text, entity.entity_type, entity.start_pos)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """
        엔티티 간 관계 추출
        
        Args:
            entities: 추출된 엔티티 리스트
            text: 원본 텍스트
            
        Returns:
            관계 리스트
        """
        relationships = []
        
        # 조항 간 참조 관계
        relationships.extend(self._extract_article_references(entities, text))
        
        # 법령-조항 관계
        relationships.extend(self._extract_law_article_relationships(entities, text))
        
        # 절차-조항 관계
        relationships.extend(self._extract_procedure_relationships(entities, text))
        
        return relationships
    
    def _extract_article_references(self, entities: List[Entity], text: str) -> List[Relationship]:
        """조항 간 참조 관계 추출"""
        relationships = []
        
        articles = [e for e in entities if e.entity_type == EntityType.ARTICLE]
        references = [e for e in entities if e.entity_type == EntityType.REFERENCE]
        
        for ref in references:
            ref_article_num = ref.properties.get("referenced_article", 0)
            for article in articles:
                article_num = article.properties.get("article_number", 0)
                if ref_article_num == article_num and ref_article_num > 0:
                    relationships.append(Relationship(
                        from_entity=article,
                        to_entity=ref,
                        relation_type="REFERENCES",
                        confidence=0.9,
                        context=text[ref.start_pos-50:ref.end_pos+50]
                    ))
        
        return relationships
    
    def _extract_law_article_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """법령-조항 관계 추출"""
        relationships = []
        
        laws = [e for e in entities if e.entity_type == EntityType.LAW]
        articles = [e for e in entities if e.entity_type == EntityType.ARTICLE]
        
        for law in laws:
            for article in articles:
                # 같은 문맥에 있는 경우 관계 설정
                if abs(law.start_pos - article.start_pos) < 200:
                    relationships.append(Relationship(
                        from_entity=law,
                        to_entity=article,
                        relation_type="CONTAINS",
                        confidence=0.8,
                        context=text[min(law.start_pos, article.start_pos):max(law.end_pos, article.end_pos)]
                    ))
        
        return relationships
    
    def _extract_procedure_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """절차-조항 관계 추출"""
        relationships = []
        
        procedures = [e for e in entities if e.entity_type == EntityType.PROCEDURE]
        articles = [e for e in entities if e.entity_type == EntityType.ARTICLE]
        
        for procedure in procedures:
            for article in articles:
                # 같은 문맥에 있는 경우 관계 설정
                if abs(procedure.start_pos - article.start_pos) < 100:
                    relationships.append(Relationship(
                        from_entity=article,
                        to_entity=procedure,
                        relation_type="DEFINES_PROCEDURE",
                        confidence=0.7,
                        context=text[min(procedure.start_pos, article.start_pos):max(procedure.end_pos, article.end_pos)]
                    ))
        
        return relationships
