"""
진짜 GraphRAG를 위한 지식 그래프 스키마

핵심 원칙:
1. 노드는 엔티티 (사물, 개념, 사건)
2. 엣지는 의미적 관계
3. 속성은 메타데이터 + 임베딩 벡터
"""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class NodeSchema:
    """노드 스키마 정의"""
    label: str
    properties: Dict[str, str]
    indexes: List[str]
    vector_index: bool = False

@dataclass
class RelationshipSchema:
    """관계 스키마 정의"""
    type: str
    from_node: str
    to_node: str
    properties: Dict[str, str]

# =============================================================================
# 노드 정의
# =============================================================================

GRAPH_SCHEMA = {
    "nodes": {
        # 법률 조항
        "LawArticle": NodeSchema(
            label="법률조항",
            properties={
                "id": "string",           # 제7조, 제8조 등
                "title": "string",        # 조항 제목
                "content": "text",        # 조항 전문
                "category": "string",     # 면책조항, 손해배상 등
                "embedding": "vector",    # 768차원 벡터
            },
            indexes=["id", "category"],
            vector_index=True
        ),
        
        # 위반 사례
        "ViolationCase": NodeSchema(
            label="위반사례",
            properties={
                "id": "string",
                "original_text": "text",      # 불공정 약관 원문
                "corrected_text": "text",     # 수정된 약관
                "violation_reason": "text",   # 시정 요청 사유
                "company": "string",          # 회사명
                "year": "integer",            # 적발 연도
                "severity": "string",         # 심각도 (high, medium, low)
                "embedding": "vector",        # 원문 임베딩
            },
            indexes=["id", "company", "year", "severity"],
            vector_index=True
        ),
        
        # 위반 유형
        "ViolationType": NodeSchema(
            label="위반유형",
            properties={
                "id": "string",
                "name": "string",             # 전면면책, 일방적변경 등
                "description": "text",        # 유형 설명
                "legal_basis": "text",        # 법적 근거
                "embedding": "vector",
            },
            indexes=["id", "name"],
            vector_index=True
        ),
        
        # 키워드/패턴
        "Keyword": NodeSchema(
            label="키워드",
            properties={
                "text": "string",
                "type": "string",             # hard_rule, soft_pattern
                "weight": "float",            # 중요도 가중치
                "regex_pattern": "string",    # 정규식 패턴
            },
            indexes=["text", "type"],
            vector_index=False
        ),
        
        # 판례
        "Precedent": NodeSchema(
            label="판례",
            properties={
                "id": "string",
                "court": "string",            # 법원
                "case_number": "string",      # 사건번호
                "decision": "text",           # 판결문
                "date": "date",
                "embedding": "vector",
            },
            indexes=["id", "case_number"],
            vector_index=True
        ),
        
        # 해석 지침
        "Guideline": NodeSchema(
            label="해석지침",
            properties={
                "id": "string",
                "source": "string",           # 금융감독원, 공정위 등
                "content": "text",
                "issued_date": "date",
                "embedding": "vector",
            },
            indexes=["id", "source"],
            vector_index=True
        ),
        
        # 회사
        "Company": NodeSchema(
            label="회사",
            properties={
                "name": "string",
                "industry": "string",         # 은행, 보험, 여신금융 등
                "violation_count": "integer",
            },
            indexes=["name", "industry"],
            vector_index=False
        ),
    },
    
    # =============================================================================
    # 관계 정의
    # =============================================================================
    
    "relationships": {
        # 위반 사례 ↔ 법률 조항
        "VIOLATES": RelationshipSchema(
            type="VIOLATES",
            from_node="ViolationCase",
            to_node="LawArticle",
            properties={
                "confidence": "float",        # 위반 확신도 0-1
                "reason": "text",             # 위반 이유
            }
        ),
        
        # 사례 간 유사도
        "SIMILAR_TO": RelationshipSchema(
            type="SIMILAR_TO",
            from_node="ViolationCase",
            to_node="ViolationCase",
            properties={
                "similarity_score": "float",  # 코사인 유사도
                "similarity_type": "string",  # semantic, lexical, structural
            }
        ),
        
        # 법률 조항 간 참조
        "REFERENCES": RelationshipSchema(
            type="REFERENCES",
            from_node="LawArticle",
            to_node="LawArticle",
            properties={
                "reference_type": "string",   # 인용, 준용, 예외 등
            }
        ),
        
        # 위반 유형 관계
        "CATEGORIZED_AS": RelationshipSchema(
            type="CATEGORIZED_AS",
            from_node="ViolationCase",
            to_node="ViolationType",
            properties={
                "confidence": "float",
            }
        ),
        
        # 키워드 ↔ 위반 유형
        "INDICATES": RelationshipSchema(
            type="INDICATES",
            from_node="Keyword",
            to_node="ViolationType",
            properties={
                "strength": "float",          # 지시 강도
            }
        ),
        
        # 키워드 출현
        "CONTAINS": RelationshipSchema(
            type="CONTAINS",
            from_node="ViolationCase",
            to_node="Keyword",
            properties={
                "count": "integer",           # 출현 횟수
                "positions": "list",          # 출현 위치
            }
        ),
        
        # 판례 연결
        "SUPPORTED_BY": RelationshipSchema(
            type="SUPPORTED_BY",
            from_node="ViolationCase",
            to_node="Precedent",
            properties={
                "relevance": "float",
            }
        ),
        
        # 지침 연결
        "GUIDED_BY": RelationshipSchema(
            type="GUIDED_BY",
            from_node="LawArticle",
            to_node="Guideline",
            properties={
                "relevance": "float",
            }
        ),
        
        # 회사 ↔ 위반 사례
        "COMMITTED_BY": RelationshipSchema(
            type="COMMITTED_BY",
            from_node="ViolationCase",
            to_node="Company",
            properties={
                "date": "date",
            }
        ),
        
        # 연쇄 위반 (한 위반이 다른 위반을 유발)
        "LEADS_TO": RelationshipSchema(
            type="LEADS_TO",
            from_node="ViolationCase",
            to_node="ViolationCase",
            properties={
                "causality_score": "float",
            }
        ),
        
        # 수정 관계
        "CORRECTED_BY": RelationshipSchema(
            type="CORRECTED_BY",
            from_node="ViolationCase",
            to_node="ViolationCase",  # 수정된 버전
            properties={
                "correction_type": "string",
            }
        ),
    }
}

# =============================================================================
# 벡터 인덱스 설정
# =============================================================================

VECTOR_INDEX_CONFIG = {
    "dimension": 768,  # multilingual-MiniLM-L12-v2
    "similarity_function": "cosine",
    "index_nodes": [
        "LawArticle",
        "ViolationCase", 
        "ViolationType",
        "Precedent",
        "Guideline"
    ]
}

# =============================================================================
# 복합 인덱스 (성능 최적화)
# =============================================================================

COMPOSITE_INDEXES = [
    {
        "label": "ViolationCase",
        "properties": ["severity", "year"]
    },
    {
        "label": "Company",
        "properties": ["industry", "violation_count"]
    }
]