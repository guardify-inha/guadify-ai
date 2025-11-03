"""
향상된 리트리버: 하이브리드 검색 + HyDE + LLM-as-Judge 통합

3단계: LLM-as-Judge 및 HyDE 기법 구현 - 통합
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector
from retriever.hybrid_retriever import HybridRetriever
from retriever.hyde_retriever import HyDERetriever
from retriever.llm_judge import LLMJudge


class EnhancedRetriever:
    """
    향상된 리트리버: 모든 기법 통합
    
    프로세스:
    1. 하이브리드 검색으로 초기 후보 선별
    2. HyDE로 가상 문서 생성 및 추가 후보 확보
    3. LLM-as-Judge로 최종 후보들 평가 및 순위화
    """
    
    def __init__(self, conn: Neo4jConnector):
        self.conn = conn
        self.hybrid_retriever = HybridRetriever(conn)
        self.hyde_retriever = HyDERetriever(conn)
        self.llm_judge = LLMJudge()
    
    def retrieve(self, query: str, top_k: int = 5, use_hyde: bool = True, use_llm_judge: bool = True) -> List[Dict]:
        """
        향상된 검색 메인 함수
        
        Args:
            query: 자연어 질의
            top_k: 반환할 결과 개수
            use_hyde: HyDE 기법 사용 여부
            use_llm_judge: LLM-as-Judge 사용 여부
            
        Returns:
            검색 결과 리스트 (위반 가능성 점수 기준 정렬)
        """
        # 1단계: 하이브리드 검색 (초기 후보 선별)
        candidates = self.hybrid_retriever.retrieve(query, top_k=15)  # 더 많은 후보 수집
        
        if not candidates:
            return []
        
        # 2단계: HyDE로 보강 (선택적)
        if use_hyde:
            candidates = self.hyde_retriever.enhance_with_hyde(query, candidates, top_k=15)
        
        # 3단계: LLM-as-Judge로 최종 평가 (선택적)
        if use_llm_judge:
            # 상위 후보들만 LLM 비교 평가 (비용 절감 + 상대적 비교로 정확도 향상)
            top_candidates = candidates[:5]  # 상위 5개만 비교 평가
            if len(top_candidates) >= 2:
                # 여러 후보를 한 번에 비교하여 상대적 순위 결정
                candidates = self.llm_judge.compare_candidates(query, top_candidates)
            else:
                # 후보가 1개 이하일 경우 배치 평가
                candidates = self.llm_judge.judge_batch(query, top_candidates, top_k=top_k)
        else:
            # LLM 평가 없이 하이브리드 점수만 사용
            candidates = candidates[:top_k]
        
        return candidates
    
    def retrieve_with_comparison(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        LLM 비교 평가를 사용한 검색
        
        여러 후보를 한 번에 비교하여 상대적 순위 결정
        """
        # 1단계: 하이브리드 검색
        candidates = self.hybrid_retriever.retrieve(query, top_k=15)
        
        if not candidates:
            return []
        
        # 2단계: HyDE 보강
        candidates = self.hyde_retriever.enhance_with_hyde(query, candidates, top_k=15)
        
        # 3단계: LLM 비교 평가
        candidates = self.llm_judge.compare_candidates(query, candidates)
        
        return candidates[:top_k]

