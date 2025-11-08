"""
GraphRAG 기반 불공정 약관 판단 시스템

진짜 GraphRAG의 힘:
1. 유사 사례 네트워크 탐색
2. 다단계 추론
3. 패턴 발견
4. 맥락 기반 판단
"""

from typing import Dict, List, Tuple
from rag.hybrid_graphrag import HybridGraphRAG
from database.neo4j_connector import Neo4jConnector
import numpy as np


class GraphRAGJudge:
    """GraphRAG 기반 약관 판단 시스템"""
    
    def __init__(self, rag: HybridGraphRAG, conn: Neo4jConnector):
        self.rag = rag
        self.conn = conn
        
        # 임계값 설정
        self.THRESHOLDS = {
            'high_risk': 0.85,      # 명백한 위반
            'medium_risk': 0.70,    # 주의 필요
            'low_risk': 0.55,       # 검토 권장
        }
    
    def judge_clause(self, user_text: str) -> Dict:
        """
        약관 조항 종합 판단
        
        Args:
            user_text: 판단할 약관 텍스트
            
        Returns:
            종합 판단 결과
        """
        print(f"\n{'='*70}")
        print(f"🔍 약관 판단 시작")
        print(f"{'='*70}\n")
        print(f"입력: {user_text[:100]}...\n")
        
        # ==================================================================
        # Phase 1: 초기 벡터 검색
        # ==================================================================
        print("📍 Phase 1: 유사 사례 검색")
        print("-" * 70)
        
        similar_cases = self.rag.search_similar_cases(
            user_text,
            top_k=10,
            similarity_threshold=0.6
        )
        
        if not similar_cases:
            return {
                'violation': False,
                'confidence': 0.0,
                'reason': '유사 위반 사례를 찾을 수 없습니다.',
                'method': 'graphrag'
            }
        
        print(f"✅ {len(similar_cases)}개 유사 사례 발견\n")
        
        # 가장 유사한 사례
        best_match = similar_cases[0]
        best_case_id = best_match['metadata']['id']
        base_similarity = best_match['similarity_score']
        
        # ==================================================================
        # Phase 2: 그래프 네트워크 탐색 (핵심!)
        # ==================================================================
        print("📍 Phase 2: 그래프 네트워크 탐색")
        print("-" * 70)
        
        # 이웃 노드 탐색
        neighborhood = self.rag.explore_graph_neighborhood(
            best_case_id,
            max_depth=2
        )
        
        print(f"  🕸️ 연결된 노드:")
        print(f"     - 유사 사례: {len(neighborhood['similar_cases'])}개")
        print(f"     - 법률 조항: {len(neighborhood['related_laws'])}개")
        print(f"     - 키워드: {len(neighborhood['keywords'])}개")
        print(f"     - 위반 유형: {len(neighborhood['violation_types'])}개\n")
        
        # ==================================================================
        # Phase 3: 패턴 분석
        # ==================================================================
        print("📍 Phase 3: 위반 패턴 분석")
        print("-" * 70)
        
        patterns = self._analyze_violation_patterns(
            user_text,
            similar_cases,
            neighborhood
        )
        
        print(f"✅ {len(patterns['common_keywords'])}개 공통 패턴 발견\n")
        
        # ==================================================================
        # Phase 4: 다단계 추론
        # ==================================================================
        print("📍 Phase 4: 다단계 추론")
        print("-" * 70)
        
        reasoning_result = self.rag.multi_hop_reasoning(
            user_text,
            max_hops=2
        )
        
        print(f"✅ {len(reasoning_result.get('reasoning_paths', []))}개 추론 경로 발견\n")
        
        # ==================================================================
        # Phase 5: 그래프 중심성 점수
        # ==================================================================
        print("📍 Phase 5: 그래프 중심성 분석")
        print("-" * 70)
        
        centrality_score = self._calculate_graph_centrality(similar_cases)
        
        print(f"✅ 그래프 중심성 점수: {centrality_score:.3f}\n")
        
        # ==================================================================
        # Phase 6: 종합 점수 계산
        # ==================================================================
        print("📍 Phase 6: 종합 판단")
        print("-" * 70)
        
        final_score = self._calculate_final_score(
            base_similarity=base_similarity,
            pattern_strength=patterns['strength'],
            centrality=centrality_score,
            reasoning_confidence=len(reasoning_result.get('reasoning_paths', [])) / 20.0
        )
        
        # 위반 여부 결정
        violation, severity = self._determine_violation(final_score)
        
        print(f"  최종 점수: {final_score:.3f}")
        print(f"  판단: {'⚠️ 위반' if violation else '✅ 정상'} (심각도: {severity})\n")
        
        # ==================================================================
        # Phase 7: 설명 생성
        # ==================================================================
        print("📍 Phase 7: 설명 생성")
        print("-" * 70)
        
        explanation = self._generate_explanation(
            user_text=user_text,
            best_match=best_match,
            neighborhood=neighborhood,
            patterns=patterns,
            reasoning=reasoning_result,
            final_score=final_score
        )
        
        print("✅ 설명 생성 완료\n")
        
        # ==================================================================
        # 최종 결과 반환
        # ==================================================================
        
        result = {
            'violation': violation,
            'severity': severity,
            'confidence': final_score,
            
            # 핵심 근거
            'primary_evidence': {
                'best_match_id': best_case_id,
                'similarity': base_similarity,
                'article_id': best_match['metadata'].get('article_id'),
            },
            
            # 그래프 컨텍스트
            'graph_context': {
                'similar_cases_count': len(neighborhood['similar_cases']),
                'related_laws': [law.get('id', '') for law in neighborhood['related_laws']],
                'keywords': [kw.get('text', '') for kw in neighborhood['keywords']],
                'centrality_score': centrality_score,
            },
            
            # 패턴 분석
            'patterns': patterns,
            
            # 추론 경로
            'reasoning_paths': len(reasoning_result.get('reasoning_paths', [])),
            
            # 상세 설명
            'explanation': explanation,
            
            # 수정 제안
            'suggestion': self._generate_suggestion(
                neighborhood['related_laws'],
                patterns
            ),
            
            # 메타데이터
            'method': 'graphrag',
            'top_similar_cases': [
                {
                    'id': case['metadata']['id'],
                    'similarity': case['similarity_score'],
                    'text': case['document'].page_content[:200]
                }
                for case in similar_cases[:3]
            ]
        }
        
        print(f"{'='*70}")
        print(f"✅ 판단 완료!")
        print(f"{'='*70}\n")
        
        return result
    
    # ======================================================================
    # 헬퍼 메서드
    # ======================================================================
    
    def _analyze_violation_patterns(
        self,
        user_text: str,
        similar_cases: List[Dict],
        neighborhood: Dict
    ) -> Dict:
        """위반 패턴 분석"""
        
        # 공통 키워드 추출
        keyword_counts = {}
        for kw in neighborhood['keywords']:
            keyword_text = kw.get('text', '')
            keyword_counts[keyword_text] = keyword_counts.get(keyword_text, 0) + 1
        
        # 빈도 기준 정렬
        common_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # 패턴 강도 계산
        pattern_strength = min(sum(count for _, count in common_keywords) / 10.0, 1.0)
        
        # 위반 유형 분포
        violation_types = {}
        for vtype in neighborhood['violation_types']:
            type_name = vtype.get('name', '')
            violation_types[type_name] = violation_types.get(type_name, 0) + 1
        
        return {
            'common_keywords': [kw for kw, _ in common_keywords],
            'keyword_frequencies': dict(common_keywords),
            'strength': pattern_strength,
            'violation_types': violation_types,
            'pattern_consistency': len(common_keywords) / max(len(similar_cases), 1)
        }
    
    def _calculate_graph_centrality(self, similar_cases: List[Dict]) -> float:
        """그래프 중심성 점수"""
        
        if not similar_cases:
            return 0.0
        
        total_centrality = 0.0
        
        for case in similar_cases[:5]:  # 상위 5개만
            case_id = case['metadata']['id']
            
            # 노드의 연결 정도 조회
            query = """
            MATCH (v:ViolationCase {id: $case_id})
            OPTIONAL MATCH (v)-[r]-()
            RETURN count(r) as degree
            """
            
            result = self.conn.execute_query(query, {'case_id': case_id})
            
            if result:
                degree = result[0]['degree']
                # 정규화 (0~1)
                total_centrality += min(degree / 20.0, 1.0)
        
        return total_centrality / min(len(similar_cases), 5)
    
    def _calculate_final_score(
        self,
        base_similarity: float,
        pattern_strength: float,
        centrality: float,
        reasoning_confidence: float
    ) -> float:
        """최종 점수 계산"""
        
        # 가중 평균
        weights = {
            'similarity': 0.4,      # 벡터 유사도
            'pattern': 0.3,         # 패턴 강도
            'centrality': 0.2,      # 그래프 중심성
            'reasoning': 0.1        # 추론 확신도
        }
        
        final_score = (
            base_similarity * weights['similarity'] +
            pattern_strength * weights['pattern'] +
            centrality * weights['centrality'] +
            reasoning_confidence * weights['reasoning']
        )
        
        return final_score
    
    def _determine_violation(self, score: float) -> Tuple[bool, str]:
        """위반 여부 및 심각도 결정"""
        
        if score >= self.THRESHOLDS['high_risk']:
            return True, 'high'
        elif score >= self.THRESHOLDS['medium_risk']:
            return True, 'medium'
        elif score >= self.THRESHOLDS['low_risk']:
            return True, 'low'
        else:
            return False, 'none'
    
    def _generate_explanation(
        self,
        user_text: str,
        best_match: Dict,
        neighborhood: Dict,
        patterns: Dict,
        reasoning: Dict,
        final_score: float
    ) -> str:
        """LLM 기반 설명 생성"""
        
        # 컨텍스트 구성
        context_parts = []
        
        # 1. 가장 유사한 사례
        context_parts.append(f"가장 유사한 위반 사례:\n{best_match['document'].page_content[:300]}")
        
        # 2. 관련 법조항
        if neighborhood['related_laws']:
            laws = [law.get('id', '') for law in neighborhood['related_laws'][:3]]
            context_parts.append(f"\n관련 법조항: {', '.join(laws)}")
        
        # 3. 공통 패턴
        if patterns['common_keywords']:
            context_parts.append(f"\n공통 위반 패턴: {', '.join(patterns['common_keywords'])}")
        
        # 4. 추론 분석
        if 'analysis' in reasoning:
            context_parts.append(f"\n추론 분석:\n{reasoning['analysis']}")
        
        context = '\n'.join(context_parts)
        
        # LLM 프롬프트
        prompt = f"""
다음 약관 조항을 분석했습니다:

[검토 대상 약관]
{user_text}

[분석 컨텍스트]
{context}

[판단 점수]
{final_score:.2f} / 1.00

위 정보를 바탕으로 다음 형식으로 설명하세요:

1. **위반 여부**: 명확하게 판단
2. **문제점**: 구체적으로 무엇이 문제인지
3. **법적 근거**: 어떤 법을 위반했는지
4. **유사 사례**: 비슷한 사례와의 공통점

각 항목을 2-3문장으로 간결하게 작성하세요.
        """
        
        try:
            response = self.rag.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"⚠️ 설명 생성 실패: {e}")
            return "설명을 생성할 수 없습니다."
    
    def _generate_suggestion(
        self,
        related_laws: List[Dict],
        patterns: Dict
    ) -> str:
        """수정 제안 생성"""
        
        if not related_laws:
            return "구체적인 수정 제안을 생성할 수 없습니다."
        
        law_ids = [law.get('id', '') for law in related_laws[:2]]
        keywords = patterns['common_keywords'][:3]
        
        prompt = f"""
다음 법조항과 패턴을 고려하여 약관 수정 제안을 작성하세요:

관련 법조항: {', '.join(law_ids)}
문제 패턴: {', '.join(keywords)}

3-4문장으로 구체적인 수정 방향을 제시하세요.
        """
        
        try:
            response = self.rag.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return "고의·중과실에 대한 책임을 명시하고, 불가항력 사유를 구체적으로 한정하세요."


# =============================================================================
# 실행 예시
# =============================================================================

if __name__ == "__main__":
    import os
    
    # 초기화
    conn = Neo4jConnector()
    rag = HybridGraphRAG(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    judge = GraphRAGJudge(rag, conn)
    
    # 테스트
    result = judge.judge_clause("회사는 어떠한 경우에도 책임을 지지 않습니다.")
    
    print(f"\n위반: {result['violation']}")
    print(f"확신도: {result['confidence']:.3f}")
    print(f"설명: {result['explanation']}")
    
    conn.close()