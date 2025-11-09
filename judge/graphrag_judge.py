"""
GraphRAG 기반 불공정 약관 판단 시스템 - 재설계 v4

[핵심 개선사항]
1. Law-Centric 구조와 완전 연결 (조-항-호)
2. 수정본(공정한 약관)과의 거리 비교 추가
3. 점수 배분 재조정 + LLM 최종 판단
4. 위반 퍼센트에 따른 표현 차별화
5. 6조 후순위 유지
"""

from typing import Dict, List, Tuple
from rag.hybrid_graphrag import HybridGraphRAG
from database.neo4j_connector import Neo4jConnector
import numpy as np
import uuid
import json
from pathlib import Path


class GraphRAGJudge:
    """GraphRAG 기반 약관 판단 시스템"""
    
    def __init__(self, rag: HybridGraphRAG, conn: Neo4jConnector):
        self.rag = rag
        self.conn = conn
        
        # 임계값 설정
        self.THRESHOLDS = {
            'high_risk': 0.85,
            'medium_risk': 0.70,
            'low_risk': 0.55,
        }
        
        # 조항 우선순위 (6조 후순위)
        self.ARTICLE_PRIORITY = {
            '제6조': 1,  # 가장 후순위
        }

        self.DEFAULT_PRIORITY = 10 # 다른 조항은 모두 10
        
        # 패턴 JSON 로드
        self._load_patterns()
    
    def _load_patterns(self):
        """patterns_by_article_v2.json 로드"""
        try:
            current_dir = Path(__file__).parent
            pattern_path = current_dir.parent / "data" / "contracts" / "reference" / "patterns_by_article_v2.json"
            
            if not pattern_path.exists():
                alternative_paths = [
                    Path("data/contracts/reference/patterns_by_article_v2.json"),
                    Path("../data/contracts/reference/patterns_by_article_v2.json"),
                ]
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        pattern_path = alt_path
                        break
            
            if pattern_path.exists():
                with open(pattern_path, 'r', encoding='utf-8') as f:
                    self.patterns = json.load(f)
                print(f"✅ 패턴 데이터 로드: {pattern_path}")
            else:
                print(f"⚠️ 패턴 파일 없음: {pattern_path}")
                self.patterns = {}
        except Exception as e:
            print(f"⚠️ 패턴 로드 실패: {e}")
            self.patterns = {}
    
    def judge_clause(self, user_text: str) -> Dict:
        """약관 조항 종합 판단"""
        print(f"\n{'='*70}")
        print(f"🔍 약관 판단 시작")
        print(f"{'='*70}\n")
        print(f"입력: {user_text[:100]}...\n")
        
        # ==================================================================
        # Phase 0: 패턴 기반 사전 분석
        # ==================================================================
        print("📍 Phase 0: 패턴 기반 위험도 분석")
        print("-" * 70)
        
        pattern_analysis = self._analyze_with_patterns(user_text)
        
        print(f"✅ 매칭 키워드: {len(pattern_analysis['matched_keywords'])}개")
        print(f"   위험도: {pattern_analysis['risk_level']}")
        print(f"   패턴 점수: {pattern_analysis['pattern_score']:.3f}\n")
        
        # ==================================================================
        # Phase 0.5: 임시 노드 생성
        # ==================================================================
        print("📍 Phase 0.5: 임시 노드 생성")
        print("-" * 70)
        
        temp_node_id = self._create_temp_node(user_text)
        print(f"✅ 임시 노드: {temp_node_id}\n")
        
        try:
            # ==================================================================
            # Phase 1: 불공정 사례 벡터 검색
            # ==================================================================
            print("📍 Phase 1: 불공정 사례 유사도 검색")
            print("-" * 70)
            
            similar_cases = self.rag.search_similar_cases(
                user_text,
                top_k=10,
                similarity_threshold=0.6
            )
            
            if not similar_cases:
                return self._fallback_judgment_with_patterns(pattern_analysis, user_text)
            
            print(f"✅ {len(similar_cases)}개 유사 사례 발견\n")
            
            # 조항 우선순위 고려 (6조 후순위)
            best_match = self._select_best_match_with_priority(similar_cases)
            best_case_id = best_match['metadata']['id']
            unfair_similarity = best_match['similarity_score']
            
            # ==================================================================
            # Phase 1.5: 공정 사례(수정본) 유사도 검색
            # ==================================================================
            print("📍 Phase 1.5: 공정 사례(수정본) 유사도 검색")
            print("-" * 70)
            
            fair_similarity = self._search_fair_cases(user_text, best_case_id)
            
            print(f"✅ 불공정 유사도: {unfair_similarity:.3f}")
            print(f"✅ 공정 유사도: {fair_similarity:.3f}\n")
            
            # ==================================================================
            # Phase 2: 법률 구조 연결 확인 (Law-Centric)
            # ==================================================================
            print("📍 Phase 2: 법률 구조 분석 (Law-Centric)")
            print("-" * 70)
            
            law_structure_info = self._analyze_law_structure(best_case_id)
            
            print(f"✅ 위반 조항: {law_structure_info['article']}")
            if law_structure_info.get('hang'):
                print(f"   항: {law_structure_info['hang']}")
            if law_structure_info.get('ho'):
                print(f"   호: {law_structure_info['ho']}")
            print(f"   상세: {law_structure_info.get('ho_content', 'N/A')[:100]}...\n")
            
            # ==================================================================
            # Phase 3: 그래프 네트워크 탐색
            # ==================================================================
            print("📍 Phase 3: 그래프 네트워크 탐색")
            print("-" * 70)
            
            neighborhood = self._explore_temp_node_neighborhood(
                temp_node_id,
                similar_cases
            )
            
            print(f"  🕸️ 연결된 노드:")
            print(f"     - 유사 사례: {len(neighborhood['similar_cases'])}개")
            print(f"     - 법률 조항: {len(neighborhood['related_laws'])}개")
            print(f"     - 키워드: {len(neighborhood['keywords'])}개\n")
            
            # ==================================================================
            # Phase 4: 통합 패턴 분석 (빈도 가중치)
            # ==================================================================
            print("📍 Phase 4: 통합 패턴 분석 (빈도 가중치)")
            print("-" * 70)
            
            patterns = self._analyze_violation_patterns_with_frequency(
                user_text,
                similar_cases,
                neighborhood,
                pattern_analysis
            )
            
            print(f"✅ 그래프 패턴 강도: {patterns['strength']:.1%}")
            print(f"✅ 키워드 빈도 상위 3개: {patterns['top_keywords']}\n")
            
            # ==================================================================
            # Phase 5: 그래프 중심성
            # ==================================================================
            print("📍 Phase 5: 그래프 중심성")
            print("-" * 70)
            
            centrality_score = self._calculate_graph_centrality_from_temp(temp_node_id)
            
            print(f"✅ 중심성: {centrality_score:.3f}\n")
            
            # ==================================================================
            # Phase 6: 수식 기반 종합 점수 계산
            # ==================================================================
            print("📍 Phase 6: 수식 기반 종합 점수")
            print("-" * 70)
            
            formula_score = self._calculate_formula_score(
                unfair_similarity=unfair_similarity,
                fair_similarity=fair_similarity,
                pattern_json_score=pattern_analysis['pattern_score'],
                pattern_graph_strength=patterns['strength'],
                centrality=centrality_score
            )
            
            print(f"  수식 점수: {formula_score:.3f}\n")
            
            # ==================================================================
            # Phase 7: LLM 최종 판단 (신규!)
            # ==================================================================
            print("📍 Phase 7: LLM 최종 추론")
            print("-" * 70)
            
            llm_judgment = self._llm_final_judgment(
                user_text=user_text,
                formula_score=formula_score,
                unfair_similarity=unfair_similarity,
                fair_similarity=fair_similarity,
                pattern_analysis=pattern_analysis,
                law_structure_info=law_structure_info,
                best_match=best_match
            )
            
            final_score = llm_judgment['adjusted_score']
            llm_reason = llm_judgment['reasoning']
            
            print(f"✅ LLM 조정 점수: {final_score:.3f}")
            print(f"✅ LLM 추론: {llm_reason[:100]}...\n")
            
            # ==================================================================
            # Phase 8: 위반 여부 및 표현 결정
            # ==================================================================
            print("📍 Phase 8: 최종 판단 및 표현")
            print("-" * 70)
            
            violation, severity = self._determine_violation(final_score)
            confidence_expression = self._get_confidence_expression(final_score)
            
            print(f"  최종 점수: {final_score:.3f}")
            print(f"  판단: {'⚠️ 위반' if violation else '✅ 정상'} (심각도: {severity})")
            print(f"  표현: {confidence_expression}\n")
            
            # ==================================================================
            # Phase 9: 설명 생성
            # ==================================================================
            print("📍 Phase 9: 설명 생성")
            print("-" * 70)
            
            explanation = self._generate_explanation(
                user_text=user_text,
                best_match=best_match,
                neighborhood=neighborhood,
                patterns=patterns,
                final_score=final_score,
                pattern_analysis=pattern_analysis,
                law_structure_info=law_structure_info,
                confidence_expression=confidence_expression
            )
            
            print("✅ 설명 완료\n")
            
            # ==================================================================
            # 최종 결과
            # ==================================================================
            
            result = {
                'violation': violation,
                'severity': severity,
                'confidence': final_score,
                'confidence_expression': confidence_expression,  # 신규!
                
                # 핵심 근거
                'primary_evidence': {
                    'best_match_id': best_case_id,
                    'unfair_similarity': unfair_similarity,
                    'fair_similarity': fair_similarity,
                    'article_id': law_structure_info['article'],
                    'hang': law_structure_info.get('hang'),
                    'ho': law_structure_info.get('ho'),
                },
                
                # 그래프 컨텍스트
                'graph_context': {
                    'similar_cases_count': len(neighborhood['similar_cases']),
                    'related_laws': [law.get('id', '') for law in neighborhood['related_laws']],
                    'keywords': [kw.get('text', '') for kw in neighborhood['keywords']],
                    'centrality_score': centrality_score,
                },
                
                # 패턴 분석
                'patterns': {
                    **patterns,
                    'matched_risk_keywords': pattern_analysis['matched_keywords'],
                    'risk_level_from_patterns': pattern_analysis['risk_level'],
                    'combined_patterns': pattern_analysis.get('combined_patterns', [])
                },
                
                # 법률 구조
                'law_structure': law_structure_info,
                
                # 상세 설명
                'explanation': explanation,
                
                # LLM 판단 (신규!)
                'llm_judgment': {
                    'formula_score': formula_score,
                    'adjusted_score': final_score,
                    'reasoning': llm_reason
                },
                
                # 수정 제안
                'suggestion': self._generate_suggestion(
                    neighborhood['related_laws'],
                    patterns,
                    pattern_analysis
                ),
                
                # 메타데이터
                'method': 'graphrag_v4_with_llm',
                'top_similar_cases': [
                    {
                        'id': case['metadata']['id'],
                        'similarity': case['similarity_score'],
                        'text': case['document'].page_content[:200],
                        'article_id': case['metadata'].get('article_id', 'N/A')
                    }
                    for case in similar_cases[:3]
                ]
            }
            
            print(f"{'='*70}")
            print(f"✅ 판단 완료!")
            print(f"{'='*70}\n")
            
            return result
        
        finally:
            self._delete_temp_node(temp_node_id)
            print(f"🧹 임시 노드 삭제\n")
    
    # ======================================================================
    # 핵심 신규/수정 메서드
    # ======================================================================
    
    def _get_confidence_expression(self, score: float) -> str:
        """
        [신규] 위반 퍼센트에 따른 표현 차별화
        
        요구사항 1번: 무조건 명확하게 판단됐다고 말하지 말 것
        """
        if score >= 0.90:
            return "위반이 명확히 판단됩니다"
        elif score >= 0.80:
            return "위반으로 강하게 추정됩니다"
        elif score >= 0.70:
            return "위반된 것으로 추정됩니다"
        elif score >= 0.60:
            return "위반 가능성이 있는 것으로 판단됩니다"
        elif score >= 0.50:
            return "위반 가능성을 배제할 수 없습니다"
        else:
            return "위반 가능성이 낮은 것으로 판단됩니다"
    
    def _llm_final_judgment(
        self,
        user_text: str,
        formula_score: float,
        unfair_similarity: float,
        fair_similarity: float,
        pattern_analysis: Dict,
        law_structure_info: Dict,
        best_match: Dict
    ) -> Dict:
        """
        [신규] LLM을 통한 최종 판단 및 점수 조정
        
        요구사항 4번: 수식으로 점수 계산 후 LLM이 최종 추론
        
        Returns:
            {
                'adjusted_score': float,  # LLM 조정 후 점수
                'reasoning': str          # LLM 추론 내용
            }
        """
        # 근거 문장 구성
        evidence_parts = []
        
        evidence_parts.append(f"수식 기반 점수: {formula_score:.3f}")
        evidence_parts.append(f"불공정 사례 유사도: {unfair_similarity:.3f}")
        evidence_parts.append(f"공정 사례 유사도: {fair_similarity:.3f}")
        
        if pattern_analysis['matched_keywords']:
            matched_kw = ', '.join([
                f"{kw['keyword']}" for kw in pattern_analysis['matched_keywords'][:5]
            ])
            evidence_parts.append(f"매칭 키워드: {matched_kw}")
        
        evidence_parts.append(f"법조항: {law_structure_info['full_path']}")
        evidence_parts.append(f"유사 사례: {best_match['document'].page_content[:200]}...")
        
        evidence = '\n'.join(evidence_parts)
        
        # LLM 프롬프트
        prompt = f"""당신은 약관 전문가입니다. 다음 약관 조항의 위반 여부를 최종 판단해주세요.

[분석 대상]
{user_text}

[수식 기반 분석 결과]
{evidence}

[임무]
1. 위 증거들을 종합적으로 검토하세요.
2. 특히 다음을 주의깊게 판단하세요:
   - "없습니다", "않습니다"와 같은 부정 표현이 있는지
   - 불공정 사례와 유사하지만 반대 의미인지
   - 문맥상 실제로 위반인지
3. 수식 점수({formula_score:.3f})를 바탕으로, 최종 조정 점수(0.0~1.0)를 제시하세요.

[응답 형식]
조정점수: [0.00~1.00 사이의 숫자]
추론: [2-3문장으로 핵심 근거 설명]

예시:
조정점수: 0.85
추론: 불공정 사례와 높은 유사도를 보이나, '없습니다'라는 명시적 부정 표현이 없어 위반 가능성이 높습니다. 공정 사례와의 거리도 멀어 위반으로 판단됩니다.
"""
        
        try:
            response = self.rag.llm.invoke(prompt)
            content = response.content
            
            # 점수 추출
            import re
            score_match = re.search(r'조정점수:\s*([\d.]+)', content)
            if score_match:
                adjusted_score = float(score_match.group(1))
                adjusted_score = max(0.0, min(1.0, adjusted_score))
            else:
                # 추출 실패 시 수식 점수 사용
                adjusted_score = formula_score
            
            # 추론 추출
            reasoning_match = re.search(r'추론:\s*(.+?)(?:\n\n|\Z)', content, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = content[:200]
            
            return {
                'adjusted_score': adjusted_score,
                'reasoning': reasoning
            }
            
        except Exception as e:
            print(f"⚠️ LLM 판단 실패: {e}")
            return {
                'adjusted_score': formula_score,
                'reasoning': "LLM 추론 실패, 수식 점수 사용"
            }
    
    def _search_fair_cases(self, user_text: str, best_unfair_case_id: str) -> float:
        """공정한 사례(수정본)와의 유사도 계산"""
        query = """
        MATCH (v:ViolationCase {id: $case_id})
        RETURN v.corrected_text as corrected_text
        """
        
        result = self.conn.execute_query(query, {'case_id': best_unfair_case_id})
        
        if not result or not result[0].get('corrected_text'):
            return 0.0
        
        corrected_text = result[0]['corrected_text']
        
        if not corrected_text or corrected_text.strip() == '':
            return 0.0
        
        # 사용자 입력과 수정본 임베딩
        user_emb = self.rag.local_embeddings.encode([user_text])[0]
        corrected_emb = self.rag.local_embeddings.encode([corrected_text])[0]
        
        # 코사인 유사도
        similarity = np.dot(user_emb, corrected_emb) / (
            np.linalg.norm(user_emb) * np.linalg.norm(corrected_emb)
        )
        
        return float(similarity)
    
    def _analyze_law_structure(self, case_id: str) -> Dict:
        """
        Law-Centric 구조 분석: ViolationCase → 호/항/조 노드
        
        Returns:
            {
                'article': '제7조',
                'article_title': '면책조항의 금지',
                'hang': '제1항' (있으면),
                'ho': '제2호' (있으면),
                'ho_content': '호의 내용',
                'full_path': '제7조 제2호'
            }
        """
        # ViolationCase → 호/항/조 관계 조회
        query = """
        MATCH (v:ViolationCase {id: $case_id})-[:VIOLATES]->(target)
        WHERE target:호 OR target:항 OR target:조
        
        OPTIONAL MATCH (target)<-[:HAS_HO]-(parent_hang:항)
        OPTIONAL MATCH (target)<-[:HAS_HANG|HAS_HO]-(parent_article:조)
        OPTIONAL MATCH (parent_hang)<-[:HAS_HANG]-(article_from_hang:조)
        
        RETURN 
            target.id as target_id,
            target.content as target_content,
            labels(target)[0] as target_type,
            parent_article.id as article_id,
            parent_article.title as article_title,
            article_from_hang.id as article_id_alt,
            article_from_hang.title as article_title_alt,
            parent_hang.id as hang_id
        LIMIT 1
        """
        
        result = self.conn.execute_query(query, {'case_id': case_id})
        
        if not result:
            # 폴백: metadata에서 article_id 가져오기
            fallback_query = """
            MATCH (v:ViolationCase {id: $case_id})
            RETURN v.article_id as article_id
            """
            fallback_result = self.conn.execute_query(fallback_query, {'case_id': case_id})
            
            if fallback_result and fallback_result[0].get('article_id'):
                return {
                    'article': fallback_result[0]['article_id'],
                    'article_title': '',
                    'full_path': fallback_result[0]['article_id']
                }
            
            return {
                'article': 'Unknown',
                'article_title': '',
                'full_path': 'Unknown'
            }
        
        data = result[0]
        target_type = data.get('target_type', '')
        target_id = data.get('target_id', '')
        target_content = data.get('target_content', '')
        
        # 조 노드 결정
        article_id = data.get('article_id') or data.get('article_id_alt', 'Unknown')
        article_title = data.get('article_title') or data.get('article_title_alt', '')
        
        # 항/호 정보 파싱
        hang_id = None
        ho_id = None
        ho_content = target_content
        
        if target_type == '호':
            ho_id = target_id
            hang_id = data.get('hang_id')
        elif target_type == '항':
            hang_id = target_id
        
        # 경로 구성
        path_parts = [article_id]
        if hang_id:
            path_parts.append(hang_id.split('_')[-1] if '_' in hang_id else hang_id)
        if ho_id:
            path_parts.append(ho_id.split('_')[-1] if '_' in ho_id else ho_id)
        
        full_path = ' '.join(path_parts)
        
        return {
            'article': article_id,
            'article_title': article_title,
            'hang': hang_id,
            'ho': ho_id,
            'ho_content': ho_content,
            'full_path': full_path
        }
    
    def _calculate_formula_score(
        self,
        unfair_similarity: float,
        fair_similarity: float,
        pattern_json_score: float,
        pattern_graph_strength: float,
        centrality: float
    ) -> float:
        """
        수식 기반 점수 계산
        
        점수 배분:
        - 불공정 유사도: 40% (핵심 지표)
        - 패턴 JSON: 30% (명시적 위험 키워드)
        - 공정 유사도: 15% (역방향, 높을수록 감점)
        - 그래프 중심성: 10%
        - 패턴 그래프: 5% (보조)
        """
        
        weights = {
            'unfair': 0.40,
            'pattern_json': 0.30,
            'fair': 0.15,
            'centrality': 0.10,
            'pattern_graph': 0.05
        }
        
        # 공정 유사도는 역방향 (높을수록 감점)
        fair_penalty = (1.0 - fair_similarity)
        
        formula_score = (
            unfair_similarity * weights['unfair'] +
            pattern_json_score * weights['pattern_json'] +
            fair_penalty * weights['fair'] +
            centrality * weights['centrality'] +
            pattern_graph_strength * weights['pattern_graph']
        )
        
        print(f"  점수 구성:")
        print(f"    - 불공정 유사도 ({weights['unfair']:.0%}): {unfair_similarity:.3f} → {unfair_similarity * weights['unfair']:.3f}")
        print(f"    - JSON 패턴 ({weights['pattern_json']:.0%}): {pattern_json_score:.3f} → {pattern_json_score * weights['pattern_json']:.3f}")
        print(f"    - 공정 페널티 ({weights['fair']:.0%}): {fair_similarity:.3f} → {fair_penalty * weights['fair']:.3f}")
        print(f"    - 그래프 중심성 ({weights['centrality']:.0%}): {centrality:.3f} → {centrality * weights['centrality']:.3f}")
        print(f"    - 그래프 패턴 ({weights['pattern_graph']:.0%}): {pattern_graph_strength:.3f} → {pattern_graph_strength * weights['pattern_graph']:.3f}")
        
        return formula_score
    
    def _analyze_violation_patterns_with_frequency(
        self,
        user_text: str,
        similar_cases: List[Dict],
        neighborhood: Dict,
        pattern_analysis: Dict
    ) -> Dict:
        """
        [수정] 패턴 분석 - 빈도 기반 가중치 반영
        
        요구사항 3번: 키워드 단순 출현이 아닌 빈도로 가중치
        """
        json_keywords = {kw['keyword']: 3 for kw in pattern_analysis['matched_keywords']}
        
        # 그래프에서 키워드 빈도 집계
        keyword_counts = {}
        for kw in neighborhood.get('keywords', []):
            if kw:
                keyword_text = kw.get('text', '')
                frequency = kw.get('frequency', 1)  # Keyword 노드의 frequency 속성
                if keyword_text:
                    keyword_counts[keyword_text] = frequency
        
        # 통합 빈도 계산
        all_keywords = {}
        for kw, weight in json_keywords.items():
            all_keywords[kw] = all_keywords.get(kw, 0) + weight
        for kw, freq in keyword_counts.items():
            all_keywords[kw] = all_keywords.get(kw, 0) + freq
        
        # 빈도 기준 정렬
        common_keywords = sorted(
            all_keywords.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 상위 10개
        top_keywords = common_keywords[:10]
        
        # 패턴 강도: 빈도 가중치 반영
        if top_keywords:
            total_frequency = sum(freq for _, freq in top_keywords)
            pattern_strength = min(total_frequency / 30.0, 1.0)  # 30회 이상 = 1.0
        else:
            pattern_strength = pattern_analysis['pattern_score']
        
        return {
            'common_keywords': [kw for kw, _ in top_keywords],
            'keyword_frequencies': dict(top_keywords),
            'top_keywords': [(kw, freq) for kw, freq in top_keywords[:3]],
            'strength': pattern_strength,
            'pattern_consistency': 0.8
        }
    
    # ======================================================================
    # 기존 메서드들 (변경 없음 또는 미세 조정)
    # ======================================================================
    
    def _analyze_with_patterns(self, user_text: str) -> Dict:
        """patterns_by_article_v2.json 패턴 분석"""
        if not self.patterns:
            return {
                'matched_keywords': [],
                'risk_level': 'unknown',
                'pattern_score': 0.0,
                'article_hints': [],
                'combined_patterns': []
            }
        
        matched_keywords = []
        article_scores = {}
        combined_patterns_found = []
        
        # 1. 범용 위험 키워드
        universal = self.patterns.get('universal_risk_keywords', {})
        if 'keywords' in universal:
            for kw_info in universal['keywords']:
                keyword = kw_info['keyword']
                if keyword in user_text:
                    matched_keywords.append({
                        'keyword': keyword,
                        'risk_level': kw_info['risk_level'],
                        'description': kw_info['description'],
                        'article': 'universal'
                    })
        
        # 2. 조항별 패턴
        for article_id in ['제6조', '제7조', '제8조', '제9조', '제10조', 
                          '제11조', '제12조', '제13조', '제14조']:
            
            if article_id not in self.patterns:
                continue
            
            article_data = self.patterns[article_id]
            article_score = 0.0
            
            for pattern in article_data.get('patterns', []):
                high_risk_matched = 0
                for kw in pattern.get('high_risk_keywords', []):
                    if kw in user_text:
                        high_risk_matched += 1
                        matched_keywords.append({
                            'keyword': kw,
                            'risk_level': pattern.get('risk_level', 'high'),
                            'description': pattern.get('description', ''),
                            'article': article_id
                        })
                
                normal_matched = 0
                for kw in pattern.get('keywords', []):
                    if kw in user_text:
                        normal_matched += 1
                
                if high_risk_matched > 0:
                    article_score += high_risk_matched * 0.3
                if normal_matched > 0:
                    article_score += normal_matched * 0.1
            
            if article_score > 0:
                article_scores[article_id] = article_score
        
        # 3. 복합 패턴
        combined = self.patterns.get('combined_pattern_risks', {})
        if 'patterns' in combined:
            for pattern in combined['patterns']:
                keywords = pattern['combination']
                if all(kw in user_text for kw in keywords):
                    combined_patterns_found.append({
                        'keywords': keywords,
                        'risk_level': pattern['risk_level'],
                        'description': pattern['description'],
                        'articles': pattern['articles']
                    })
        
        # 4. 위험도 결정
        risk_scores = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for kw in matched_keywords:
            risk_level = kw.get('risk_level', 'low')
            if risk_level in risk_scores:
                risk_scores[risk_level] += 1
        
        for cp in combined_patterns_found:
            risk_level = cp.get('risk_level', 'high')
            if risk_level in risk_scores:
                risk_scores[risk_level] += 2
        
        if risk_scores['critical'] > 0:
            final_risk_level = 'critical'
        elif risk_scores['high'] > 0:
            final_risk_level = 'high'
        elif risk_scores['medium'] > 0:
            final_risk_level = 'medium'
        elif risk_scores['low'] > 0:
            final_risk_level = 'low'
        else:
            final_risk_level = 'none'
        
        # 5. 패턴 점수
        total_matches = sum(risk_scores.values())
        pattern_score = min(total_matches / 10.0, 1.0)
        
        risk_multipliers = {
            'critical': 1.5,
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8
        }
        pattern_score *= risk_multipliers.get(final_risk_level, 1.0)
        pattern_score = min(pattern_score, 1.0)
        
        # 6. 조항 힌트
        article_hints = sorted(
            article_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            'matched_keywords': matched_keywords,
            'risk_level': final_risk_level,
            'pattern_score': pattern_score,
            'article_hints': [article for article, _ in article_hints],
            'combined_patterns': combined_patterns_found
        }
    
    def _fallback_judgment_with_patterns(self, pattern_analysis: Dict, user_text: str) -> Dict:
        """유사 사례 없을 때 패턴만으로 판단"""
        pattern_score = pattern_analysis['pattern_score']
        risk_level = pattern_analysis['risk_level']
        
        violation = risk_level in ['critical', 'high']
        
        severity_map = {
            'critical': 'high',
            'high': 'medium',
            'medium': 'low',
            'low': 'none',
            'none': 'none'
        }
        severity = severity_map.get(risk_level, 'none')
        confidence_expression = self._get_confidence_expression(pattern_score)
        
        return {
            'violation': violation,
            'severity': severity,
            'confidence': pattern_score,
            'confidence_expression': confidence_expression,
            'reason': f'패턴 분석: {len(pattern_analysis["matched_keywords"])}개 위험 키워드',
            'method': 'pattern_only',
            'patterns': {
                'matched_risk_keywords': pattern_analysis['matched_keywords'],
                'risk_level_from_patterns': risk_level,
                'combined_patterns': pattern_analysis.get('combined_patterns', []),
            },
            'explanation': f"{confidence_expression}. 유사 사례가 없어 패턴 분석만 수행했습니다."
        }
    
    def _create_temp_node(self, user_text: str) -> str:
        """임시 노드 생성"""
        temp_id = f"TEMP_{uuid.uuid4().hex[:8]}"
        
        embedding = self.rag.local_embeddings.encode([user_text])[0].tolist()
        
        query = """
        CREATE (t:TempNode {
            id: $id,
            text: $text,
            embedding: $embedding,
            created_at: timestamp()
        })
        RETURN t.id as id
        """
        
        self.conn.execute_query(query, {
            'id': temp_id,
            'text': user_text,
            'embedding': embedding
        })
        
        return temp_id
    
    def _delete_temp_node(self, temp_id: str):
        """임시 노드 삭제"""
        query = """
        MATCH (t:TempNode {id: $id})
        DETACH DELETE t
        """
        self.conn.execute_query(query, {'id': temp_id})
    
    def _explore_temp_node_neighborhood(self, temp_node_id: str, similar_cases: List[Dict]) -> Dict:
        """임시 노드 주변 탐색"""
        # 유사 사례 연결
        for case in similar_cases[:5]:
            case_id = case['metadata']['id']
            similarity = case['similarity_score']
            
            query = """
            MATCH (t:TempNode {id: $temp_id})
            MATCH (v:ViolationCase {id: $case_id})
            CREATE (t)-[:TEMP_SIMILAR {similarity: $similarity}]->(v)
            """
            
            self.conn.execute_query(query, {
                'temp_id': temp_node_id,
                'case_id': case_id,
                'similarity': similarity
            })
        
        # 주변 탐색
        query = """
        MATCH (t:TempNode {id: $temp_id})-[:TEMP_SIMILAR]->(v:ViolationCase)
        
        OPTIONAL MATCH (v)-[:VIOLATES]->(law)
        WHERE law:호 OR law:항 OR law:조
        
        OPTIONAL MATCH (v)-[:CONTAINS]->(kw:Keyword)
        
        RETURN 
            collect(DISTINCT v) as cases,
            collect(DISTINCT law) as laws,
            collect(DISTINCT kw) as keywords
        """
        
        result = self.conn.execute_query(query, {'temp_id': temp_node_id})
        
        if not result:
            return {
                'similar_cases': [],
                'related_laws': [],
                'keywords': []
            }
        
        data = result[0]
        
        return {
            'similar_cases': [dict(c) for c in data.get('cases', []) if c],
            'related_laws': [dict(l) for l in data.get('laws', []) if l],
            'keywords': [dict(k) for k in data.get('keywords', []) if k]
        }
    
    def _calculate_graph_centrality_from_temp(self, temp_node_id: str) -> float:
        """임시 노드 그래프 중심성"""
        query = """
        MATCH (t:TempNode {id: $temp_id})
        OPTIONAL MATCH (t)-[r:TEMP_SIMILAR]->()
        
        WITH t, count(r) as direct_connections
        
        OPTIONAL MATCH (t)-[:TEMP_SIMILAR]->(v1:ViolationCase)
                         -[:SIMILAR_TO]->(v2:ViolationCase)
        
        WITH direct_connections, count(DISTINCT v2) as indirect_connections
        
        RETURN 
            direct_connections,
            indirect_connections,
            direct_connections + indirect_connections * 0.5 as total_centrality
        """
        
        result = self.conn.execute_query(query, {'temp_id': temp_node_id})
        
        if not result or not result[0]:
            return 0.0
        
        data = result[0]
        total = data.get('total_centrality', 0)
        normalized = min(total / 10.0, 1.0)
        
        print(f"  직접 연결: {data.get('direct_connections', 0)}개")
        print(f"  간접 연결: {data.get('indirect_connections', 0)}개")
        
        return normalized
    
    def _select_best_match_with_priority(self, similar_cases: List[Dict]) -> Dict:
        """조항 우선순위 고려 (6조 후순위)"""
        if not similar_cases:
            return None
        
        scored_cases = []
        max_priority = max(self.ARTICLE_PRIORITY.values())
        
        for case in similar_cases:
            article_id = case['metadata'].get('article_id', '')
            priority = self.ARTICLE_PRIORITY.get(article_id, 0)
            priority_score = priority / max_priority
            combined_score = (case['similarity_score'] * 0.6) + (priority_score * 0.4)
            
            scored_cases.append({
                'case': case,
                'combined_score': combined_score,
                'article_id': article_id,
                'priority': priority
            })
        
        scored_cases.sort(key=lambda x: x['combined_score'], reverse=True)
        best = scored_cases[0]
        
        print(f"  🎯 선택된 조항: {best['article_id']}")
        print(f"     유사도: {best['case']['similarity_score']:.3f}")
        print(f"     우선순위: {best['priority']} (6조=1, 8조=10)")
        
        if best['article_id'] == '제6조':
            print(f"     ⚠️ 6조 선택 (포괄적)")
        
        print()
        
        return best['case']
    
    def _determine_violation(self, score: float) -> Tuple[bool, str]:
        """위반 여부 결정"""
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
        final_score: float,
        pattern_analysis: Dict,
        law_structure_info: Dict,
        confidence_expression: str
    ) -> str:
        """LLM 설명 생성"""
        context_parts = []
        
        context_parts.append(f"판단: {confidence_expression}")
        
        if pattern_analysis['matched_keywords']:
            matched_kw = ', '.join([
                f"{kw['keyword']}({kw['risk_level']})"
                for kw in pattern_analysis['matched_keywords'][:5]
            ])
            context_parts.append(f"위험 키워드: {matched_kw}")
        
        context_parts.append(f"\n위반 조항: {law_structure_info['full_path']}")
        context_parts.append(f"조항 내용: {law_structure_info.get('ho_content', 'N/A')[:150]}")
        context_parts.append(f"\n유사 사례:\n{best_match['document'].page_content[:300]}")
        
        context = '\n'.join(context_parts)
        
        prompt = f"""
다음 약관을 분석했습니다:

[검토 대상]
{user_text}

[분석 결과]
{context}

[판단 점수]
{final_score:.2f} / 1.00

다음 형식으로 설명:
1. **위반 여부**: {confidence_expression}
2. **문제점**: 위험 키워드 언급
3. **법적 근거**: {law_structure_info['full_path']}
4. **유사 사례**: 공통점

각 2-3문장으로.
        """
        
        try:
            response = self.rag.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"{confidence_expression}. 패턴 분석: {len(pattern_analysis['matched_keywords'])}개 위험 키워드"
    
    def _generate_suggestion(
        self,
        related_laws: List[Dict],
        patterns: Dict,
        pattern_analysis: Dict
    ) -> str:
        """수정 제안"""
        law_ids = [law.get('id', '') for law in related_laws[:2]] if related_laws else []
        keywords = patterns.get('common_keywords', [])[:3]
        risk_keywords = [kw['keyword'] for kw in pattern_analysis['matched_keywords'][:3]]
        
        prompt = f"""
약관 수정 제안:

법조항: {', '.join(law_ids) if law_ids else '패턴 기반'}
문제 패턴: {', '.join(keywords)}
위험 키워드: {', '.join(risk_keywords)}

3-4문장으로 수정 방향 제시.
        """
        
        try:
            response = self.rag.llm.invoke(prompt)
            return response.content
        except Exception as e:
            if risk_keywords:
                return f"위험 키워드 제거/완화: {', '.join(risk_keywords)}"
            return "고의·중과실 책임 명시, 불가항력 구체화"


if __name__ == "__main__":
    import os
    from database.neo4j_connector import Neo4jConnector
    from rag.hybrid_graphrag import HybridGraphRAG
    
    conn = Neo4jConnector()
    rag = HybridGraphRAG(
        driver=conn.driver,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    judge = GraphRAGJudge(rag, conn)
    
    result = judge.judge_clause("회사는 어떠한 경우에도 책임을 지지 않으며, 고객이 모든 손해를 부담합니다.")
    
    print(f"\n위반: {result['violation']}")
    print(f"표현: {result['confidence_expression']}")
    print(f"확신도: {result['confidence']:.3f}")
    print(f"위반 조항: {result['law_structure']['full_path']}")
    print(f"설명: {result['explanation']}")
    
    conn.close()