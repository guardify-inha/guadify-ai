"""
GraphRAG 기반 불공정 약관 판단 시스템 - v6 (LLM 역할 수정)

[핵심 변경사항]
1. ✅ 상대적 거리 기반 불공정도 계산 (Phase 1.5 개선!)
2. ✅ Keyword 노드 구조 변경 반영 (frequency → case_count, prevalence)
3. ✅ severity 제거 반영 (실시간 계산으로 변경)
4. ✅ 정규표현식 키워드 매칭 지원
5. ✅ 중주제/소주제 계층 구조 지원
6. Law-Centric 구조와 완전 연결 (조-항-호)
7. 점수 배분 재조정 + LLM 의미 반전 검증 ← 수정!
8. 위반 퍼센트에 따른 표현 차별화
9. 6조 후순위 유지
"""

from typing import Dict, List, Tuple
from rag.hybrid_graphrag import HybridGraphRAG
from database.neo4j_connector import Neo4jConnector
import numpy as np
import uuid
import json
from pathlib import Path
import re


class GraphRAGJudge:
    """GraphRAG 기반 약관 판단 시스템 v6"""
    
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

        self.DEFAULT_PRIORITY = 10  # 다른 조항은 모두 10
        
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
        # Phase 0: 패턴 기반 사전 분석 (정규표현식 포함!)
        # ==================================================================
        print("📍 Phase 0: 패턴 기반 위험도 분석 (정규표현식 포함)")
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
            # Phase 1.5: 상대적 거리 기반 불공정도 계산 (개선!)
            # ==================================================================
            print("📍 Phase 1.5: 상대적 거리 기반 불공정도 계산")
            print("-" * 70)
            
            relative_unfairness = self._calculate_relative_unfairness(
                user_text,
                best_case_id,
                unfair_similarity
            )
            
            print(f"✅ 위반사례 유사도: {unfair_similarity:.3f}")
            print(f"✅ 수정본 유사도: {relative_unfairness['fair_similarity']:.3f}")
            print(f"✅ 위반사례 거리: {relative_unfairness['unfair_distance']:.3f}")
            print(f"✅ 수정본 거리: {relative_unfairness['fair_distance']:.3f}")
            print(f"✅ 상대적 불공정도: {relative_unfairness['unfairness_score']:.3f}")
            print(f"   해석: 입력 문장이 위반사례에 {'가깝습니다' if relative_unfairness['unfairness_score'] > 0.5 else '멀고, 수정본에 가깝습니다'}\n")
            
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
            # Phase 3: 그래프 네트워크 탐색 (변경: Keyword 노드 구조)
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
            # Phase 4: 통합 패턴 분석 (변경: case_count 기반)
            # ==================================================================
            print("📍 Phase 4: 통합 패턴 분석 (사례 수 기반)")
            print("-" * 70)
            
            patterns = self._analyze_violation_patterns_with_case_count(
                user_text,
                similar_cases,
                neighborhood,
                pattern_analysis
            )
            
            print(f"✅ 그래프 패턴 강도: {patterns['strength']:.1%}")
            print(f"✅ 키워드 상위 3개: {patterns['top_keywords']}\n")
            
            # ==================================================================
            # Phase 4.5: 실시간 severity 계산 (신규!)
            # ==================================================================
            print("📍 Phase 4.5: 실시간 심각도 계산 (Keyword 기반)")
            print("-" * 70)
            
            computed_severity = self._compute_severity_from_keywords(
                neighborhood['keywords'],
                pattern_analysis['matched_keywords']
            )
            
            print(f"✅ 계산된 심각도: {computed_severity}\n")
            
            # ==================================================================
            # Phase 6: 수식 기반 종합 점수 계산 (개선!)
            # ==================================================================
            print("📍 Phase 6: 수식 기반 종합 점수")
            print("-" * 70)
            
            formula_score = self._calculate_formula_score_v2(
                unfair_similarity=unfair_similarity,
                relative_unfairness=relative_unfairness['unfairness_score'],
                pattern_json_score=pattern_analysis['pattern_score'],
                pattern_graph_strength=patterns['strength']
            )
            
            print(f"  수식 점수: {formula_score:.3f}\n")
            
            # ==================================================================
            # Phase 7: LLM 의미 반전 검증 (수정!)
            # ==================================================================
            print("📍 Phase 7: LLM 의미 반전 검증")
            print("-" * 70)
            
            llm_judgment = self._llm_semantic_reversal_check(
                user_text=user_text,
                formula_score=formula_score,
                unfair_similarity=unfair_similarity,
                relative_unfairness=relative_unfairness,
                pattern_analysis=pattern_analysis,
                law_structure_info=law_structure_info,
                best_match=best_match
            )
            
            final_score = llm_judgment['adjusted_score']
            llm_reason = llm_judgment['reasoning']
            is_reversed = llm_judgment['is_reversed']
            
            print(f"✅ 의미 반전 여부: {is_reversed}")
            print(f"✅ 최종 점수: {final_score:.3f}")
            print(f"✅ LLM 추론: {llm_reason[:100]}...\n")
            
            # ==================================================================
            # Phase 8: 위반 여부 및 표현 결정
            # ==================================================================
            print("📍 Phase 8: 최종 판단 및 표현")
            print("-" * 70)
            
            violation, severity = self._determine_violation(final_score)
            confidence_expression = self._get_confidence_expression(final_score)
            
            print(f"  최종 점수: {final_score:.3f}")
            print(f"  판단: {'⚠️ 위반' if violation else '✅ 정상'}")
            print(f"  심각도: {severity} (계산된 심각도: {computed_severity})")
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
                'computed_severity': computed_severity,
                'confidence': final_score,
                'confidence_expression': confidence_expression,
                
                # 핵심 근거
                'primary_evidence': {
                    'best_match_id': best_case_id,
                    'unfair_similarity': unfair_similarity,
                    'fair_similarity': relative_unfairness['fair_similarity'],
                    'unfair_distance': relative_unfairness['unfair_distance'],
                    'fair_distance': relative_unfairness['fair_distance'],
                    'relative_unfairness': relative_unfairness['unfairness_score'],
                    'article_id': law_structure_info['article'],
                    'hang': law_structure_info.get('hang'),
                    'ho': law_structure_info.get('ho'),
                },
                
                # 그래프 컨텍스트
                'graph_context': {
                    'similar_cases_count': len(neighborhood['similar_cases']),
                    'related_laws': [law.get('id', '') for law in neighborhood['related_laws']],
                    'keywords': [
                        {
                            'text': kw.get('text', ''),
                            'case_count': kw.get('case_count', 0),
                            'prevalence': kw.get('prevalence', 0.0),
                            'risk_level': kw.get('risk_level', 'medium')
                        }
                        for kw in neighborhood['keywords']
                    ],
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
                
                # LLM 판단
                'llm_judgment': {
                    'formula_score': formula_score,
                    'adjusted_score': final_score,
                    'reasoning': llm_reason,
                    'is_reversed': is_reversed
                },
                
                # 수정 제안
                'suggestion': self._generate_suggestion(
                    neighborhood['related_laws'],
                    patterns,
                    pattern_analysis
                ),
                
                # 메타데이터
                'method': 'graphrag_v6_semantic_check',
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
    # 핵심 개선: 상대적 거리 기반 불공정도 계산
    # ======================================================================
    
    def _calculate_relative_unfairness(
        self,
        user_text: str,
        best_unfair_case_id: str,
        unfair_similarity: float
    ) -> Dict:
        """[핵심 개선] 상대적 거리 기반 불공정도 계산"""
        query = """
        MATCH (v:ViolationCase {id: $case_id})
        RETURN v.corrected_text as corrected_text
        """
        
        result = self.conn.execute_query(query, {'case_id': best_unfair_case_id})
        
        if not result or not result[0].get('corrected_text'):
            return {
                'fair_similarity': 0.0,
                'unfair_distance': 1.0 - unfair_similarity,
                'fair_distance': 1.0,
                'unfairness_score': unfair_similarity
            }
        
        corrected_text = result[0]['corrected_text']
        
        if not corrected_text or corrected_text.strip() == '':
            return {
                'fair_similarity': 0.0,
                'unfair_distance': 1.0 - unfair_similarity,
                'fair_distance': 1.0,
                'unfairness_score': unfair_similarity
            }
        
        user_emb = self.rag.local_embeddings.encode([user_text])[0]
        corrected_emb = self.rag.local_embeddings.encode([corrected_text])[0]
        
        fair_similarity = float(np.dot(user_emb, corrected_emb) / (
            np.linalg.norm(user_emb) * np.linalg.norm(corrected_emb)
        ))
        
        unfair_distance = 1.0 - unfair_similarity
        fair_distance = 1.0 - fair_similarity
        
        total_distance = unfair_distance + fair_distance
        
        if total_distance < 1e-6:
            unfairness_score = unfair_similarity
        else:
            unfairness_score = 1.0 - (unfair_distance / total_distance)
        
        unfairness_score = max(0.0, min(1.0, unfairness_score))
        
        return {
            'fair_similarity': fair_similarity,
            'unfair_distance': unfair_distance,
            'fair_distance': fair_distance,
            'unfairness_score': unfairness_score
        }
    
    # ======================================================================
    # 수식 기반 점수 계산 (개선!)
    # ======================================================================
    
    def _calculate_formula_score_v2(
        self,
        unfair_similarity: float,
        relative_unfairness: float,
        pattern_json_score: float,
        pattern_graph_strength: float
    ) -> float:
        """[개선] 수식 기반 점수 계산 - 상대적 불공정도 반영"""
        weights = {
            'unfair': 0.35,
            'relative': 0.35,
            'pattern_json': 0.25,
            'pattern_graph': 0.05
        }
        
        formula_score = (
            unfair_similarity * weights['unfair'] +
            relative_unfairness * weights['relative'] +
            pattern_json_score * weights['pattern_json'] +
            pattern_graph_strength * weights['pattern_graph']
        )
        
        print(f"  점수 구성:")
        print(f"    - 위반사례 유사도 ({weights['unfair']:.0%}): {unfair_similarity:.3f} → {unfair_similarity * weights['unfair']:.3f}")
        print(f"    - 상대적 불공정도 ({weights['relative']:.0%}): {relative_unfairness:.3f} → {relative_unfairness * weights['relative']:.3f}")
        print(f"    - JSON 패턴 ({weights['pattern_json']:.0%}): {pattern_json_score:.3f} → {pattern_json_score * weights['pattern_json']:.3f}")
        print(f"    - 그래프 패턴 ({weights['pattern_graph']:.0%}): {pattern_graph_strength:.3f} → {pattern_graph_strength * weights['pattern_graph']:.3f}")
        
        return formula_score
    
    # ======================================================================
    # LLM 의미 반전 검증 (수정!)
    # ======================================================================
    
    def _llm_semantic_reversal_check(
        self,
        user_text: str,
        formula_score: float,
        unfair_similarity: float,
        relative_unfairness: Dict,
        pattern_analysis: Dict,
        law_structure_info: Dict,
        best_match: Dict
    ) -> Dict:
        """
        LLM 의미 반전 검증
        
        역할: 점수 조정 X, 의미 반전 오류만 검증
        - 임베딩 유사도는 "책임지지 않습니다" vs "책임집니다"를 구분 못함
        - LLM이 문맥을 보고 실제 의미가 반대인지만 판단
        - 반전이 아니면 수식 점수 그대로 사용
        """
        evidence_parts = []
        
        evidence_parts.append(f"수식 기반 점수: {formula_score:.3f}")
        evidence_parts.append(f"위반사례 유사도: {unfair_similarity:.3f}")
        evidence_parts.append(f"수정본 유사도: {relative_unfairness['fair_similarity']:.3f}")
        evidence_parts.append(f"상대적 불공정도: {relative_unfairness['unfairness_score']:.3f}")
        
        if pattern_analysis['matched_keywords']:
            matched_kw = ', '.join([
                f"{kw['keyword']}({kw['method']})"
                for kw in pattern_analysis['matched_keywords'][:5]
            ])
            evidence_parts.append(f"매칭 키워드: {matched_kw}")
        
        evidence_parts.append(f"법조항: {law_structure_info['full_path']}")
        evidence_parts.append(f"유사 사례: {best_match['document'].page_content[:200]}...")
        
        evidence = '\n'.join(evidence_parts)
        
        prompt = f"""당신은 약관 전문가입니다. 다음 약관 조항의 의미 반전 여부만 검증해주세요.

[분석 대상]
{user_text}

[수식 기반 분석 결과]
{evidence}

[중요 주의사항]
임베딩 유사도는 "책임지지 않습니다" vs "책임집니다"를 구분하지 못합니다.
당신의 임무는 의미 반전 오류만 찾는 것입니다. 점수는 조정하지 마세요.

[검증 사항]
1. "없습니다", "않습니다", "안 됩니다" 등 부정 표현 확인
2. 불공정 사례와 유사한 단어를 사용하지만 **반대 의미**인지 확인
3. 문맥상 실제로 위반인지 확인

[판단 기준]
- 의미 반전 발견 (불공정→공정) → "반전": true
- 의미 반전 없음 (실제 위반) → "반전": false

[응답 형식]
반전: [true 또는 false]
추론: [2-3문장으로 핵심 근거 설명]

예시 1:
입력: "회사는 책임집니다" (유사 사례: "회사는 책임지지 않습니다")
→ 반전: true
→ 추론: 부정 표현이 없고 오히려 책임을 명시하여 불공정 사례와 반대 의미입니다.

예시 2:
입력: "회사는 책임지지 않습니다"
→ 반전: false
→ 추론: 불공정 사례와 동일하게 책임을 회피하는 표현으로 위반입니다.
"""
        
        try:
            response = self.rag.llm.invoke(prompt)
            content = response.content
            
            # 의미 반전 여부 추출
            is_reversed = False
            if '반전: true' in content or '반전:true' in content:
                is_reversed = True
            
            # 추론 추출
            reasoning_match = re.search(r'추론:\s*(.+?)(?:\n\n|\Z)', content, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = content[:200]
            
            # 점수 결정
            if is_reversed:
                # 의미 반전 발견 → 점수를 0.3으로 낮춤 (공정한 약관)
                adjusted_score = 0.3
                reasoning = f"[의미 반전 검출] {reasoning}"
                print(f"  ⚠️ 의미 반전 검출! 점수 하향: {formula_score:.3f} → {adjusted_score:.3f}")
            else:
                # 의미 반전 없음 → 수식 점수 그대로 사용
                adjusted_score = formula_score
                print(f"  ✅ 의미 반전 없음, 수식 점수 유지: {formula_score:.3f}")
            
            return {
                'adjusted_score': adjusted_score,
                'reasoning': reasoning,
                'is_reversed': is_reversed
            }
            
        except Exception as e:
            print(f"⚠️ LLM 검증 실패: {e}")
            return {
                'adjusted_score': formula_score,
                'reasoning': "LLM 검증 실패, 수식 점수 사용",
                'is_reversed': False
            }
    
    # ======================================================================
    # 신규/수정 메서드 (Graph Builder v2 반영)
    # ======================================================================
    
    def _compute_severity_from_keywords(
        self,
        graph_keywords: List[Dict],
        pattern_keywords: List[Dict]
    ) -> str:
        """[신규] Keyword 기반 실시간 severity 계산"""
        risk_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for kw in graph_keywords:
            if kw:
                risk_level = kw.get('risk_level', 'low')
                if risk_level in risk_counts:
                    case_count = kw.get('case_count', 1)
                    weight = min(case_count / 10.0, 2.0)
                    risk_counts[risk_level] += weight
        
        for kw_info in pattern_keywords:
            risk_level = kw_info.get('risk_level', 'low')
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1.5
        
        print(f"  위험도 집계:")
        print(f"    - critical: {risk_counts['critical']:.1f}")
        print(f"    - high: {risk_counts['high']:.1f}")
        print(f"    - medium: {risk_counts['medium']:.1f}")
        print(f"    - low: {risk_counts['low']:.1f}")
        
        if risk_counts['critical'] >= 2 or (risk_counts['critical'] >= 1 and risk_counts['high'] >= 1):
            return 'critical'
        elif risk_counts['critical'] >= 1 or risk_counts['high'] >= 2:
            return 'high'
        elif risk_counts['high'] >= 1 or risk_counts['medium'] >= 3:
            return 'medium'
        elif risk_counts['medium'] >= 1 or risk_counts['low'] >= 2:
            return 'low'
        else:
            return 'none'
    
    def _analyze_violation_patterns_with_case_count(
        self,
        user_text: str,
        similar_cases: List[Dict],
        neighborhood: Dict,
        pattern_analysis: Dict
    ) -> Dict:
        """[수정] 패턴 분석 - case_count 기반으로 변경"""
        json_keywords = {kw['keyword']: 3 for kw in pattern_analysis['matched_keywords']}
        
        keyword_case_counts = {}
        for kw in neighborhood.get('keywords', []):
            if kw:
                keyword_text = kw.get('text', '')
                case_count = kw.get('case_count', 1)
                if keyword_text:
                    keyword_case_counts[keyword_text] = case_count
        
        all_keywords = {}
        for kw, weight in json_keywords.items():
            all_keywords[kw] = all_keywords.get(kw, 0) + weight
        for kw, case_count in keyword_case_counts.items():
            all_keywords[kw] = all_keywords.get(kw, 0) + case_count
        
        common_keywords = sorted(
            all_keywords.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_keywords = common_keywords[:10]
        
        if top_keywords:
            total_case_count = sum(count for _, count in top_keywords)
            pattern_strength = min(total_case_count / 50.0, 1.0)
        else:
            pattern_strength = pattern_analysis['pattern_score']
        
        return {
            'common_keywords': [kw for kw, _ in top_keywords],
            'keyword_case_counts': dict(top_keywords),
            'top_keywords': [(kw, count) for kw, count in top_keywords[:3]],
            'strength': pattern_strength,
            'pattern_consistency': 0.8
        }
    
    def _explore_temp_node_neighborhood(
        self,
        temp_node_id: str,
        similar_cases: List[Dict]
    ) -> Dict:
        """[수정] 임시 노드 주변 탐색 - Keyword 노드 구조 변경 반영"""
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
        
        query = """
        MATCH (t:TempNode {id: $temp_id})-[:TEMP_SIMILAR]->(v:ViolationCase)
        
        OPTIONAL MATCH (v)-[:VIOLATES]->(law)
        WHERE law:호 OR law:항 OR law:조
        
        OPTIONAL MATCH (v)-[:CONTAINS]->(kw:Keyword)
        
        RETURN 
            collect(DISTINCT v) as cases,
            collect(DISTINCT law) as laws,
            collect(DISTINCT {
                text: kw.text,
                case_count: kw.case_count,
                prevalence: kw.prevalence,
                percentage: kw.percentage,
                risk_level: kw.risk_level
            }) as keywords
        """
        
        result = self.conn.execute_query(query, {'temp_id': temp_node_id})
        
        if not result:
            return {
                'similar_cases': [],
                'related_laws': [],
                'keywords': []
            }
        
        data = result[0]
        
        keywords = [kw for kw in data.get('keywords', []) if kw and kw.get('text')]
        
        return {
            'similar_cases': [dict(c) for c in data.get('cases', []) if c],
            'related_laws': [dict(l) for l in data.get('laws', []) if l],
            'keywords': keywords
        }
    
    def _analyze_with_patterns(self, user_text: str) -> Dict:
        """[수정] patterns_by_article_v2.json 패턴 분석 (정규표현식 지원)"""
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
        
        universal = self.patterns.get('universal_risk_keywords', {})
        
        if 'keywords' in universal:
            for kw_info in universal['keywords']:
                keyword = kw_info['keyword']
                if keyword in user_text:
                    matched_keywords.append({
                        'keyword': keyword,
                        'risk_level': kw_info['risk_level'],
                        'description': kw_info['description'],
                        'article': 'universal',
                        'method': 'string'
                    })
        
        if 'regex_patterns' in universal:
            for pattern_info in universal['regex_patterns']:
                if re.search(pattern_info['regex'], user_text):
                    matched_keywords.append({
                        'keyword': pattern_info['keyword'],
                        'risk_level': pattern_info['risk_level'],
                        'description': pattern_info.get('description', ''),
                        'article': 'universal',
                        'method': 'regex'
                    })
        
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
                            'article': article_id,
                            'method': 'string'
                        })
                
                normal_matched = 0
                for kw in pattern.get('keywords', []):
                    if kw in user_text:
                        normal_matched += 1
                
                if high_risk_matched > 0:
                    article_score += high_risk_matched * 0.3
                if normal_matched > 0:
                    article_score += normal_matched * 0.1
            
            if 'regex_patterns' in article_data:
                for pattern_info in article_data['regex_patterns']:
                    if re.search(pattern_info['regex'], user_text):
                        matched_keywords.append({
                            'keyword': pattern_info['keyword'],
                            'risk_level': pattern_info['risk_level'],
                            'description': pattern_info.get('description', ''),
                            'article': article_id,
                            'method': 'regex'
                        })
                        article_score += 0.3
            
            if article_score > 0:
                article_scores[article_id] = article_score
        
        combined = self.patterns.get('combined_pattern_risks', {})
        if 'patterns' in combined:
            for pattern in combined['patterns']:
                keywords = pattern['combination']
                
                all_matched = True
                for kw in keywords:
                    regex_pattern = self._find_regex_for_keyword(kw)
                    
                    if regex_pattern:
                        if not re.search(regex_pattern, user_text):
                            all_matched = False
                            break
                    else:
                        if kw not in user_text:
                            all_matched = False
                            break
                
                if all_matched:
                    combined_patterns_found.append({
                        'keywords': keywords,
                        'risk_level': pattern['risk_level'],
                        'description': pattern['description'],
                        'articles': pattern['articles']
                    })
        
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
    
    def _find_regex_for_keyword(self, keyword: str) -> str:
        """키워드에 대응하는 정규표현식 찾기"""
        if not self.patterns:
            return None
        
        universal = self.patterns.get('universal_risk_keywords', {})
        if 'regex_patterns' in universal:
            for pattern_info in universal['regex_patterns']:
                if pattern_info['keyword'] == keyword:
                    return pattern_info['regex']
        
        for article_id in ['제6조', '제7조', '제8조', '제9조', '제10조', 
                          '제11조', '제12조', '제13조', '제14조']:
            
            if article_id not in self.patterns:
                continue
            
            article_data = self.patterns[article_id]
            
            if 'regex_patterns' in article_data:
                for pattern_info in article_data['regex_patterns']:
                    if pattern_info['keyword'] == keyword:
                        return pattern_info['regex']
        
        return None
    
    # ======================================================================
    # 기존 메서드들
    # ======================================================================
    
    def _get_confidence_expression(self, score: float) -> str:
        """위반 퍼센트에 따른 표현 차별화"""
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
    
    def _analyze_law_structure(self, case_id: str) -> Dict:
        """Law-Centric 구조 분석"""
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
            fallback_query = """
            MATCH (v:ViolationCase {id: $case_id})
            RETURN v.article_id as article_id
            """
            fallback_result = self.conn.execute_query(fallback_query, {'case_id': case_id})
            
            if fallback_result and fallback_result[0].get('article_id'):
                article_id = fallback_result[0]['article_id']
                return {
                    'article': article_id or 'Unknown',
                    'article_title': '',
                    'full_path': article_id if article_id else 'Unknown',
                    'ho_content': ''
                }
            
            return {
                'article': 'Unknown',
                'article_title': '',
                'full_path': 'Unknown',
                'ho_content': ''
            }
        
        data = result[0]
        target_type = data.get('target_type', '')
        target_id = data.get('target_id', '')
        target_content = data.get('target_content', '')
        
        article_id = data.get('article_id') or data.get('article_id_alt') or 'Unknown'
        article_title = data.get('article_title') or data.get('article_title_alt', '')
        
        hang_id = None
        ho_id = None
        ho_content = target_content or ''
        
        if target_type == '호':
            ho_id = target_id
            hang_id = data.get('hang_id')
        elif target_type == '항':
            hang_id = target_id
        
        path_parts = []
        
        if article_id and article_id != 'Unknown':
            path_parts.append(article_id)
        else:
            path_parts.append('Unknown')
        
        if hang_id:
            hang_display = hang_id.split('_')[-1] if '_' in hang_id else hang_id
            if hang_display:
                path_parts.append(hang_display)
        
        if ho_id:
            ho_display = ho_id.split('_')[-1] if '_' in ho_id else ho_id
            if ho_display:
                path_parts.append(ho_display)
        
        full_path = ' '.join([str(p) for p in path_parts if p is not None and str(p).strip()])
        
        return {
            'article': article_id,
            'article_title': article_title,
            'hang': hang_id,
            'ho': ho_id,
            'ho_content': ho_content,
            'full_path': full_path if full_path else 'Unknown'
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
            'computed_severity': severity,
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
    
    def _select_best_match_with_priority(self, similar_cases: List[Dict]) -> Dict:
        """조항 우선순위 고려 (6조 후순위)"""
        if not similar_cases:
            return None
        
        scored_cases = []
        max_priority = max(self.ARTICLE_PRIORITY.values())
        
        for case in similar_cases:
            article_id = case['metadata'].get('article_id', '')
            priority = self.ARTICLE_PRIORITY.get(article_id, max_priority)
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
        print(f"     우선순위: {best['priority']} (6조=1, 나머지=높음)")
        print(f"     종합 점수: {best['combined_score']:.3f}")
        
        if best['article_id'] == '제6조':
            print(f"     ⚠️ 6조 선택 (포괄적 조항)")
        
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
                f"{kw['keyword']}({kw['risk_level']},{kw['method']})"
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