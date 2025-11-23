"""
GraphRAG 기반 불공정 약관 판단 시스템 - v8.1 (Fixed)

[수정사항]
1. ✅ _generate_explanation() 시그니처 수정
2. ✅ _generate_suggestion() 시그니처 수정
3. ✅ 임시 노드 생성/삭제 제거 (불필요)
4. ✅ Prototypical Networks 구현 개선 (여러 prototype 사용)
5. ✅ Temperature scaling 개선
"""

from typing import Dict, List, Tuple
from rag.hybrid_graphrag import HybridGraphRAG
from database.neo4j_connector import Neo4jConnector
import numpy as np
import json
from pathlib import Path
import re


class GraphRAGJudge:
    """GraphRAG 기반 약관 판단 시스템 (v8.1 - Fixed)"""
    
    def __init__(self, rag: HybridGraphRAG, conn: Neo4jConnector):
        self.rag = rag
        self.conn = conn
        
        # 임계값 설정
        self.THRESHOLDS = {
            'high_risk': 0.85,
            'medium_risk': 0.70,
            'low_risk': 0.55,
        }
        
        # Prototypical Networks 파라미터
        self.TEMPERATURE = 0.5  # 0.4 → 0.5 (더 부드러운 확률 분포)
        
        # 조항 우선순위
        self.ARTICLE_PRIORITY = {
            '제6조': 1,
        }
        self.DEFAULT_PRIORITY = 10
        
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
        print(f"🔍 약관 판단 시작 (v8.1 - Fixed)")
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
        
        # 조항 우선순위 고려
        best_match = self._select_best_match_with_priority(similar_cases)
        best_case_id = best_match['metadata']['id']
        unfair_similarity = best_match['similarity_score']
        
        # ==================================================================
        # Phase 2: Prototypical Networks 기반 상대적 불공정도
        # ==================================================================
        print("📍 Phase 2: Prototypical Networks 기반 불공정도 계산")
        print("-" * 70)
        
        relative_unfairness = self._calculate_prototypical_unfairness(
            user_text,
            similar_cases,  # 여러 사례 전달
            best_case_id
        )
        
        print(f"✅ 방법론: {relative_unfairness['method']}")
        print(f"✅ Unfair prototype 거리: {relative_unfairness.get('unfair_distance', 'N/A')}")
        print(f"✅ Fair prototype 거리: {relative_unfairness.get('fair_distance', 'N/A')}")
        print(f"✅ 상대적 불공정도 (P(unfair)): {relative_unfairness['unfairness_score']:.3f}")
        if relative_unfairness.get('temperature'):
            print(f"   온도 파라미터 τ: {relative_unfairness['temperature']}")
        print(f"   해석: {relative_unfairness['interpretation']}\n")
        
        # ==================================================================
        # Phase 3: 법률 구조 분석
        # ==================================================================
        print("📍 Phase 3: 법률 구조 분석")
        print("-" * 70)
        
        law_structure_info = self._analyze_law_structure(best_case_id)
        
        print(f"✅ 위반 조항: {law_structure_info['article']}")
        if law_structure_info.get('hang'):
            print(f"   항: {law_structure_info['hang']}")
        if law_structure_info.get('ho'):
            print(f"   호: {law_structure_info['ho']}")
        print(f"   상세: {law_structure_info.get('ho_content', 'N/A')[:100]}...\n")
        
        # ==================================================================
        # Phase 4: 간소화된 수식 기반 종합 점수 계산
        # ==================================================================
        print("📍 Phase 4: 종합 점수 계산 (3가지 요소)")
        print("-" * 70)
        
        formula_score = self._calculate_simplified_score(
            unfair_similarity=unfair_similarity,
            relative_unfairness=relative_unfairness['unfairness_score'],
            pattern_json_score=pattern_analysis['pattern_score']
        )
        
        print(f"  최종 수식 점수: {formula_score:.3f}\n")
        
        # ==================================================================
        # Phase 5: LLM 의미 반전 검증
        # ==================================================================
        print("📍 Phase 5: LLM 의미 반전 검증")
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
        # Phase 6: 최종 판단
        # ==================================================================
        print("📍 Phase 6: 최종 판단 및 표현")
        print("-" * 70)
        
        violation, severity = self._determine_violation(final_score)
        confidence_expression = self._get_confidence_expression(final_score)
        
        print(f"  최종 점수: {final_score:.3f}")
        print(f"  판단: {'⚠️ 위반' if violation else '✅ 정상'}")
        print(f"  심각도: {severity}")
        print(f"  표현: {confidence_expression}\n")
        
        # ==================================================================
        # Phase 7: 설명 생성
        # ==================================================================
        print("📍 Phase 7: 설명 생성")
        print("-" * 70)
        
        explanation = self._generate_explanation(
            user_text=user_text,
            best_match=best_match,
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
            'confidence_expression': confidence_expression,
            
            # 핵심 근거
            'primary_evidence': {
                'best_match_id': best_case_id,
                'unfair_similarity': unfair_similarity,
                'unfair_distance': relative_unfairness.get('unfair_distance'),
                'fair_distance': relative_unfairness.get('fair_distance'),
                'relative_unfairness': relative_unfairness['unfairness_score'],
                'method': relative_unfairness['method'],
                'article_id': law_structure_info['article'],
                'hang': law_structure_info.get('hang'),
                'ho': law_structure_info.get('ho'),
            },
            
            # 패턴 분석
            'patterns': {
                'matched_risk_keywords': pattern_analysis['matched_keywords'],
                'risk_level_from_patterns': pattern_analysis['risk_level'],
                'pattern_score': pattern_analysis['pattern_score'],
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
            
            # 수정 제안 (수정됨)
            'suggestion': self._generate_suggestion(
                pattern_analysis=pattern_analysis,
                law_structure_info=law_structure_info
            ),
            
            # 메타데이터
            'method': 'graphrag_v8.1_fixed',
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
    
    # ======================================================================
    # 핵심: Prototypical Networks (개선)
    # ======================================================================
    
    def _calculate_prototypical_unfairness(
        self,
        user_text: str,
        similar_cases: List[Dict],
        best_unfair_case_id: str
    ) -> Dict:
        """
        Prototypical Networks 기반 불공정도 계산 (개선)
        
        개선사항:
        1. 여러 unfair 사례의 평균으로 prototype 생성
        2. 여러 fair 사례의 평균으로 prototype 생성
        3. Squared Euclidean distance 사용
        """
        
        # 1. Fair prototypes 수집 (수정본들)
        query = """
        MATCH (v:ViolationCase)
        WHERE v.id IN $case_ids AND v.corrected_text IS NOT NULL
        RETURN v.id as id, v.text as unfair_text, v.corrected_text as fair_text
        LIMIT 5
        """
        
        case_ids = [case['metadata']['id'] for case in similar_cases[:5]]
        results = self.conn.execute_query(query, {'case_ids': case_ids})
        
        if not results:
            # Fallback: 단일 사례만 사용
            return self._calculate_single_case_unfairness(user_text, best_unfair_case_id)
        
        # 2. 임베딩 생성
        user_embedding = self.rag.local_embeddings.encode([user_text])[0]
        
        unfair_embeddings = []
        fair_embeddings = []
        
        for result in results:
            unfair_text = result.get('unfair_text', '')
            fair_text = result.get('fair_text', '')
            
            if unfair_text:
                unfair_embeddings.append(
                    self.rag.local_embeddings.encode([unfair_text])[0]
                )
            
            if fair_text and fair_text.strip():
                fair_embeddings.append(
                    self.rag.local_embeddings.encode([fair_text])[0]
                )
        
        if not unfair_embeddings or not fair_embeddings:
            return self._calculate_single_case_unfairness(user_text, best_unfair_case_id)
        
        # 3. Prototype 생성 (평균)
        unfair_prototype = np.mean(unfair_embeddings, axis=0)
        fair_prototype = np.mean(fair_embeddings, axis=0)
        
        # 4. Squared Euclidean distance 계산
        dist_to_unfair = float(np.sum((user_embedding - unfair_prototype) ** 2))
        dist_to_fair = float(np.sum((user_embedding - fair_prototype) ** 2))
        
        # 5. Temperature-scaled softmax
        # d(x, c) = ||x - c||^2 이므로 logit은 -d(x, c)
        logits = np.array([-dist_to_unfair, -dist_to_fair])
        scaled_logits = logits / self.TEMPERATURE
        
        # Numerical stability
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        unfairness_score = float(probs[0])
        unfairness_score = max(0.0, min(1.0, unfairness_score))
        
        # 6. 해석
        distance_ratio = dist_to_unfair / (dist_to_fair + 1e-8)
        
        if unfairness_score >= 0.7:
            interpretation = f"불공정 prototype에 매우 가깝습니다 (거리비: {distance_ratio:.2f})"
        elif unfairness_score >= 0.5:
            interpretation = f"불공정 영역에 속합니다 (거리비: {distance_ratio:.2f})"
        elif unfairness_score >= 0.3:
            interpretation = f"중립 영역입니다 (거리비: {distance_ratio:.2f})"
        else:
            interpretation = f"공정 prototype에 가깝습니다 (거리비: {distance_ratio:.2f})"
        
        return {
            'unfair_distance': dist_to_unfair,
            'fair_distance': dist_to_fair,
            'unfairness_score': unfairness_score,
            'temperature': self.TEMPERATURE,
            'method': 'prototypical_networks_multi_prototype',
            'interpretation': interpretation,
            'distance_ratio': distance_ratio,
            'num_unfair_prototypes': len(unfair_embeddings),
            'num_fair_prototypes': len(fair_embeddings)
        }
    
    def _calculate_single_case_unfairness(
        self,
        user_text: str,
        best_unfair_case_id: str
    ) -> Dict:
        """Fallback: 단일 사례 기반 불공정도 계산"""
        
        query = """
        MATCH (v:ViolationCase {id: $case_id})
        RETURN v.corrected_text as corrected_text, v.text as unfair_text
        """
        
        result = self.conn.execute_query(query, {'case_id': best_unfair_case_id})
        
        if not result or not result[0].get('corrected_text'):
            # 수정본도 없으면 유사도만 사용
            user_emb = self.rag.local_embeddings.encode([user_text])[0]
            
            # 최소한의 거리 추정
            return {
                'unfair_distance': 0.5,
                'fair_distance': 1.0,
                'unfairness_score': 0.7,  # 보수적 추정
                'temperature': None,
                'method': 'single_similarity_fallback',
                'interpretation': '수정본 없음 - 보수적 추정'
            }
        
        corrected_text = result[0]['corrected_text']
        unfair_text = result[0].get('unfair_text', '')
        
        # 단일 prototype 사용
        user_embedding = self.rag.local_embeddings.encode([user_text])[0]
        unfair_prototype = self.rag.local_embeddings.encode([unfair_text])[0] if unfair_text else None
        fair_prototype = self.rag.local_embeddings.encode([corrected_text])[0]
        
        if unfair_prototype is not None:
            dist_to_unfair = float(np.sum((user_embedding - unfair_prototype) ** 2))
        else:
            dist_to_unfair = 0.5  # 추정값
        
        dist_to_fair = float(np.sum((user_embedding - fair_prototype) ** 2))
        
        # Softmax
        logits = np.array([-dist_to_unfair, -dist_to_fair])
        scaled_logits = logits / self.TEMPERATURE
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        unfairness_score = float(probs[0])
        
        distance_ratio = dist_to_unfair / (dist_to_fair + 1e-8)
        
        if unfairness_score >= 0.7:
            interpretation = f"불공정에 가깝습니다 (단일 사례, 거리비: {distance_ratio:.2f})"
        elif unfairness_score >= 0.5:
            interpretation = f"불공정 영역입니다 (단일 사례, 거리비: {distance_ratio:.2f})"
        else:
            interpretation = f"공정에 가깝습니다 (단일 사례, 거리비: {distance_ratio:.2f})"
        
        return {
            'unfair_distance': dist_to_unfair,
            'fair_distance': dist_to_fair,
            'unfairness_score': unfairness_score,
            'temperature': self.TEMPERATURE,
            'method': 'prototypical_networks_single_prototype',
            'interpretation': interpretation,
            'distance_ratio': distance_ratio
        }
    
    # ======================================================================
    # 간소화된 점수 계산
    # ======================================================================
    
    def _calculate_simplified_score(
        self,
        unfair_similarity: float,
        relative_unfairness: float,
        pattern_json_score: float
    ) -> float:
        """
        간소화된 수식 기반 점수 계산
        
        구성:
        - 50%: 위반사례 유사도
        - 35%: Prototypical 상대적 불공정도
        - 15%: JSON 패턴 점수
        """
        weights = {
            'unfair': 0.50,
            'relative': 0.35,
            'pattern_json': 0.15
        }
        
        formula_score = (
            unfair_similarity * weights['unfair'] +
            relative_unfairness * weights['relative'] +
            pattern_json_score * weights['pattern_json']
        )
        
        print(f"  점수 구성:")
        print(f"    - 위반사례 유사도 ({weights['unfair']:.0%}): {unfair_similarity:.3f} → {unfair_similarity * weights['unfair']:.3f}")
        print(f"    - Prototypical 불공정도 ({weights['relative']:.0%}): {relative_unfairness:.3f} → {relative_unfairness * weights['relative']:.3f}")
        print(f"    - JSON 패턴 ({weights['pattern_json']:.0%}): {pattern_json_score:.3f} → {pattern_json_score * weights['pattern_json']:.3f}")
        
        return formula_score
    
    # ======================================================================
    # 패턴 분석
    # ======================================================================
    
    def _analyze_with_patterns(self, user_text: str) -> Dict:
        """patterns_by_article_v2.json 패턴 분석"""
        if not self.patterns:
            return {
                'matched_keywords': [],
                'risk_level': 'unknown',
                'pattern_score': 0.0,
                'article_hints': []
            }
        
        matched_keywords = []
        article_scores = {}
        
        # Universal 키워드
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
        
        # 조항별 키워드
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
                
                normal_matched = sum(1 for kw in pattern.get('keywords', []) if kw in user_text)
                
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
        
        # Risk level 결정
        risk_scores = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for kw in matched_keywords:
            risk_level = kw.get('risk_level', 'low')
            if risk_level in risk_scores:
                risk_scores[risk_level] += 1
        
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
        
        # Pattern score
        total_matches = sum(risk_scores.values())
        pattern_score = min(total_matches / 10.0, 1.0)
        
        risk_multipliers = {'critical': 1.5, 'high': 1.2, 'medium': 1.0, 'low': 0.8}
        pattern_score *= risk_multipliers.get(final_risk_level, 1.0)
        pattern_score = min(pattern_score, 1.0)
        
        article_hints = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'matched_keywords': matched_keywords,
            'risk_level': final_risk_level,
            'pattern_score': pattern_score,
            'article_hints': [article for article, _ in article_hints]
        }
    
    # ======================================================================
    # 법률 구조 분석
    # ======================================================================
    
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
            parent_hang.id as hang_id
        LIMIT 1
        """
        
        result = self.conn.execute_query(query, {'case_id': case_id})
        
        if not result:
            fallback_query = "MATCH (v:ViolationCase {id: $case_id}) RETURN v.article_id as article_id"
            fallback_result = self.conn.execute_query(fallback_query, {'case_id': case_id})
            
            if fallback_result and fallback_result[0].get('article_id'):
                article_id = fallback_result[0]['article_id']
                return {
                    'article': article_id or 'Unknown',
                    'article_title': '',
                    'full_path': article_id if article_id else 'Unknown',
                    'ho_content': ''
                }
            
            return {'article': 'Unknown', 'article_title': '', 'full_path': 'Unknown', 'ho_content': ''}
        
        data = result[0]
        target_type = data.get('target_type', '')
        target_content = data.get('target_content', '')
        
        article_id = data.get('article_id') or data.get('article_id_alt') or 'Unknown'
        article_title = data.get('article_title', '')
        
        hang_id = data.get('hang_id')
        ho_id = data.get('target_id') if target_type == '호' else None
        ho_content = target_content or ''
        
        # Path 구성
        path_parts = [article_id]
        
        if hang_id:
            hang_display = hang_id.split('_')[-1] if '_' in hang_id else hang_id
            if hang_display:
                path_parts.append(hang_display)
        
        if ho_id:
            ho_display = ho_id.split('_')[-1] if '_' in ho_id else ho_id
            if ho_display:
                path_parts.append(ho_display)
        
        full_path = ' '.join([str(p) for p in path_parts if p and str(p).strip()])
        
        return {
            'article': article_id,
            'article_title': article_title,
            'hang': hang_id,
            'ho': ho_id,
            'ho_content': ho_content,
            'full_path': full_path if full_path else 'Unknown'
        }
    
    # ======================================================================
    # LLM 검증 및 설명 (수정됨)
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
        """LLM 의미 반전 검증"""
        evidence_parts = [
            f"수식 점수: {formula_score:.3f}",
            f"위반사례 유사도: {unfair_similarity:.3f}",
            f"불공정도 확률: {relative_unfairness['unfairness_score']:.3f}"
        ]
        
        if relative_unfairness.get('fair_distance'):
            evidence_parts.append(f"Unfair 거리: {relative_unfairness['unfair_distance']:.3f}")
            evidence_parts.append(f"Fair 거리: {relative_unfairness['fair_distance']:.3f}")
        
        if pattern_analysis['matched_keywords']:
            matched_kw = ', '.join([f"{kw['keyword']}({kw['method']})" for kw in pattern_analysis['matched_keywords'][:5]])
            evidence_parts.append(f"키워드: {matched_kw}")
        
        evidence_parts.append(f"조항: {law_structure_info['full_path']}")
        evidence_parts.append(f"유사사례: {best_match['document'].page_content[:200]}...")
        
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
추론: [2-3문장으로 핵심 근거 설명]"""
        
        try:
            response = self.rag.llm.invoke(prompt)
            content = response.content
            
            is_reversed = False
            if '반전: true' in content or '반전:true' in content:
                is_reversed = True
            
            reasoning_match = re.search(r'추론:\s*(.+?)(?:\n\n|\Z)', content, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = content[:200]
            
            if is_reversed:
                adjusted_score = 0.3
                reasoning = f"[의미 반전 검출] {reasoning}"
                print(f"  ⚠️ 의미 반전 검출! 점수 하향: {formula_score:.3f} → {adjusted_score:.3f}")
            else:
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
            },
            'explanation': f"{confidence_expression}. 유사 사례가 없어 패턴 분석만 수행했습니다.",
            'suggestion': self._generate_suggestion(
                pattern_analysis=pattern_analysis,
                law_structure_info={'article': 'Unknown', 'full_path': 'Unknown'}
            )
        }
    
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
    
    # ======================================================================
    # 설명 및 제안 생성 (수정됨 - 시그니처 일치)
    # ======================================================================
    
    def _generate_explanation(
        self,
        user_text: str,
        best_match: Dict,
        final_score: float,
        pattern_analysis: Dict,
        law_structure_info: Dict,
        confidence_expression: str
    ) -> str:
        """LLM 설명 생성 (수정된 시그니처)"""
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
        
        prompt = f"""다음 약관을 분석했습니다:

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

각 2-3문장으로."""
        
        try:
            response = self.rag.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"{confidence_expression}. 패턴 분석: {len(pattern_analysis['matched_keywords'])}개 위험 키워드"
    
    def _generate_suggestion(
        self,
        pattern_analysis: Dict,
        law_structure_info: Dict
    ) -> str:
        """수정 제안 (수정된 시그니처)"""
        risk_keywords = [kw['keyword'] for kw in pattern_analysis['matched_keywords'][:3]]
        
        prompt = f"""약관 수정 제안:

위반 조항: {law_structure_info['full_path']}
위험 키워드: {', '.join(risk_keywords) if risk_keywords else '없음'}

3-4문장으로 수정 방향 제시."""
        
        try:
            response = self.rag.llm.invoke(prompt)
            return response.content
        except Exception as e:
            if risk_keywords:
                return f"위험 키워드 제거/완화: {', '.join(risk_keywords)}"
            return "고의·중과실 책임 명시, 불가항력 구체화"