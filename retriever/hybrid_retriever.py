"""
하이브리드 검색 리트리버: 의미 + 키워드 + 그래프 구조 결합

2단계: 하이브리드 검색 전략 수립 - 구현
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
from collections import defaultdict

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector

# 임베딩 모델 로드
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception:
    try:
        MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        print(f"Warning: 임베딩 모델 로드 실패: {e}")
        MODEL = None


def cosine_similarity(v1, v2):
    """코사인 유사도 계산"""
    if v1 is None or v2 is None:
        return 0.0
    v1, v2 = np.array(v1), np.array(v2)
    if v1.size == 0 or v2.size == 0:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


class HybridRetriever:
    """
    하이브리드 검색 리트리버
    
    의미적 유사도(50%) + 키워드 매칭(30%) + 그래프 구조(20%) 결합
    """
    
    def __init__(self, conn: Neo4jConnector):
        self.conn = conn
        self.model = MODEL
        
        # 조항별 키워드 및 패턴 (judge_clause.py의 ARTICLE_PATTERNS 활용)
        self.article_patterns = {
            "제6조": {
                "keywords": ["부당", "불리", "예상", "어려운", "본질", "권리", "제한"],
                "patterns": [r"부당.*불리", r"예상.*어려운", r"본질.*권리.*제한"]
            },
            "제7조": {
                "keywords": ["면책", "책임", "배상", "배제", "제한", "담보", "고의", "과실", "손해", "피해", "책임없", "책임지지않", "피해배상.*않", "손해배상.*않"],
                "patterns": [r"책임.*없", r"면책", r"배상.*않", r"손해.*책임.*없",
                            r"어떠한.*책임.*지지.*않", r"어떠한.*경우.*책임.*없", 
                            r"피해배상.*않", r"손해배상.*않", r"어떠한.*피해.*않", r"어떠한.*손해.*않"]
            },
            "제8조": {
                "keywords": ["손해금", "지연", "위약금", "배상금", "과도", "과중", "지연.*손해금", "과도.*손해", "과중.*손해",
                            "배.*배", "배상.*배", "금액.*배", "원금.*배", "계약금액.*배"],
                "patterns": [
                    r"과도.*손해배상", r"과중.*손해배상", r"과도.*손해배상금", r"과중.*손해배상금",
                    r"지연.*손해금", r"위약금.*배", r"손해배상.*배", r"손해금.*배",
                    r"\d+배.*손해", r"원금.*\d+배", r"계약금액.*\d+배", r"연\s*\d+%", r"일일\s*\d+%",
                    r"과중.*지연.*손해금", r"부당.*과중.*손해배상"
                ]
            },
            "제9조": {
                "keywords": ["해제", "해지", "원상회복", "존속", "계약.*해지.*없"],
                "patterns": [r"해제", r"해지.*없", r"해지.*수.*없", r"원상회복", r"존속기간"]
            },
            "제10조": {
                "keywords": ["급부", "변경", "일방적", "중지", "일방적.*변경", "일방적.*중지", "급부.*일방", "서비스.*변경"],
                "patterns": [
                    r"일방적.*변경", r"급부.*변경", r"일방적.*중지", r"급부.*일방.*결정",
                    r"급부.*일방.*변경", r"급부.*중지", r"제3자.*대행", r"일방적.*서비스.*변경",
                    r"사업자.*일방.*변경", r"일방적.*이행.*중지"
                ]
            },
            "제11조": {
                "keywords": ["기한", "이익", "박탈", "상실", "항변권", "상계권", "기한의", "이익.*박탈", "이익.*상실"],
                "patterns": [r"기한.*이익.*박탈", r"기한.*이익.*상실", r"항변권.*배제", r"상계권.*배제", 
                            r"기한.*이익", r"항변권", r"상계권"]
            },
            "제12조": {
                "keywords": ["의사표시", "간주", "의제", "동의.*간주", "답변.*간주", "부작위.*간주", "의제.*의사표시"],
                "patterns": [
                    r"동의.*간주", r"답변.*간주", r"작위.*간주", r"부작위.*간주",
                    r"의사표시.*간주", r"의사표시.*의제", r"의제.*의사표시",
                    r"의사표시.*형식", r"의사표시.*요건", r"의사표시.*도달.*간주",
                    r"의사표시.*기한.*불확정", r"의사표시.*기한.*부당"
                ]
            },
            "제13조": {
                "keywords": ["대리인", "대리인.*체결", "대리인.*의하여", "대리인.*이행", "대리인.*의무", "대리인.*채무"],
                "patterns": [
                    r"대리인.*의하여.*계약", r"대리인.*체결.*경우", r"대리인.*의무.*이행.*책임",
                    r"대리인.*이행.*책임", r"대리인.*의무.*책임", r"대리인.*채무.*이행",
                    r"고객.*대리인.*의무.*이행", r"대리인.*고객.*의무.*이행",
                    r"대리인.*책임.*가중", r"대리인.*책임.*지운다",
                    r"대리인.*의하여.*의무", r"대리인.*의하여.*채무"
                ]
            },
            "제14조": {
                "keywords": ["소송", "관할", "입증", "소송.*금지", "관할법원", "재판관할"],
                "patterns": [r"소송.*금지", r"관할법원", r"재판관할", r"소송.*관할", r"입증책임"]
            }
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드 추출"""
        # 일반적인 키워드 추출
        words = re.findall(r'\w+', text.lower())
        
        # 조항별 키워드 수집
        found_keywords = []
        for article_id, info in self.article_patterns.items():
            for kw in info.get('keywords', []):
                if kw in text.lower():
                    found_keywords.append(kw)
        
        # 중복 제거 및 정렬
        unique_keywords = list(set(words + found_keywords))
        return unique_keywords[:15]  # 최대 15개
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        1단계: 의미적 유사도 기반 검색
        
        질의와 법률 조항(조/항/호)의 임베딩을 비교하여 유사한 조항 검색
        """
        if not self.model:
            return []
        
        # 질의 임베딩 생성
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # 모든 조/항/호 노드 조회
        query_cypher = """
        MATCH (n)
        WHERE (n:조 OR n:항 OR n:호) 
          AND n.content IS NOT NULL 
          AND n.content <> ''
          AND n.embedding IS NOT NULL
        RETURN 
            n.id as id,
            labels(n)[0] as node_type,
            n.content as content,
            n.embedding as embedding,
            n.title as title,
            n.article_id as article_id
        """
        
        nodes = self.conn.execute_query(query_cypher)
        
        # 유사도 계산 및 정렬
        candidates = []
        for node in nodes:
            node_embedding = node.get('embedding')
            if not node_embedding:
                continue
            
            similarity = cosine_similarity(query_embedding, node_embedding)
            
            candidates.append({
                'id': node.get('id'),
                'node_type': node.get('node_type'),
                'content': node.get('content'),
                'title': node.get('title'),
                'article_id': node.get('article_id'),
                'semantic_score': similarity
            })
        
        # 유사도 기준 정렬 후 Top-K 반환
        candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
        return candidates[:top_k]
    
    def keyword_search(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        2단계: 키워드 및 패턴 매칭 점수 계산
        
        후보 조항들에 대해 키워드 및 패턴 매칭 점수 부여
        조항별 특성 반영하여 가중치 적용
        """
        query_lower = query.lower()
        
        for candidate in candidates:
            content = candidate.get('content', '').lower()
            title = candidate.get('title', '').lower() if candidate.get('title') else ''
            text = (content + ' ' + title).lower()
            article_id = candidate.get('article_id', '')
            
            # 1. 조항별 키워드 매칭 점수
            keyword_score = 0.0
            pattern_score = 0.0
            article_bonus = 0.0
            
            # 후보 조항의 article_id에 해당하는 패턴 확인
            if article_id in self.article_patterns:
                article_info = self.article_patterns[article_id]
                
                # 키워드 매칭 (질의에 키워드가 있고, 조항 내용에도 있으면 높은 점수)
                article_keywords = article_info.get('keywords', [])
                matched_keywords = []
                
                # 정규식 키워드와 일반 키워드 분리
                regex_keywords = [kw for kw in article_keywords if '*' in kw or '.' in kw]
                normal_keywords = [kw for kw in article_keywords if kw not in regex_keywords]
                
                # 일반 키워드 매칭
                for kw in normal_keywords:
                    if kw in query_lower:
                        matched_keywords.append(kw)
                        if kw in text:
                            # 특정 조항의 핵심 키워드는 더 높은 가중치
                            if article_id == "제8조" and kw in ["과도", "과중", "배상금", "손해금", "위약금"]:
                                weight = 0.4  # 제8조 핵심 키워드
                            elif article_id == "제13조" and "대리인" in kw:
                                weight = 0.5  # 제13조 핵심 키워드
                            elif article_id == "제10조" and "일방" in kw:
                                weight = 0.4  # 제10조 핵심 키워드
                            elif article_id == "제12조" and ("간주" in kw or "의제" in kw):
                                weight = 0.4  # 제12조 핵심 키워드
                            else:
                                weight = min(len(kw) / 10.0, 0.3) if len(kw) > 3 else 0.1
                            keyword_score += weight
                
                # 정규식 키워드 매칭 (패턴으로 처리)
                for regex_kw in regex_keywords:
                    pattern = regex_kw.replace('*', '.*')
                    if re.search(pattern, query_lower):
                        matched_keywords.append(regex_kw)
                        if re.search(pattern, text):
                            keyword_score += 0.3
                
                keyword_score = min(keyword_score, 1.5)  # 최대값 증가
                
                # 패턴 매칭 (질의와 조항 모두 매칭)
                article_patterns = article_info.get('patterns', [])
                for pattern in article_patterns:
                    if re.search(pattern, query_lower):
                        # 조항 내용에서도 패턴 확인하면 보너스
                        if re.search(pattern, text):
                            pattern_score += 0.5
                        else:
                            pattern_score += 0.2
                
                pattern_score = min(pattern_score, 2.0)  # 패턴은 더 높은 점수 가능
                
                # 조항별 보너스: 질의에서 해당 조항의 핵심 키워드가 많이 나타나면
                if matched_keywords:
                    article_bonus = min(len(matched_keywords) * 0.1, 0.3)
            
            # 2. 일반 키워드 매칭 (기존 방식, 낮은 가중치)
            general_keywords = self.extract_keywords(query)
            general_matched = sum(1 for kw in general_keywords if kw in text)
            general_score = (general_matched / max(len(general_keywords), 1)) * 0.2
            
            # 최종 키워드 점수 (조항별 매칭이 더 중요)
            final_keyword_score = (
                keyword_score * 0.4 +      # 조항별 키워드 매칭 (40%)
                pattern_score * 0.4 +      # 패턴 매칭 (40%)
                article_bonus * 0.15 +     # 조항 보너스 (15%)
                general_score * 0.05       # 일반 매칭 (5%)
            )
            
            candidate['keyword_score'] = min(final_keyword_score, 2.0)  # 최대 2.0
            candidate['matched_keywords'] = matched_keywords if article_id in self.article_patterns else []
            candidate['pattern_matched'] = pattern_score > 0
        
        return candidates
    
    def graph_structure_score(self, candidates: List[Dict]) -> List[Dict]:
        """
        3단계: 그래프 구조 점수 계산
        
        위반사례와의 연결 강도, 그래프 경로 등을 고려
        """
        # 각 후보 조항의 위반사례 개수 확인
        for candidate in candidates:
            node_id = candidate.get('id')
            node_type = candidate.get('node_type')
            
            # 위반사례 개수 조회
            if node_type == '조':
                query = """
                MATCH (a:조 {id: $node_id})-[:HAS_VIOLATION]->(v:위반사례)
                RETURN count(v) as violation_count
                """
            elif node_type == '항':
                query = """
                MATCH (h:항 {id: $node_id})-[:HAS_VIOLATION]->(v:위반사례)
                RETURN count(v) as violation_count
                """
            elif node_type == '호':
                query = """
                MATCH (ho:호 {id: $node_id})-[:HAS_VIOLATION]->(v:위반사례)
                RETURN count(v) as violation_count
                """
            else:
                candidate['graph_score'] = 0.0
                continue
            
            result = self.conn.execute_query(query, {"node_id": node_id})
            violation_count = result[0].get('violation_count', 0) if result else 0
            
            # 그래프 점수 계산 (위반사례가 많을수록 높은 점수, 로그 스케일)
            if violation_count > 0:
                graph_score = min(np.log1p(violation_count) / 5.0, 1.0)  # 최대 1.0
            else:
                graph_score = 0.0
            
            candidate['graph_score'] = graph_score
            candidate['violation_count'] = violation_count
        
        return candidates
    
    def compute_final_score(self, candidates: List[Dict]) -> List[Dict]:
        """
        4단계: 통합 점수 계산 및 재순위화
        
        최종 점수 = (의미적 유사도 × 0.4) + (키워드 점수 × 0.4) + (그래프 점수 × 0.2)
        키워드 매칭 가중치를 높여서 조항별 특성 반영 강화
        """
        for candidate in candidates:
            semantic = candidate.get('semantic_score', 0.0)
            keyword = candidate.get('keyword_score', 0.0)
            graph = candidate.get('graph_score', 0.0)
            
            # 키워드 점수가 높으면 보너스 (패턴 매칭이 있을 때)
            if candidate.get('pattern_matched', False):
                keyword_bonus = 0.1
            else:
                keyword_bonus = 0.0
            
            # 키워드 점수 정규화 (최대 2.0 -> 1.0으로)
            normalized_keyword = min(keyword / 2.0, 1.0) if keyword > 1.0 else keyword
            
            final_score = (
                (semantic * 0.4) +           # 의미적 유사도 (40%)
                (normalized_keyword * 0.4) + # 키워드/패턴 매칭 (40%)
                (graph * 0.2) +              # 그래프 구조 (20%)
                keyword_bonus                # 패턴 매칭 보너스
            )
            candidate['final_score'] = min(final_score, 1.0)  # 최대 1.0
        
        # 최종 점수 기준 정렬
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        하이브리드 검색 메인 함수
        
        Args:
            query: 자연어 질의
            top_k: 반환할 결과 개수
            
        Returns:
            검색 결과 리스트 (최종 점수 기준 정렬)
        """
        # 1단계: 의미적 검색 (Top-K 후보 선별) - 더 넓게 검색
        candidates = self.semantic_search(query, top_k=30)  # 20 -> 30으로 증가
        
        if not candidates:
            return []
        
        # 2단계: 키워드 점수 계산
        candidates = self.keyword_search(query, candidates)
        
        # 3단계: 그래프 구조 점수 계산
        candidates = self.graph_structure_score(candidates)
        
        # 4단계: 통합 점수 계산 및 재순위화
        candidates = self.compute_final_score(candidates)
        
        # 최종 Top-K 반환
        return candidates[:top_k]
    
    def get_node_detail(self, node_id: str, node_type: str) -> Optional[Dict]:
        """노드 상세 정보 조회"""
        query = f"""
        MATCH (n:{node_type} {{id: $node_id}})
        OPTIONAL MATCH (n)-[:HAS_VIOLATION]->(v:위반사례)
        RETURN 
            n.id as id,
            n.content as content,
            n.title as title,
            collect(v.id) as violation_ids,
            count(v) as violation_count
        """
        
        result = self.conn.execute_query(query, {"node_id": node_id})
        return result[0] if result else None

