"""
위반 사례 추가 (조 구조 절대 변경 안 함)
"""
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Neo4jConnector

# 임베딩
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    EMBEDDING_AVAILABLE = True
    print("✓ 임베딩 모델 로드")
except Exception as e:
    MODEL = None
    EMBEDDING_AVAILABLE = False
    print(f"⚠️  임베딩 없음: {e}")


class ViolationCaseBuilder:
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
        self.patterns = self.load_patterns()
    
    def load_patterns(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        pattern_file = os.path.join(project_root, 'data', 'contracts', 'reference', 'patterns_by_article.json')
        
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def build_violation_graph(self, csv_path=None):
        if csv_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            csv_path = os.path.join(project_root, 'data', 'contracts', 'reference', 'corrected_terms.csv')
        
        print("\n" + "=" * 70)
        print("📊 위반 사례 추가")
        print("=" * 70)
        
        # ✅ 시작 전 조 개수 확인
        initial_count = self.count_articles()
        print(f"\n시작 전 조 개수: {initial_count}개")
        
        if initial_count == 0:
            print("❌ 조가 없습니다! main.py를 먼저 실행하세요.")
            return
        
        # CSV 로드
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            print(f"✓ CSV 로드: {len(df)}개 행")
        except Exception as e:
            print(f"✗ CSV 로드 실패: {e}")
            return
        
        # 통계
        created_violations = 0
        created_corrections = 0
        skipped = 0
        
        print("\n위반 사례 추가 중...")
        
        # 각 행 처리
        for idx, row in df.iterrows():
            category = str(row.get('대주제', ''))
            article_id = self.map_category_to_article(category)
            
            if not article_id:
                skipped += 1
                continue
            
            # ✅ 조 존재 확인 (삭제 없이 확인만)
            if not self.article_exists(article_id):
                skipped += 1
                continue
            
            # 가장 하위 노드 찾기 (조는 건드리지 않음)
            target_node = self.find_lowest_node(article_id)
            
            if not target_node:
                skipped += 1
                continue
            
            # 위반 사례 생성 (CREATE만 사용, MERGE 없음)
            violation_id = self.create_violation_case(
                article_id=article_id,
                target_node=target_node,
                case_id=str(row.get('ID', idx)),
                unfair_text=str(row.get('불공정 약관 원문', '')),
                reason=str(row.get('시정 요청 사유', '')),
                legal_basis=str(row.get('근거 조항', '')),
                company=str(row.get('파일명', '')),
                category=category
            )
            
            if violation_id:
                created_violations += 1
                
                # 수정본
                correction = str(row.get('수정 후 약관 조항', ''))
                if correction and correction.strip() and correction != 'nan':
                    self.create_correction(violation_id, correction)
                    created_corrections += 1
            
            if (idx + 1) % 20 == 0:
                print(f"   진행: {idx + 1}/{len(df)}")
        
        # ✅ 종료 후 조 개수 확인
        final_count = self.count_articles()
        
        print(f"\n✅ 완료!")
        print(f"   위반사례: {created_violations}개")
        print(f"   수정본: {created_corrections}개")
        print(f"   건너뜀: {skipped}개")
        print(f"\n조 개수 확인:")
        print(f"   시작: {initial_count}개")
        print(f"   종료: {final_count}개")
        
        if initial_count != final_count:
            print(f"\n⚠️  경고: 조 개수가 변경되었습니다! ({initial_count} → {final_count})")
        else:
            print(f"   ✅ 조 개수 유지됨")
        
        # 패턴 추가 (조 속성만 변경, 삭제/생성 없음)
        if self.patterns:
            self.add_patterns_to_articles()
        
        self.print_statistics()
    
    def count_articles(self) -> int:
        """조 개수 확인"""
        query = "MATCH (a:조) RETURN count(a) as count"
        result = self.connector.execute_query(query)
        return result[0]['count'] if result else 0
    
    def map_category_to_article(self, category: str) -> str:
        """카테고리 → 조항 매핑"""
        mappings = {
            '면책': '제7조',
            '책임': '제7조',
            '배상': '제7조',
            '담보': '제7조',
            '손해배상': '제8조',
            '지연': '제8조',
            '위약': '제8조',
            '해제': '제9조',
            '해지': '제9조',
            '변경': '제10조',
            '급부': '제10조',
            '중지': '제10조',
            '기한': '제11조',
            '이익': '제11조',
            '의사표시': '제12조',
            '간주': '제12조',
            '의제': '제12조',
            '대리인': '제13조',
            '소송': '제14조',
            '관할': '제14조',
            '입증': '제14조',
            '일반원칙': '제6조'
        }
        
        for keyword, article in mappings.items():
            if keyword in category:
                return article
        
        return '제6조'
    
    def article_exists(self, article_id: str) -> bool:
        """조 존재 확인 (읽기 전용)"""
        query = "MATCH (a:조 {id: $article_id}) RETURN count(a) as count"
        result = self.connector.execute_query(query, {"article_id": article_id})
        return result[0]['count'] > 0 if result else False
    
    def find_lowest_node(self, article_id: str):
        """가장 하위 노드 찾기 (읽기 전용)"""
        # 호
        query = """
        MATCH (a:조 {id: $article_id})-[:HAS_HANG]->(h:항)-[:HAS_HO]->(o:호)
        RETURN o.id as node_id LIMIT 1
        """
        result = self.connector.execute_query(query, {"article_id": article_id})
        if result:
            return {"id": result[0]["node_id"], "type": "호"}
        
        # 항
        query = """
        MATCH (a:조 {id: $article_id})-[:HAS_HANG]->(h:항)
        RETURN h.id as node_id LIMIT 1
        """
        result = self.connector.execute_query(query, {"article_id": article_id})
        if result:
            return {"id": result[0]["node_id"], "type": "항"}
        
        # 조
        return {"id": article_id, "type": "조"}
    
    def create_violation_case(self, article_id, target_node, case_id, 
                               unfair_text, reason, legal_basis, company, category):
        """위반 사례 생성 (CREATE만 사용, 조는 건드리지 않음)"""
        
        # 패턴
        patterns = []
        pattern_keywords = []
        if article_id in self.patterns:
            for p in self.patterns[article_id].get('patterns', []):
                if any(kw in unfair_text for kw in p['keywords']):
                    patterns.append(p['description'])
                    pattern_keywords.extend(p['keywords'])
        
        # 임베딩
        embedding = None
        if EMBEDDING_AVAILABLE and MODEL:
            try:
                embedding = MODEL.encode(unfair_text).tolist()
            except:
                pass
        
        violation_id = f"violation_{case_id}_{article_id}"
        
        # ✅ MATCH만 사용 (조/항/호는 절대 생성/삭제 안 함)
        if target_node["type"] == "호":
            match_clause = "MATCH (n:호 {id: $target_id})"
        elif target_node["type"] == "항":
            match_clause = "MATCH (n:항 {id: $target_id})"
        else:
            match_clause = "MATCH (n:조 {id: $target_id})"
        
        # ✅ CREATE로 위반사례만 생성
        query = f"""
        {match_clause}
        CREATE (v:위반사례 {{
            id: $violation_id,
            case_id: $case_id,
            article_id: $article_id,
            unfair_text: $unfair_text,
            reason: $reason,
            legal_basis: $legal_basis,
            company: $company,
            category: $category,
            patterns: $patterns,
            pattern_keywords: $pattern_keywords,
            embedding: $embedding
        }})
        CREATE (n)-[:HAS_VIOLATION]->(v)
        RETURN v.id as id
        """
        
        try:
            result = self.connector.execute_query(query, {
                "target_id": target_node["id"],
                "violation_id": violation_id,
                "case_id": case_id,
                "article_id": article_id,
                "unfair_text": unfair_text[:1000],
                "reason": reason[:500],
                "legal_basis": legal_basis[:200],
                "company": company[:100],
                "category": category[:100],
                "patterns": patterns,
                "pattern_keywords": list(set(pattern_keywords)),
                "embedding": embedding
            })
            return result[0]["id"] if result else None
        except Exception as e:
            print(f"   ✗ 생성 실패: {e}")
            return None
    
    def create_correction(self, violation_id, corrected_text):
        """수정본 생성"""
        embedding = None
        if EMBEDDING_AVAILABLE and MODEL:
            try:
                embedding = MODEL.encode(corrected_text).tolist()
            except:
                pass
        
        query = """
        MATCH (v:위반사례 {id: $violation_id})
        CREATE (c:수정본 {
            id: $correction_id,
            violation_id: $violation_id,
            corrected_text: $corrected_text,
            embedding: $embedding
        })
        CREATE (v)-[:HAS_CORRECTION]->(c)
        """
        
        try:
            self.connector.execute_query(query, {
                "violation_id": violation_id,
                "correction_id": f"{violation_id}_corrected",
                "corrected_text": corrected_text[:1000],
                "embedding": embedding
            })
        except:
            pass
    
    def add_patterns_to_articles(self):
        """조항에 패턴 추가 (SET만 사용, 생성/삭제 없음)"""
        for article_id, data in self.patterns.items():
            descriptions = [p['description'] for p in data.get('patterns', [])]
            keywords = []
            for p in data.get('patterns', []):
                keywords.extend(p['keywords'])
            
            # ✅ SET만 사용 (조를 건드리지 않고 속성만 추가)
            query = """
            MATCH (a:조 {id: $article_id})
            SET a.patterns = $patterns,
                a.pattern_keywords = $keywords
            """
            
            try:
                self.connector.execute_query(query, {
                    "article_id": article_id,
                    "patterns": descriptions,
                    "keywords": list(set(keywords))
                })
            except:
                pass
    
    def print_statistics(self):
        """통계"""
        print("\n" + "=" * 70)
        print("📊 최종 통계")
        print("=" * 70)
        
        queries = {
            "조": "MATCH (n:조) RETURN count(n) as count",
            "항": "MATCH (n:항) RETURN count(n) as count",
            "호": "MATCH (n:호) RETURN count(n) as count",
            "위반사례": "MATCH (n:위반사례) RETURN count(n) as count",
            "수정본": "MATCH (n:수정본) RETURN count(n) as count"
        }
        
        for name, query in queries.items():
            result = self.connector.execute_query(query)
            print(f"   {name}: {result[0]['count']}개")
        
        # 조항별
        print("\n조항별 위반사례:")
        query = """
        MATCH (a:조)
        OPTIONAL MATCH (a)-[:HAS_VIOLATION]->(v1)
        OPTIONAL MATCH (a)-[:HAS_HANG]->()-[:HAS_VIOLATION]->(v2)
        OPTIONAL MATCH (a)-[:HAS_HANG]->()-[:HAS_HO]->()-[:HAS_VIOLATION]->(v3)
        WITH a, count(DISTINCT v1) + count(DISTINCT v2) + count(DISTINCT v3) as total
        RETURN a.id as article, a.title as title, total as count
        ORDER BY a.id
        """
        
        result = self.connector.execute_query(query)
        for row in result:
            print(f"   {row['article']} ({row['title']}): {row['count']}개")


def main():
    print("=" * 70)
    print("위반 사례 추가 (조 구조 유지)")
    print("=" * 70)
    
    connector = Neo4jConnector()
    
    try:
        builder = ViolationCaseBuilder(connector)
        builder.build_violation_graph()
        
        print("\n✅ 완료!")
        
    except Exception as e:
        print(f"\n✗ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        connector.close()


if __name__ == "__main__":
    main()