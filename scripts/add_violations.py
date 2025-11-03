"""
위반 사례 추가 (조 구조 절대 변경 안 함)
- (v3 수정) map_category_to_article에서 매핑 실패 시, '제6조'로 배정하지 않고 'None'을 반환하여 '스킵' 처리.
- (v3 수정) CREATE 대신 MERGE를 사용해 중복 생성 방지 (안정성)
- (v2 수정) build_violation_graph가 여러 CSV 파일 (기존 + ai.csv)을 순차적으로 처리.
"""
import pandas as pd
import json
import os
import sys
from typing import Optional # (v3) Optional 임포트

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from database import Neo4jConnector
except ImportError:
    print("FATAL: Neo4jConnector를 database/neo4j_connector.py에서 찾을 수 없습니다.")
    sys.exit(1)

# 임베딩 모델 로드
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
        """patterns_by_article_v2.json 로드"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        pattern_file = os.path.join(project_root, 'data', 'contracts', 'reference', 'patterns_by_article_v2.json')
        
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
                print(f"✓ 패턴 파일 로드: {pattern_file}")
                return patterns
        except FileNotFoundError:
            print(f"⚠️  패턴 파일 없음: {pattern_file}")
            return {}
        except Exception as e:
            print(f"⚠️  패턴 로드 실패: {e}")
            return {}

    # (신규) CSV 처리 로직을 별도 함수로 분리
    def process_dataframe(self, df: pd.DataFrame) -> dict:
        """DataFrame을 순회하며 위반 사례를 생성합니다."""
        created_violations = 0
        created_corrections = 0
        skipped = 0

        print(f"  {len(df)}개 행 처리 시작...")

        # 각 행 처리
        for idx, row in df.iterrows():
            # CSV 컬럼명 확인
            unfair_text = str(row.get('불공정약관원문', '')) # ai.csv는 '불공정약관원문'
            if not unfair_text:
                unfair_text = str(row.get('불공정 약관 원문', '')) # 기존 CSV는 '불공정 약관 원문'

            reason = str(row.get('시정요청사유', '')) # ai.csv
            if not reason:
                reason = str(row.get('시정 요청 사유', '')) # 기존 CSV

            legal_basis = str(row.get('근거조항(약관법)', '')) # ai.csv
            if not legal_basis:
                legal_basis = str(row.get('근거 조항(약관법)', '')) # 기존 CSV

            correction = str(row.get('수정 후 약관 조항', '')) # 공통

            category = str(row.get('대주제', ''))
            article_id = self.map_category_to_article(category)
            
            # --- (v3 수정) ---
            # article_id가 None (매핑 실패)이면 스킵
            if not article_id:
                skipped += 1
                continue
            # --- (수정 끝) ---
            
            # 조 존재 확인
            if not self.article_exists(article_id):
                print(f"    경고: {article_id}가 DB에 존재하지 않아 건너뜁니다. (main.py 먼저 실행 필요)")
                skipped += 1
                continue
            
            target_node = self.find_lowest_node(article_id)
            if not target_node:
                skipped += 1
                continue
            
            # 위반 사례 생성
            violation_id = self.create_violation_case(
                article_id=article_id,
                target_node=target_node,
                case_id=str(row.get('ID', f"ai_{idx}")), # ai.csv는 ID가 V-001, 기존 CSV는 10001.0
                unfair_text=unfair_text,
                reason=reason,
                legal_basis=legal_basis,
                company=str(row.get('파일명', 'ai_generated')), # ai.csv는 '파일명' 없음
                category=category
            )
            
            if violation_id:
                created_violations += 1
                
                # 수정본
                if correction and correction.strip() and correction != 'nan':
                    self.create_correction(violation_id, correction)
                    created_corrections += 1
            
            if (idx + 1) % 50 == 0:
                print(f"    진행: {idx + 1}/{len(df)}")

        return {
            "created": created_violations,
            "corrections": created_corrections,
            "skipped": skipped
        }

    # (수정) build_violation_graph를 다중 CSV 로더로 변경
    def build_violation_graph(self, csv_path=None):
        """위반 사례 그래프 구축 (다중 CSV 처리)"""
        
        print("\n" + "=" * 70)
        print("📊 위반 사례 추가")
        print("=" * 70)
        
        initial_count = self.count_articles()
        print(f"\n시작 전 조 개수: {initial_count}개")
        
        if initial_count == 0:
            print("❌ 조가 없습니다! main.py를 먼저 실행하세요.")
            return

        # --- (수정) 처리할 CSV 파일 목록 정의 ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        reference_dir = os.path.join(project_root, 'data', 'contracts', 'reference')
        
        csv_files_to_process = []
        if csv_path:
             # 특정 파일만 지정한 경우
            csv_files_to_process.append(csv_path)
            print(f"지정된 단일 파일 처리: {csv_path}")
        else:
            # 기본 파일 목록
            csv_files_to_process.append(os.path.join(reference_dir, '보도자료_데이터_전처리_최종.csv'))
            csv_files_to_process.append(os.path.join(reference_dir, 'ai.csv')) # ai.csv 추가
            print(f"기본 파일 2개 처리: {', '.join([os.path.basename(f) for f in csv_files_to_process])}")
        # --- (수정 끝) ---

        # 통계 (전체 합산)
        total_created_violations = 0
        total_created_corrections = 0
        total_skipped = 0
        
        # 각 CSV 파일 순차 처리
        for path in csv_files_to_process:
            print(f"\n--- {os.path.basename(path)} 처리 시작 ---")
            
            # CSV 로드
            try:
                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(path, encoding='cp949')
                
                print(f"✓ CSV 로드 성공: {len(df)}개 행")
                print(f"  컬럼: {', '.join(df.columns.tolist()[:5])}...")
            except FileNotFoundError:
                print(f"❌ CSV 파일을 찾을 수 없습니다: {path}")
                continue
            except Exception as e:
                print(f"✗ CSV 로드 실패: {e}")
                continue
            
            # (신규) 분리된 헬퍼 함수 호출
            stats = self.process_dataframe(df)
            
            # 통계 합산
            total_created_violations += stats['created']
            total_created_corrections += stats['corrections']
            total_skipped += stats['skipped']

            print(f"--- {os.path.basename(path)} 처리 완료 (위반 {stats['created']} / 수정 {stats['corrections']} / 스킵 {stats['skipped']}) ---")

        
        # ✅ 종료 후 조 개수 확인 (전체 완료 후)
        final_count = self.count_articles()
        
        print(f"\n✅ 모든 CSV 파일 처리 완료!")
        print(f"  총 위반사례: {total_created_violations}개")
        print(f"  총 수정본: {total_created_corrections}개")
        print(f"  총 건너뜀: {total_skipped}개")
        print(f"\n조 개수 확인:")
        print(f"  시작: {initial_count}개")
        print(f"  종료: {final_count}개")
        
        if initial_count != final_count:
            print(f"\n⚠️  경고: 조 개수가 변경되었습니다! ({initial_count} → {final_count})")
        else:
            print(f"  ✅ 조 개수 유지됨")
        
        # 패턴 추가 (모든 사례가 추가된 후 마지막에 한 번만 실행)
        if self.patterns:
            self.add_patterns_to_articles()
        
        self.print_statistics()
    
    def count_articles(self) -> int:
        """조 개수 확인"""
        query = "MATCH (a:조) RETURN count(a) as count"
        result = self.connector.execute_query(query)
        return result[0]['count'] if result else 0
    
    # --- (v3 수정) ---
    def map_category_to_article(self, category: str) -> Optional[str]:
        """카테고리 → 조항 매핑. 실패 시 'None' 반환"""
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
            '계약의 해제': '제9조',
            '변경': '제10조',
            '급부': '제10조',
            '중지': '제10조',
            '채무의 이행': '제10조',
            '기한': '제11조',
            '이익': '제11조',
            '고객의 권익': '제11조',
            '의사표시': '제12조',
            '간주': '제12조',
            '의제': '제12조',
            '대리인': '제13조',
            '소송': '제14조',
            '관할': '제14조',
            '입증': '제14조',
            '일반원칙': '제6조'
        }
        
        # 'nan' 또는 빈 문자열은 즉시 스킵
        if not category or pd.isna(category):
             print(f"    정보: '대주제'가 비어있습니다. 이 항목을 건너뜁니다 (skipped).")
             return None

        for keyword, article in mappings.items():
            if keyword in category:
                return article
        
        # 매핑되는 키워드가 없으면 '제6조' 대신 None 반환
        print(f"    정보: '{category}'에 매핑되는 키워드 없음. 이 항목을 건너뜁니다 (skipped).")
        return None
    # --- (수정 끝) ---
    
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
        """(v3 수정) 위반 사례 생성 (MERGE 사용, 조는 건드리지 않음)"""
        
        # 패턴 매칭 (v2 JSON 구조에 맞춤)
        patterns = []
        pattern_keywords = []
        high_risk_keywords = []
        risk_level = "medium" # 기본값
        
        if article_id in self.patterns:
            article_patterns = self.patterns[article_id].get('patterns', [])
            for p in article_patterns:
                # 기본 키워드 체크
                if any(kw in unfair_text for kw in p.get('keywords', [])):
                    patterns.append(p.get('description', ''))
                    pattern_keywords.extend(p.get('keywords', []))
                    
                    # 고위험 키워드 체크
                    high_risk = p.get('high_risk_keywords', [])
                    if any(kw in unfair_text for kw in high_risk):
                        high_risk_keywords.extend(high_risk)
                        # 패턴에 지정된 위험도 반영
                        risk_level = p.get('risk_level', 'medium')
        
        # 범용 위험 키워드 체크 (universal_risk_keywords)
        universal_risks = self.patterns.get('universal_risk_keywords', {})
        if universal_risks:
            for risk_item in universal_risks.get('keywords', []):
                keyword = risk_item.get('keyword', '')
                if keyword and keyword in unfair_text:
                    high_risk_keywords.append(keyword)
                    # 범용 키워드 위험도와 기존 위험도 중 더 높은 것을 선택
                    current_risk_num = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(risk_level, 2)
                    new_risk_num = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(risk_item.get('risk_level'), 2)
                    if new_risk_num > current_risk_num:
                        risk_level = risk_item.get('risk_level')
        
        # 임베딩
        embedding = None
        if EMBEDDING_AVAILABLE and MODEL:
            try:
                embedding = MODEL.encode(unfair_text).tolist()
            except:
                pass
        
        # case_id가 'V-'로 시작하면 ai.csv 데이터로 간주, 아니면 기존 데이터 ID
        violation_id = f"violation_{case_id}_{article_id}"
        
        # MATCH만 사용 (조/항/호는 절대 생성/삭제 안 함)
        if target_node["type"] == "호":
            match_clause = "MATCH (n:호 {id: $target_id})"
        elif target_node["type"] == "항":
            match_clause = "MATCH (n:항 {id: $target_id})"
        else:
            match_clause = "MATCH (n:조 {id: $target_id})"
        
        # (v3 수정) CREATE 대신 MERGE를 사용하여 중복 ID 생성 방지
        query = f"""
        {match_clause}
        MERGE (v:위반사례 {{id: $violation_id}})
        ON CREATE SET
            v.case_id = $case_id,
            v.article_id = $article_id,
            v.unfair_text = $unfair_text,
            v.reason = $reason,
            v.legal_basis = $legal_basis,
            v.company = $company,
            v.category = $category,
            v.patterns = $patterns,
            v.pattern_keywords = $pattern_keywords,
            v.high_risk_keywords = $high_risk_keywords,
            v.risk_level = $risk_level,
            v.embedding = $embedding
        MERGE (n)-[r:HAS_VIOLATION]->(v)
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
                "high_risk_keywords": list(set(high_risk_keywords)),
                "risk_level": risk_level,
                "embedding": embedding
            })
            return result[0]["id"] if result else None
        except Exception as e:
            # 중복 ID 등으로 인한 UNIQUE 제약 조건 위반일 수 있음
            if "already exists" in str(e):
                # 이미 존재하면 MERGE가 처리했어야 하므로 사실상 에러지만, 일단 경고만
                print(f"    경고: 위반사례 ID '{violation_id}' 처리 중 문제 발생 (Maybe exists). 건너뜁니다.")
            else:
                print(f"    ✗ 생성 실패 ({case_id}): {e}")
            return None
    
    def create_correction(self, violation_id, corrected_text):
        """수정본 생성"""
        embedding = None
        if EMBEDDING_AVAILABLE and MODEL:
            try:
                embedding = MODEL.encode(corrected_text).tolist()
            except:
                pass
        
        correction_id = f"{violation_id}_corrected"

        # (v3 수정) MERGE를 사용하여 중복 생성을 방지
        query = """
        MATCH (v:위반사례 {id: $violation_id})
        MERGE (c:수정본 {id: $correction_id})
        ON CREATE SET
            c.violation_id = $violation_id,
            c.corrected_text = $corrected_text,
            c.embedding = $embedding
        MERGE (v)-[r:HAS_CORRECTION]->(c)
        """
        
        try:
            self.connector.execute_query(query, {
                "violation_id": violation_id,
                "correction_id": correction_id,
                "corrected_text": corrected_text[:1000],
                "embedding": embedding
            })
        except:
            pass
    
    def add_patterns_to_articles(self):
        """조항에 패턴 추가 (SET만 사용, 생성/삭제 없음)"""
        print("\n패턴 정보를 조에 추가 중...")
        
        for article_id, data in self.patterns.items():
            if article_id not in ['제6조', '제7조', '제8조', '제9조', '제10조', '제11조', '제12조', '제13조', '제14조']:
                continue
            
            descriptions = [p.get('description', '') for p in data.get('patterns', [])]
            keywords = []
            high_risk_keywords = []
            
            for p in data.get('patterns', []):
                keywords.extend(p.get('keywords', []))
                high_risk_keywords.extend(p.get('high_risk_keywords', []))
            
            query = """
            MATCH (a:조 {id: $article_id})
            SET a.patterns = $patterns,
                a.pattern_keywords = $keywords,
                a.high_risk_keywords = $high_risk_keywords,
                a.title = $title,
                a.case_count = $case_count
            """
            
            try:
                self.connector.execute_query(query, {
                    "article_id": article_id,
                    "patterns": descriptions,
                    "keywords": list(set(keywords)),
                    "high_risk_keywords": list(set(high_risk_keywords)),
                    "title": data.get('title', ''),
                    "case_count": data.get('case_count', 0)
                })
            except Exception as e:
                print(f"    ✗ 패턴 추가 실패 ({article_id}): {e}")
        
        print("✓ 패턴 추가 완료")
    
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
            print(f"  {name}: {result[0]['count']}개")
        
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
            print(f"  {row['article']} ({row['title']}): {row['count']}개")
        
        print("\n위험도별 위반사례:")
        query = """
        MATCH (v:위반사례)
        WITH v.risk_level as level, count(v) as count
        RETURN level, count
        ORDER BY 
            CASE level
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                WHEN 'low' THEN 4
                ELSE 5
            END
        """
        result = self.connector.execute_query(query)
        for row in result:
            level_emoji = {
                'critical': '⚫', 'high': '🔴', 'medium': '🟡', 'low': '🟢'
            }
            emoji = level_emoji.get(row['level'], '⚪')
            print(f"  {emoji} {str(row['level']).upper()}: {row['count']}개")


def main():
    print("=" * 70)
    print("위반 사례 추가 (조 구조 유지, 다중 CSV 처리)")
    print("=" * 70)
    
    connector = Neo4jConnector()
    
    try:
        builder = ViolationCaseBuilder(connector)
        # 인자 없이 호출하면 기본 목록 (기존 CSV + ai.csv)을 처리
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

