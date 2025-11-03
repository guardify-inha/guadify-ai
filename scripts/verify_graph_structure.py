"""
위반사례 노드와 조/항/호 노드 간의 관계 구조 검증

1단계: 그래프 모델링 및 시맨틱 보강 - 1.4
"""
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from database.neo4j_connector import Neo4jConnector

def check_node_counts(conn: Neo4jConnector) -> Dict[str, int]:
    """각 노드 타입별 개수 확인"""
    queries = {
        "법률": "MATCH (n:법률) RETURN count(n) as count",
        "조": "MATCH (n:조) RETURN count(n) as count",
        "항": "MATCH (n:항) RETURN count(n) as count",
        "호": "MATCH (n:호) RETURN count(n) as count",
        "위반사례": "MATCH (n:위반사례) RETURN count(n) as count",
        "수정본": "MATCH (n:수정본) RETURN count(n) as count"
    }
    
    counts = {}
    for label, query in queries.items():
        result = conn.execute_query(query)
        counts[label] = result[0]['count'] if result else 0
    
    return counts

def check_relationships(conn: Neo4jConnector) -> Dict[str, int]:
    """관계 타입별 개수 확인"""
    queries = {
        "법률→조": "MATCH (:법률)-[r:HAS_ARTICLE]->(:조) RETURN count(r) as count",
        "조→항": "MATCH (:조)-[r:HAS_HANG]->(:항) RETURN count(r) as count",
        "조→호": "MATCH (:조)-[r:HAS_HO]->(:호) RETURN count(r) as count",
        "항→호": "MATCH (:항)-[r:HAS_HO]->(:호) RETURN count(r) as count",
        "조→위반사례": "MATCH (:조)-[r:HAS_VIOLATION]->(:위반사례) RETURN count(r) as count",
        "항→위반사례": "MATCH (:항)-[r:HAS_VIOLATION]->(:위반사례) RETURN count(r) as count",
        "호→위반사례": "MATCH (:호)-[r:HAS_VIOLATION]->(:위반사례) RETURN count(r) as count",
        "위반사례→수정본": "MATCH (:위반사례)-[r:HAS_CORRECTION]->(:수정본) RETURN count(r) as count"
    }
    
    rel_counts = {}
    for rel_name, query in queries.items():
        result = conn.execute_query(query)
        rel_counts[rel_name] = result[0]['count'] if result else 0
    
    return rel_counts

def check_violation_attributes(conn: Neo4jConnector) -> Dict:
    """위반사례 노드의 속성 확인"""
    # 필수 속성 확인
    query = """
    MATCH (v:위반사례)
    RETURN 
        count(v) as total,
        sum(CASE WHEN v.unfair_text IS NOT NULL AND v.unfair_text <> '' THEN 1 ELSE 0 END) as has_unfair_text,
        sum(CASE WHEN v.reason IS NOT NULL AND v.reason <> '' THEN 1 ELSE 0 END) as has_reason,
        sum(CASE WHEN v.embedding IS NOT NULL THEN 1 ELSE 0 END) as has_embedding,
        sum(CASE WHEN v.id IS NOT NULL THEN 1 ELSE 0 END) as has_id
    """
    
    result = conn.execute_query(query)
    if result:
        return result[0]
    return {}

def check_correction_attributes(conn: Neo4jConnector) -> Dict:
    """수정본 노드의 속성 확인"""
    query = """
    MATCH (c:수정본)
    RETURN 
        count(c) as total,
        sum(CASE WHEN c.corrected_text IS NOT NULL AND c.corrected_text <> '' THEN 1 ELSE 0 END) as has_corrected_text,
        sum(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as has_embedding,
        sum(CASE WHEN c.id IS NOT NULL THEN 1 ELSE 0 END) as has_id
    """
    
    result = conn.execute_query(query)
    if result:
        return result[0]
    return {}

def find_orphaned_violations(conn: Neo4jConnector) -> List[Dict]:
    """조/항/호와 연결되지 않은 위반사례 찾기"""
    query = """
    MATCH (v:위반사례)
    WHERE NOT (v)<-[:HAS_VIOLATION]-()
    RETURN v.id as id, v.unfair_text as text
    LIMIT 10
    """
    
    return conn.execute_query(query)

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🔍 Neo4j 그래프 구조 검증")
    print("=" * 60)
    
    conn = Neo4jConnector()
    
    try:
        # 1. 노드 개수 확인
        print("\n📊 노드 개수:")
        counts = check_node_counts(conn)
        for label, count in counts.items():
            print(f"  • {label}: {count}개")
        
        # 2. 관계 개수 확인
        print("\n🔗 관계 개수:")
        rel_counts = check_relationships(conn)
        for rel_name, count in rel_counts.items():
            print(f"  • {rel_name}: {count}개")
        
        # 3. 위반사례 속성 확인
        print("\n📋 위반사례 노드 속성:")
        violation_attrs = check_violation_attributes(conn)
        if violation_attrs:
            total = violation_attrs.get('total', 0)
            if total > 0:
                print(f"  • 전체: {total}개")
                print(f"  • unfair_text 있음: {violation_attrs.get('has_unfair_text', 0)}개")
                print(f"  • reason 있음: {violation_attrs.get('has_reason', 0)}개")
                print(f"  • embedding 있음: {violation_attrs.get('has_embedding', 0)}개")
                print(f"  • id 있음: {violation_attrs.get('has_id', 0)}개")
            else:
                print("  ⚠️ 위반사례 노드가 없습니다.")
        
        # 4. 수정본 속성 확인
        print("\n📋 수정본 노드 속성:")
        correction_attrs = check_correction_attributes(conn)
        if correction_attrs:
            total = correction_attrs.get('total', 0)
            if total > 0:
                print(f"  • 전체: {total}개")
                print(f"  • corrected_text 있음: {correction_attrs.get('has_corrected_text', 0)}개")
                print(f"  • embedding 있음: {correction_attrs.get('has_embedding', 0)}개")
                print(f"  • id 있음: {correction_attrs.get('has_id', 0)}개")
            else:
                print("  ⚠️ 수정본 노드가 없습니다.")
        
        # 5. 고아 위반사례 확인
        print("\n🔍 고아 위반사례 (연결되지 않은 위반사례):")
        orphans = find_orphaned_violations(conn)
        if orphans:
            print(f"  ⚠️ {len(orphans)}개 발견 (최대 10개 표시):")
            for orphan in orphans:
                print(f"    - {orphan.get('id', 'N/A')}: {orphan.get('text', '')[:50]}...")
        else:
            print("  ✓ 모든 위반사례가 조/항/호와 연결되어 있습니다.")
        
        # 6. 개선 제안
        print("\n💡 개선 제안:")
        suggestions = []
        
        if violation_attrs and violation_attrs.get('has_embedding', 0) < violation_attrs.get('total', 0):
            missing = violation_attrs['total'] - violation_attrs['has_embedding']
            suggestions.append(f"위반사례 노드 {missing}개에 임베딩이 없습니다. 생성 필요.")
        
        if rel_counts.get('조→위반사례', 0) == 0 and rel_counts.get('항→위반사례', 0) == 0 and rel_counts.get('호→위반사례', 0) == 0:
            suggestions.append("위반사례와 조/항/호 간의 관계가 없습니다. 데이터 구축 필요.")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("  ✓ 구조가 양호합니다.")
        
        print("\n" + "=" * 60)
        print("✅ 검증 완료")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()

