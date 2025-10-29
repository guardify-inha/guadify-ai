"""
법률 구조를 Neo4j 그래프로 변환
"""
from database.neo4j_connector import Neo4jConnector
from data.law_structure import LAW_STRUCTURE
from config.settings import settings


class GraphBuilder:
    """법률 구조를 그래프 데이터베이스에 구축"""
    
    def __init__(self, connector: Neo4jConnector):
        """
        Args:
            connector: Neo4j 연결 객체
        """
        self.connector = connector
    
    def build_law_graph(self):
        """법률 전체 구조를 그래프로 구축"""
        print("\n📚 법률 그래프 구축 시작...")
        
        # 1. 법률 노드 생성
        law_id = self.create_law_node()
        print(f"  ✓ 법률 노드 생성: {settings.LAW_NAME}")
        
        # 2. 조, 항, 호 노드 생성 및 연결
        total_articles = len(LAW_STRUCTURE)
        for idx, (article_id, article_data) in enumerate(LAW_STRUCTURE.items(), 1):
            self.create_article_structure(law_id, article_id, article_data)
            print(f"  ✓ {article_id} 구조 생성 완료 ({idx}/{total_articles})")
        
        print("\n✅ 법률 그래프 구축 완료!")
        self.print_statistics()
    
    def create_law_node(self):
        """법률 노드 생성"""
        query = """
        CREATE (law:법률 {
            id: $law_id,
            name: $name
        })
        RETURN law.id as id
        """
        result = self.connector.execute_query(query, {
            "law_id": "약관규제법",
            "name": settings.LAW_NAME
        })
        return result[0]["id"]
    
    def create_article_structure(self, law_id, article_id, article_data):
        """
        조 및 하위 항, 호 구조 생성
        
        Args:
            law_id: 법률 ID
            article_id: 조 ID (예: "제6조")
            article_data: 조 데이터
        """
        # 1. 조 노드 생성
        article_node_id = self.create_article_node(
            law_id, article_id, article_data["title"], article_data["content"]
        )
        
        # 2. 항 노드 생성
        for hang_id, hang_data in article_data["항들"].items():
            hang_node_id = self.create_hang_node(
                article_node_id, article_id, hang_id, hang_data["content"]
            )
            
            # 3. 호 노드 생성 (있는 경우)
            if hang_data["호들"]:
                for ho_id, ho_content in hang_data["호들"].items():
                    self.create_ho_node(
                        hang_node_id, article_id, hang_id, ho_id, ho_content
                    )
    
    def create_article_node(self, law_id, article_id, title, content):
        """조 노드 생성 및 법률과 연결"""
        query = """
        MATCH (law:법률 {id: $law_id})
        CREATE (article:조 {
            id: $article_id,
            title: $title,
            content: $content
        })
        CREATE (law)-[:HAS_ARTICLE]->(article)
        RETURN article.id as id
        """
        result = self.connector.execute_query(query, {
            "law_id": law_id,
            "article_id": article_id,
            "title": title,
            "content": content
        })
        return result[0]["id"]
    
    def create_hang_node(self, article_id, article_num, hang_id, content):
        """항 노드 생성 및 조와 연결"""
        query = """
        MATCH (article:조 {id: $article_id})
        CREATE (hang:항 {
            id: $hang_full_id,
            article_id: $article_id,
            hang_num: $hang_id,
            content: $content
        })
        CREATE (article)-[:HAS_HANG]->(hang)
        RETURN hang.id as id
        """
        hang_full_id = f"{article_num}_{hang_id}"
        result = self.connector.execute_query(query, {
            "article_id": article_id,
            "hang_full_id": hang_full_id,
            "hang_id": hang_id,
            "content": content
        })
        return result[0]["id"]
    
    def create_ho_node(self, hang_id, article_num, hang_num, ho_id, content):
        """호 노드 생성 및 항과 연결"""
        query = """
        MATCH (hang:항 {id: $hang_id})
        CREATE (ho:호 {
            id: $ho_full_id,
            hang_id: $hang_id,
            ho_num: $ho_id,
            content: $content
        })
        CREATE (hang)-[:HAS_HO]->(ho)
        RETURN ho.id as id
        """
        ho_full_id = f"{article_num}_{hang_num}_{ho_id}"
        self.connector.execute_query(query, {
            "hang_id": hang_id,
            "ho_full_id": ho_full_id,
            "ho_id": ho_id,
            "content": content
        })
    
    def print_statistics(self):
        """그래프 통계 출력"""
        queries = {
            "법률": "MATCH (n:법률) RETURN count(n) as count",
            "조": "MATCH (n:조) RETURN count(n) as count",
            "항": "MATCH (n:항) RETURN count(n) as count",
            "호": "MATCH (n:호) RETURN count(n) as count"
        }
        
        print("\n📊 생성된 노드 통계:")
        for node_type, query in queries.items():
            result = self.connector.execute_query(query)
            count = result[0]["count"]
            print(f"  • {node_type}: {count}개")
