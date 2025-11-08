
✅ database/graph_builder.py는 “법률 구조” (조, 항, 호 등 법 자체 구조)
✅ pipeline/graph_builder.py는 “데이터셋 (CSV)” 기반 그래프

# Step 1: 환경 설정
pip install -r requirements.txt

# Step 2: Neo4j 실행
docker run -d --name neo4j-graphrag \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/testpassword123 \
    -e NEO4J_PLUGINS='["apoc"]' \
    neo4j:5.14.0

# Step 3: 벡터 인덱스 생성
python scripts/setup_vector_indexes.py

# Step 4: GraphRAG 그래프 구축 (이게 핵심!)
python -c '\
from pipeline.graph_builder import GraphRAGBuilder; \
from database.neo4j_connector import Neo4jConnector; \
conn = Neo4jConnector(); \
builder = GraphRAGBuilder(conn); \
builder.build_from_multiple_csv(["data/contracts/reference/보도자료_데이터_전처리_최종.csv", "data/contracts/reference/ai.csv"]); \
conn.close(); \
print("✅ 두 개의 CSV 기반 GraphRAG 구축 완료!"); \
'




# Step 5: 테스트
streamlit run scripts/app.py