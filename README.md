
법률중심 그래프 사용

# Step 1: 환경 설정
pip install -r requirements.txt

# Step 2: Neo4j 실행
docker run -d --name neo4j-graphrag \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/testpassword123 \
    -e NEO4J_PLUGINS='["apoc"]' \
    neo4j:5.14.0

# Step 3: 법률 그래프 먼저 구축 (Law-Centric의 핵심!)
python main.py 

# Step 4: 벡터 인덱스 생성
python scripts/setup_vector_indexes.py

# Step 5: GraphRAG 그래프 구축 (이게 핵심!)
python -c '
from pipeline.graph_builder import GraphRAGBuilder
from database.neo4j_connector import Neo4jConnector
conn = Neo4jConnector()
builder = GraphRAGBuilder(conn)
builder.build_from_multiple_csv([
    "data/contracts/reference/보도자료_데이터_전처리_최종.csv",
    "data/contracts/reference/ai.csv"
])
conn.close()
print("✅ Law-Centric GraphRAG 구축 완료!")
'



# Step 6: 테스트
streamlit run scripts/app.py



_____
finetuning 적용버전 실행순서


python main.py (그래프 리셋하려고)
python scripts/rebuild_graph.py
python scripts/train_model.py


혼동행렬 테스트(학습데이터)
python scripts/test_ai_csv.py

혼동행렬 테스트(테스트데이터)
python scripts/test_test_input.py

조항별 테스트?
python scripts/test_with_test_input.py