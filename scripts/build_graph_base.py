"""
베이스 모델(BAAI/bge-m3)로 Neo4j 그래프 처음부터 구축

사용 시점:
- 처음 그래프를 구축할 때
- 베이스 모델로 재학습하기 전 초기화할 때

주요 기능:
1. 베이스 모델로 이중 임베딩 생성
2. ViolationCase 노드 생성
3. 법률 관계(VIOLATES) 생성
4. 유사도 관계(SIMILAR_TO) 생성
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path
from tqdm import tqdm
import json
import re

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from database.neo4j_connector import Neo4jConnector

# rebuild_graph.py의 GraphRAGRebuilder 클래스 재사용
from scripts.rebuild_graph import GraphRAGRebuilder


def build_graph_with_base_model(csv_paths: List[str]):
    """
    베이스 모델로 그래프 구축
    
    Args:
        csv_paths: CSV 파일 경로 리스트
    """
    print("\n" + "="*80)
    print("🚀 베이스 모델(BAAI/bge-m3)로 그래프 구축 시작")
    print("="*80)
    print(f"📁 CSV 파일: {len(csv_paths)}개")
    for path in csv_paths:
        print(f"   - {path}")
    print()

    # Neo4j 연결
    conn = Neo4jConnector()

    # 베이스 모델로 Rebuilder 생성
    rebuilder = GraphRAGRebuilder(
        neo4j_connector=conn,
        model_path='BAAI/bge-m3'  # 베이스 모델
    )

    # 그래프 구축
    rebuilder.rebuild_from_multiple_csv(csv_paths)

    # 연결 종료
    conn.close()

    print("\n" + "="*80)
    print("✅ 베이스 모델로 그래프 구축 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("1. 파인튜닝 실행:")
    print("   python scripts/train_model.py")
    print("\n2. 파인튜닝 모델로 임베딩 업데이트:")
    print("   python scripts/update_embeddings.py")
    print()


if __name__ == "__main__":
    build_graph_with_base_model([
        "data/contracts/reference/보도자료_데이터_전처리_최종.csv",
        "data/contracts/reference/ai.csv"
    ])
