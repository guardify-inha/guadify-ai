"""
BAAI/bge-m3 모델 파인튜닝 스크립트
Neo4j에서 데이터를 추출하여 Contrastive Learning으로 학습

학습 전략:
- Anchor: original_text (위반 문장)
- Positive: 같은 article_id를 가진 다른 위반 문장
- Negative: 해당 노드의 corrected_text (반대 의미)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import random
from collections import defaultdict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from database.neo4j_connector import Neo4jConnector
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

class ContractViolationDataset:
    """약관 위반 탐지용 데이터셋"""

    def __init__(self, neo4j_connector):
        self.conn = neo4j_connector
        self.data_by_article = defaultdict(list)
        self.all_data = []

    def load_from_neo4j(self):
        """Neo4j에서 데이터 로드"""
        print("="*80)
        print("📊 Neo4j에서 데이터 로드 중...")
        print("="*80)

        query = """
        MATCH (v:ViolationCase)
        WHERE v.original_text IS NOT NULL
          AND v.corrected_text IS NOT NULL
          AND trim(v.corrected_text) <> ''
        RETURN v.id as id,
               v.original_text as original_text,
               v.corrected_text as corrected_text,
               v.article_id as article_id
        """

        results = self.conn.execute_query(query)

        print(f"✅ {len(results)}개 케이스 로드 완료\n")

        # article_id별로 그룹화
        for r in results:
            data = {
                'id': r['id'],
                'original_text': r['original_text'],
                'corrected_text': r['corrected_text'],
                'article_id': r['article_id']
            }
            self.all_data.append(data)

            if r['article_id']:
                self.data_by_article[r['article_id']].append(data)

        # 통계 출력
        print("📈 데이터 분포:")
        print(f"  전체 케이스: {len(self.all_data)}개")
        print(f"  조항 종류: {len(self.data_by_article)}개")
        print(f"\n조항별 케이스 수:")
        for article_id, cases in sorted(self.data_by_article.items(),
                                       key=lambda x: len(x[1]),
                                       reverse=True)[:10]:
            print(f"    {article_id}: {len(cases)}개")
        print()

        return len(self.all_data)

    def create_triplets(self, num_samples_per_case: int = 3) -> List[InputExample]:
        """
        Triplet 생성 for Contrastive Learning

        전략:
        - Anchor: original_text (위반 문장)
        - Positive: 같은 article_id의 다른 original_text
        - Negative: 해당 케이스의 corrected_text (반대 의미)

        Args:
            num_samples_per_case: 각 케이스당 생성할 샘플 수
        """
        print("="*80)
        print("🔄 Training Triplets 생성 중...")
        print("="*80)

        examples = []

        for case in self.all_data:
            anchor = case['original_text']
            negative = case['corrected_text']
            article_id = case['article_id']

            # 같은 조항의 다른 케이스들
            same_article_cases = self.data_by_article.get(article_id, [])

            # 자기 자신 제외
            same_article_cases = [c for c in same_article_cases if c['id'] != case['id']]

            if not same_article_cases:
                # Positive를 찾을 수 없으면 자기 자신을 사용 (차선책)
                positive = anchor
            else:
                # 여러 개의 positive 샘플 생성
                for _ in range(min(num_samples_per_case, len(same_article_cases))):
                    positive_case = random.choice(same_article_cases)
                    positive = positive_case['original_text']

                    # InputExample: (texts=[anchor, positive], label=1.0)
                    examples.append(InputExample(
                        texts=[anchor, positive],
                        label=1.0  # 같은 조항 = 유사
                    ))

                    # Hard Negative도 추가 (corrected_text)
                    examples.append(InputExample(
                        texts=[anchor, negative],
                        label=0.0  # 반대 의미 = 다름
                    ))

        print(f"✅ {len(examples)}개 Training Examples 생성 완료")
        print(f"   - Positive pairs: {len([e for e in examples if e.label == 1.0])}개")
        print(f"   - Negative pairs: {len([e for e in examples if e.label == 0.0])}개")
        print()

        # 셔플
        random.shuffle(examples)

        return examples


def train_bge_m3_model(
    output_dir: str = "./my_fine_tuned_model",
    base_model: str = "BAAI/bge-m3",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100
):
    """
    BAAI/bge-m3 모델 파인튜닝

    Args:
        output_dir: 학습된 모델 저장 경로
        base_model: 베이스 모델 이름
        num_epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        warmup_steps: Warmup 스텝 수
    """
    print("="*80)
    print("🚀 BAAI/bge-m3 Fine-tuning 시작")
    print("="*80)
    print(f"📦 베이스 모델: {base_model}")
    print(f"💾 저장 경로: {output_dir}")
    print(f"🔧 설정:")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Learning Rate: {learning_rate}")
    print(f"   - Warmup Steps: {warmup_steps}")
    print()

    # 1. Neo4j 연결
    print("🔌 Neo4j 연결 중...")
    conn = Neo4jConnector()

    # 2. 데이터셋 로드
    dataset = ContractViolationDataset(conn)
    num_cases = dataset.load_from_neo4j()

    if num_cases == 0:
        print("❌ 데이터가 없습니다. Neo4j 연결과 데이터를 확인하세요.")
        conn.close()
        return

    # 3. Triplets 생성
    train_examples = dataset.create_triplets(num_samples_per_case=3)

    # 4. 모델 로드
    print("="*80)
    print(f"🧠 {base_model} 모델 로드 중...")
    print("="*80)

    model = SentenceTransformer(base_model)

    print(f"✅ 모델 로드 완료")
    print(f"   - 임베딩 차원: {model.get_sentence_embedding_dimension()}차원")
    print()

    # 5. DataLoader 생성
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )

    # 6. Loss Function
    # CosineSimilarityLoss: (anchor, positive/negative, label) 형태 지원
    train_loss = losses.CosineSimilarityLoss(model)

    print("="*80)
    print("🎯 학습 시작")
    print("="*80)
    print(f"📊 Training Examples: {len(train_examples)}개")
    print(f"📦 Batches per Epoch: {len(train_dataloader)}개")
    print()

    # 7. 학습
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
        save_best_model=True,
        optimizer_params={'lr': learning_rate}
    )

    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print(f"💾 모델 저장 위치: {output_dir}")
    print()

    # 8. 테스트
    print("🧪 모델 테스트...")
    test_texts = [
        "회사는 귀책사유 없이 일체의 책임을 지지 않습니다.",
        "회사는 고의 또는 중대한 과실로 인한 손해를 배상합니다."
    ]

    embeddings = model.encode(test_texts)
    print(f"✅ 테스트 임베딩 생성 완료: {embeddings.shape}")
    print(f"   문장 1 (위반): {test_texts[0][:30]}...")
    print(f"   문장 2 (준수): {test_texts[1][:30]}...")
    print(f"   임베딩 차원: {embeddings.shape[1]}차원")
    print()

    # 9. 정리
    conn.close()

    print("="*80)
    print("🎉 모든 작업 완료!")
    print("="*80)
    print(f"다음 단계: scripts/rebuild_graph.py 실행하여 Neo4j 그래프 재구성")
    print()


if __name__ == "__main__":
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 학습 실행
    train_bge_m3_model(
        output_dir="./my_fine_tuned_model",
        base_model="BAAI/bge-m3",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100
    )
