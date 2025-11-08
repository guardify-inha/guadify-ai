"""법률 용어 사전 벡터 스토어 구축 스크립트"""
import os
import sys
import json
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings


def load_legal_dictionary() -> list[dict]:
    """
    법률 용어 사전 로드
    
    실제 사용 시에는 data/dictionary/legal_terms.json 파일을 준비해야 합니다.
    형식: [{"term": "용어", "explanation": "쉬운 설명"}, ...]
    """
    dict_path = Path(settings.dictionary_data_path) / "legal_terms.json"
    
    if not dict_path.exists():
        print(f"경고: {dict_path} 파일이 존재하지 않습니다.")
        print("샘플 데이터를 생성합니다...")
        return create_sample_dictionary()
    
    with open(dict_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_sample_dictionary() -> list[dict]:
    """샘플 법률 용어 사전 생성"""
    sample_dict = [
        {
            "term": "갑(甲)",
            "explanation": "계약에서 돈이나 물건을 주는 쪽, 또는 계약을 제안하는 쪽을 의미합니다."
        },
        {
            "term": "을(乙)",
            "explanation": "계약에서 돈이나 물건을 받는 쪽, 또는 계약 제안을 받는 쪽을 의미합니다."
        },
        {
            "term": "채권",
            "explanation": "다른 사람(채무자)에게 돈을 받을 권리입니다."
        },
        {
            "term": "채무",
            "explanation": "다른 사람(채권자)에게 돈을 지급해야 할 의무입니다."
        },
        {
            "term": "손해배상",
            "explanation": "계약을 위반하거나 잘못을 저질러서 다른 사람에게 손해를 입혔을 때, 그 손해를 돈으로 보상하는 것입니다."
        },
        {
            "term": "위약금",
            "explanation": "계약을 어겼을 때 미리 정해진 금액을 지급하는 것입니다."
        },
        {
            "term": "해지",
            "explanation": "계약을 끝내는 것입니다."
        },
        {
            "term": "해제",
            "explanation": "계약을 처음부터 없었던 것으로 만드는 것입니다."
        },
        {
            "term": "이행",
            "explanation": "계약에서 정한 내용을 실제로 실행하는 것입니다."
        },
        {
            "term": "불이행",
            "explanation": "계약에서 정한 내용을 실행하지 않는 것입니다."
        },
        {
            "term": "과실",
            "explanation": "주의를 기울이지 않아서 실수하거나 잘못을 저지른 것입니다."
        },
        {
            "term": "고의",
            "explanation": "일부러, 의도적으로 행동하는 것입니다."
        },
        {
            "term": "면제",
            "explanation": "책임이나 의무를 없애주는 것입니다."
        },
        {
            "term": "소멸시효",
            "explanation": "일정 기간이 지나면 권리를 행사할 수 없게 되는 제도입니다."
        },
        {
            "term": "소송",
            "explanation": "법원에 가서 다툼을 해결하는 절차입니다."
        },
        {
            "term": "중재",
            "explanation": "법원이 아닌 제3자(중재인)에게 분쟁을 해결해 달라고 요청하는 것입니다."
        },
        {
            "term": "관할",
            "explanation": "어떤 법원이 그 사건을 처리할 수 있는지 정하는 것입니다."
        },
        {
            "term": "일방적",
            "explanation": "한쪽만의 의견이나 결정으로 행동하는 것입니다."
        },
        {
            "term": "상대방",
            "explanation": "계약의 다른 쪽 당사자를 의미합니다."
        },
        {
            "term": "당사자",
            "explanation": "계약에 참여하는 사람들을 의미합니다."
        },
    ]
    return sample_dict


def build_vector_store():
    """법률 용어 사전 벡터 스토어 구축"""
    print("법률 용어 사전 벡터 스토어 구축을 시작합니다...")
    
    # 1. 용어 사전 로드
    print("1. 법률 용어 사전 로드 중...")
    dictionary = load_legal_dictionary()
    print(f"   {len(dictionary)}개의 용어를 로드했습니다.")
    
    # 2. 텍스트 변환 (용어 + 설명을 하나의 텍스트로)
    print("2. 텍스트 변환 중...")
    texts = []
    for entry in dictionary:
        # 검색 시 용어와 설명 모두 고려할 수 있도록 결합
        text = f"{entry['term']}: {entry['explanation']}"
        texts.append(text)
    
    # 3. 임베딩 모델 초기화
    print("3. 임베딩 모델 초기화 중...")
    print(f"   사용 모델: {settings.embedding_provider} - {settings.embedding_model}")
    from utils.embeddings import get_embeddings
    embeddings = get_embeddings()
    
    # 4. 벡터 스토어 생성
    print("4. 벡터 스토어 생성 중...")
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings
    )
    
    # 5. 원본 딕셔너리도 함께 저장 (메타데이터로)
    # FAISS는 메타데이터를 지원하므로, 각 텍스트에 원본 딕셔너리 정보를 연결
    # 여기서는 간단하게 별도 JSON 파일로 저장
    dict_metadata_path = Path(settings.legal_dictionary_store_path) / "dictionary_metadata.json"
    dict_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dict_metadata_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    
    # 6. 벡터 스토어 저장
    print("5. 벡터 스토어 저장 중...")
    store_path = Path(settings.legal_dictionary_store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    
    vector_store.save_local(str(store_path))
    print(f"   저장 완료: {store_path}")
    
    print("\n법률 용어 사전 벡터 스토어 구축이 완료되었습니다!")


if __name__ == "__main__":
    build_vector_store()



