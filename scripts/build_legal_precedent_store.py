"""법률 선례 벡터 스토어 구축 스크립트"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings
from utils.text_splitter import create_chunk_splitter
from utils.embeddings import get_embeddings


def load_structured_terms_act() -> tuple[list[str], list[dict]]:
    """
    구조화된 약관법 조항 로드
    
    Returns:
        tuple: (문서 텍스트 리스트, 메타데이터 리스트)
    """
    import json
    legal_docs_path = Path(settings.legal_docs_path)
    structured_file = legal_docs_path / "약관법_구조화.json"
    
    texts = []
    metadatas = []
    
    if structured_file.exists():
        print(f"로딩: {structured_file}")
        with open(structured_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        for article in articles:
            # 전체 조항 내용을 텍스트로 저장
            text = f"약관법 {article['article']} ({article['title']})\n\n{article['full_content']}"
            texts.append(text)
            
            # 메타데이터 저장
            metadata = {
                "article": article["article"],
                "title": article["title"],
                "category": article["category"],
                "keywords": ", ".join(article["keywords"]),
                "priority": article["priority"],
                "type": "terms_act",
                "source": "약관법_구조화.json"
            }
            metadatas.append(metadata)
            
            # 각 하위 호도 별도 문서로 저장 (더 세밀한 검색을 위해)
            for sub_article in article.get("sub_articles", []):
                sub_text = f"약관법 {article['article']} {sub_article['number']}\n{article['title']} - {sub_article['content']}"
                texts.append(sub_text)
                
                sub_metadata = {
                    "article": article["article"],
                    "sub_article": sub_article["number"],
                    "title": article["title"],
                    "category": article["category"],
                    "keywords": ", ".join(article["keywords"]),
                    "priority": article["priority"],
                    "type": "terms_act_sub",
                    "source": "약관법_구조화.json"
                }
                metadatas.append(sub_metadata)
    else:
        print(f"경고: {structured_file} 파일이 없습니다. 기존 TXT 파일을 사용합니다.")
    
    return texts, metadatas


def load_legal_documents() -> tuple[list[str], list[dict]]:
    """
    법률 문서들을 로드 (구조화된 약관법 + 기타 문서)
    
    Returns:
        tuple: (문서 텍스트 리스트, 메타데이터 리스트)
    """
    import csv
    import json
    legal_docs_path = Path(settings.legal_docs_path)
    all_texts = []
    all_metadatas = []
    
    if not legal_docs_path.exists():
        print(f"경고: {legal_docs_path} 디렉토리가 존재하지 않습니다.")
        print("샘플 데이터를 생성합니다...")
        sample_docs = create_sample_legal_documents()
        all_texts.extend(sample_docs)
        all_metadatas.extend([{"type": "sample", "source": "sample"} for _ in sample_docs])
    else:
        # 1. 구조화된 약관법 조항 로드 (우선)
        terms_act_texts, terms_act_metadatas = load_structured_terms_act()
        all_texts.extend(terms_act_texts)
        all_metadatas.extend(terms_act_metadatas)
        
        # 2. 기타 TXT 파일 로드 (약관법_구조화.json 제외)
        for file_path in legal_docs_path.glob("*.txt"):
            if "약관법_구조화" in file_path.name or "약관법 핵심조항" in file_path.name:
                continue  # 구조화된 파일은 이미 로드됨
            
            print(f"로딩: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                all_texts.append(content)
                all_metadatas.append({
                    "type": "text_file",
                    "source": file_path.name
                })
        
        # 3. CSV 파일 로드
        for file_path in legal_docs_path.glob("*.csv"):
            print(f"로딩: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        # CSV의 각 행에서 중요한 정보 추출
                        doc_parts = []
                        
                        # 불공정 약관 원문
                        if row.get("불공정 약관 원문"):
                            doc_parts.append(f"불공정 약관 원문: {row['불공정 약관 원문']}")
                        
                        # 시정 요청 사유
                        if row.get("시정 요청 사유"):
                            doc_parts.append(f"시정 요청 사유: {row['시정 요청 사유']}")
                        
                        # 근거 조항
                        if row.get("근거 조항(약관법)"):
                            doc_parts.append(f"근거 조항(약관법): {row['근거 조항(약관법)']}")
                        elif row.get("근거 조항"):  # 다른 CSV 파일 형식 지원
                            doc_parts.append(f"근거 조항: {row['근거 조항']}")
                        
                        if row.get("근거조항(약관법 외)"):
                            doc_parts.append(f"근거조항(약관법 외): {row['근거조항(약관법 외)']}")
                        
                        # 수정 후 약관 조항
                        if row.get("수정 후 약관 조항"):
                            doc_parts.append(f"수정 후 약관 조항: {row['수정 후 약관 조항']}")
                        
                        # 주제 정보도 추가
                        if row.get("대주제"):
                            doc_parts.append(f"대주제: {row['대주제']}")
                        if row.get("중주제"):
                            doc_parts.append(f"중주제: {row['중주제']}")
                        if row.get("소주제"):
                            doc_parts.append(f"소주제: {row['소주제']}")
                        
                        # 모든 정보를 하나의 문서로 결합
                        if doc_parts:
                            combined_doc = "\n".join(doc_parts)
                            all_texts.append(combined_doc)
                            
                            # 메타데이터 생성
                            metadata = {
                                "type": "case",
                                "source": file_path.name,
                                "case_id": row.get("ID", ""),
                                "category": row.get("대주제", ""),
                                "sub_category": row.get("중주제", ""),
                                "violated_article": row.get("근거 조항(약관법)", "") or row.get("근거 조항", "")
                            }
                            all_metadatas.append(metadata)
                            
            except Exception as e:
                print(f"  CSV 파일 읽기 오류 ({file_path}): {e}")
                continue
    
    return all_texts, all_metadatas

def create_sample_legal_documents() -> list[str]:
    """샘플 법률 문서 생성 (실제 사용 시 실제 문서로 교체 필요)"""
    sample_docs = [
        """약관의 규제에 관한 법률 제6조 (일반원칙)
① 약관의 내용이 신의성실의 원칙에违反하여 계약의 목적을 달성할 수 없게 하거나 
계약상의 권리·의무의 공정성을 해하는 조항은 무효로 한다.

② 약관의 내용 중 다음 각 호의 어느 하나에 해당하는 조항은 무효로 한다.
1. 고객에게 부당하게 불리한 조항
2. 고객의 계약 해지권을 부당하게 제한하는 조항
3. 고객에게 부당하게 불리한 손해배상액의 예정 또는 위약금에 관한 조항
4. 고객에게 부당하게 불리한 지연 손해금에 관한 조항""",
        
        """약관의 규제에 관한 법률 제7조 (일방적 해지권 제한 금지)
사업자는 고객에게 부당하게 불리한 조건으로 계약을 해지할 수 있는 권리를 약관에 
정할 수 없다. 다만, 다음 각 호의 어느 하나에 해당하는 경우에는 그러하지 아니하다.
1. 계속적 계약관계에서 고객이 계약상 의무를 이행하지 아니한 경우
2. 계약의 목적을 달성할 수 없는 경우""",
        
        """약관의 규제에 관한 법률 제8조 (손해배상액의 예정)
약관에 정한 손해배상액의 예정 또는 위약금에 관한 조항이 고객에게 부당하게 불리한 
경우에는 그 조항은 무효로 한다.""",
        
        """공정거래위원회 불공정 약관 시정 사례
사례 1: "회사는 어떠한 경우에도 손해배상 책임을 지지 않는다"는 조항
→ 이는 약관법 제6조, 제8조에 위배되는 무효 조항입니다. 
고객에게 부당하게 불리한 조항으로 판단됩니다.

사례 2: "고객은 일방적으로 계약을 해지할 수 없으며, 해지 시 위약금을 지급해야 한다"는 조항
→ 약관법 제7조에 위배될 수 있습니다. 고객의 계약 해지권을 부당하게 제한하는 조항입니다.

사례 3: "회사는 사전 통지 없이 약관을 변경할 수 있으며, 고객은 이에 동의한 것으로 간주한다"는 조항
→ 고객의 동의 없이 약관을 변경할 수 있다는 조항은 무효입니다.""",
        
        """대법원 판례 (2010다12345)
약관에 '회사는 고객에게 발생한 모든 손해에 대해 일체의 책임을 지지 않는다'는 조항이 있는 경우,
이는 약관법 제6조 제1항에 위배되어 무효이다. 다만, 회사의 고의 또는 중과실로 인한 손해에 
대한 책임을 면제하는 조항은 유효할 수 있다.""",
    ]
    return sample_docs


def build_vector_store():
    """법률 선례 벡터 스토어 구축 (메타데이터 포함)"""
    print("법률 선례 벡터 스토어 구축을 시작합니다...")
    
    # 1. 문서 로드 (메타데이터 포함)
    print("1. 법률 문서 로드 중...")
    documents, metadatas = load_legal_documents()
    print(f"   {len(documents)}개의 문서를 로드했습니다.")
    
    # 2. 텍스트 분할 (약관법 구조화 데이터는 분할하지 않음)
    print("2. 텍스트 처리 중...")
    text_splitter = create_chunk_splitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    all_chunks = []
    all_chunk_metadatas = []
    
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        
        # 약관법 구조화 데이터와 CSV 사례 데이터는 분할하지 않고 그대로 사용
        if metadata.get("type") in ["terms_act", "terms_act_sub", "case"]:
            all_chunks.append(doc)
            all_chunk_metadatas.append(metadata)
        else:
            # 기타 문서는 청킹
            chunks = text_splitter.split_text(doc)
            for chunk in chunks:
                all_chunks.append(chunk)
                # 청크에도 원본 메타데이터 포함
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunks.index(chunk)
                all_chunk_metadatas.append(chunk_metadata)
    
    print(f"   {len(all_chunks)}개의 문서/청크를 준비했습니다.")
    print(f"   - 약관법 조항: {sum(1 for m in all_chunk_metadatas if m.get('type') in ['terms_act', 'terms_act_sub'])}개")
    print(f"   - CSV 사례 데이터: {sum(1 for m in all_chunk_metadatas if m.get('type') == 'case')}개")
    print(f"   - 기타 문서: {len(all_chunks) - sum(1 for m in all_chunk_metadatas if m.get('type') in ['terms_act', 'terms_act_sub', 'case'])}개")
    
    # 3. 임베딩 모델 초기화
    print("3. 임베딩 모델 초기화 중...")
    print(f"   사용 모델: {settings.embedding_provider} - {settings.embedding_model}")
    embeddings = get_embeddings()
    
    # 4. 벡터 스토어 생성 (메타데이터 포함)
    print("4. 벡터 스토어 생성 중...")
    vector_store = FAISS.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_chunk_metadatas
    )
    
    # 5. 저장
    print("5. 벡터 스토어 저장 중...")
    store_path = Path(settings.legal_precedent_store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    
    vector_store.save_local(str(store_path))
    print(f"   저장 완료: {store_path}")
    
    print("\n법률 선례 벡터 스토어 구축이 완료되었습니다!")
    print(f"   총 {len(all_chunks)}개의 문서가 메타데이터와 함께 저장되었습니다.")


if __name__ == "__main__":
    build_vector_store()

