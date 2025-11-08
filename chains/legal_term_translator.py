"""법률 용어 번역 체인"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import settings
from utils.embeddings import get_embeddings
import json
from pathlib import Path


def load_legal_dictionary_store():
    """법률 용어 사전 벡터 스토어 로드"""
    embeddings = get_embeddings()
    
    store_path = settings.legal_dictionary_store_path
    try:
        vector_store = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"벡터 스토어 로드 실패: {e}")
        return None


def load_dictionary_metadata() -> Dict[str, str]:
    """법률 용어 사전 메타데이터 로드"""
    metadata_path = Path(settings.legal_dictionary_store_path) / "dictionary_metadata.json"
    
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
        
        # {용어: 설명} 형태의 딕셔너리로 변환
        return {entry["term"]: entry["explanation"] for entry in dictionary}
    except Exception as e:
        print(f"메타데이터 로드 실패: {e}")
        return {}


def create_legal_term_translator_chain():
    """
    법률 용어 번역 체인 생성
    
    입력: {"clause": "계약서 조항 텍스트"}
    출력: "쉽게 풀이한 조항 텍스트"
    """
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.3,  # 창의적 번역을 위해 약간 높게 설정
        openai_api_key=settings.openai_api_key
    )
    
    # 벡터 스토어 및 메타데이터 로드
    vector_store = load_legal_dictionary_store()
    dictionary_metadata = load_dictionary_metadata()
    
    # 법률 용어 검색 함수
    def retrieve_legal_terms(clause: str) -> str:
        """법률 용어 및 설명 검색"""
        if vector_store is None:
            # 벡터 스토어가 없으면 메타데이터만 사용
            return format_dictionary_metadata(dictionary_metadata)
        
        try:
            # 유사한 법률 용어 검색
            docs = vector_store.similarity_search(
                clause,
                k=settings.top_k_retrieval
            )
            
            # 검색된 용어들을 컨텍스트로 결합
            found_terms = []
            for doc in docs:
                # "용어: 설명" 형식에서 용어 추출
                text = doc.page_content
                if ":" in text:
                    term = text.split(":")[0].strip()
                    if term not in found_terms:
                        found_terms.append(text)
            
            # 메타데이터에서 추가 용어 찾기
            context = "\n".join(found_terms)
            
            # 메타데이터도 추가
            metadata_context = format_dictionary_metadata(dictionary_metadata)
            if metadata_context:
                context = f"{context}\n\n{metadata_context}" if context else metadata_context
            
            return context
        except Exception as e:
            return format_dictionary_metadata(dictionary_metadata)
    
    def format_dictionary_metadata(metadata: Dict[str, str]) -> str:
        """메타데이터를 문자열로 포맷팅"""
        if not metadata:
            return ""
        
        formatted = []
        for term, explanation in metadata.items():
            formatted.append(f"{term}: {explanation}")
        return "\n".join(formatted)
    
    # 메인 처리 함수
    def process_clause(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """조항 처리"""
        clause = input_dict.get("clause", "")
        legal_terms_context = retrieve_legal_terms(clause)
        
        return {
            "clause": clause,
            "legal_terms_context": legal_terms_context
        }
    
    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 문서를 일반인이 이해하기 쉬운 말로 번역하는 전문가입니다.
주어진 계약서 조항에 포함된 어려운 법률 용어를 참고 자료를 바탕으로 
쉽게 풀이하여 전체 조항을 재작성하세요.

요구사항:
1. 법률 용어의 정확한 의미를 유지하면서 일반인이 이해할 수 있는 쉬운 말로 번역
2. 원본 조항의 문맥과 의미를 정확히 보존
3. 전문 용어는 쉬운 말로 대체하되, 의미는 정확히 전달
4. 문장 구조를 자연스럽게 유지
5. 번역된 텍스트만 반환 (추가 설명 없이)"""),
        ("human", """계약서 조항:
{clause}

법률 용어 사전:
{legal_terms_context}

위 조항을 쉬운 말로 번역해 주세요.""")
    ])
    
    # 체인 구성
    chain = (
        RunnablePassthrough()
        | process_clause
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# 전역 체인 인스턴스 (재사용을 위해)
_legal_term_translator_chain = None


def get_legal_term_translator_chain():
    """법률 용어 번역 체인 싱글톤"""
    global _legal_term_translator_chain
    if _legal_term_translator_chain is None:
        _legal_term_translator_chain = create_legal_term_translator_chain()
    return _legal_term_translator_chain



