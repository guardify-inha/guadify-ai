"""불공정 약관 탐지 체인"""
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from config import settings
from utils.embeddings import get_embeddings
from utils.risk_keywords import contains_risk_keywords, get_matched_keywords
from utils.hybrid_search import keyword_search, hybrid_search, extract_keywords
import json
from pathlib import Path

# Reranking을 위한 CrossEncoder (지연 로딩)
_reranker = None

# Few-shot 예시 (지연 로딩)
_few_shot_examples = None


def load_few_shot_examples() -> List[Dict[str, Any]]:
    """Few-shot 예시 로드"""
    global _few_shot_examples
    if _few_shot_examples is not None:
        return _few_shot_examples
    
    examples_path = Path(settings.data_path) / "few_shot_examples.json"
    
    if examples_path.exists():
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                _few_shot_examples = json.load(f)
            return _few_shot_examples
        except Exception as e:
            print(f"Few-shot 예시 로드 오류: {e}")
            return []
    else:
        return []


def select_relevant_examples(clause: str, examples: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    입력 조항과 유사한 Few-shot 예시 선택
    
    Args:
        clause: 입력 조항
        examples: 예시 리스트
        top_k: 선택할 예시 개수
    
    Returns:
        선택된 예시 리스트
    """
    if not examples:
        return []
    
    # 간단한 키워드 매칭으로 유사도 계산
    clause_keywords = set(extract_keywords(clause))
    
    scored_examples = []
    for example in examples:
        example_keywords = set(extract_keywords(example.get("input_clause", "")))
        
        # Jaccard 유사도
        intersection = len(clause_keywords & example_keywords)
        union = len(clause_keywords | example_keywords)
        similarity = intersection / union if union > 0 else 0.0
        
        scored_examples.append((example, similarity))
    
    # 유사도 순으로 정렬
    scored_examples.sort(key=lambda x: x[1], reverse=True)
    
    return [ex for ex, score in scored_examples[:top_k] if score > 0]


def get_reranker():
    """CrossEncoder reranker 싱글톤"""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            # 한국어 법률 문서에 최적화된 reranking 모델 사용
            _reranker = CrossEncoder(settings.rerank_model)
            print(f"Reranking 모델 로드 완료: {settings.rerank_model}")
        except ImportError:
            print("경고: sentence-transformers가 설치되지 않았습니다. reranking을 건너뜁니다.")
            return None
        except Exception as e:
            print(f"경고: Reranking 모델 로드 실패 ({settings.rerank_model}): {e}")
            print("기본 영어 모델로 대체합니다.")
            try:
                _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e2:
                print(f"기본 모델 로드도 실패: {e2}")
                return None
    return _reranker


def rerank_documents(query: str, documents: List[Document], top_k: int = 5, threshold: float = 0.0) -> tuple[List[Document], List[float]]:
    """
    검색된 문서들을 관련성 점수로 재정렬
    
    Args:
        query: 검색 쿼리
        documents: 검색된 문서 리스트
        top_k: 반환할 상위 문서 개수
        threshold: 최소 관련성 점수 (이하 문서 제외)
    
    Returns:
        (재정렬된 문서 리스트, 점수 리스트) 튜플
    """
    reranker = get_reranker()
    
    if reranker is None or len(documents) == 0:
        return documents[:top_k], [0.0] * len(documents[:top_k])
    
    try:
        # 쿼리-문서 쌍 생성
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 관련성 점수 계산
        scores = reranker.predict(pairs)
        
        # 점수와 문서를 함께 정렬
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # threshold 이상인 문서만 필터링
        filtered_docs = [(doc, score) for doc, score in scored_docs if score >= threshold]
        
        docs = [doc for doc, score in filtered_docs[:top_k]]
        scores_list = [score for doc, score in filtered_docs[:top_k]]
        
        return docs, scores_list
    except Exception as e:
        print(f"Reranking 오류: {e}")
        return documents[:top_k], [0.0] * len(documents[:top_k])


def load_legal_precedent_store():
    """법률 선례 벡터 스토어 로드"""
    embeddings = get_embeddings()
    
    store_path = settings.legal_precedent_store_path
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


def create_unfair_term_detector_chain():
    """
    불공정 약관 탐지 체인 생성
    
    입력: {"clause": "계약서 조항 텍스트"}
    출력: {"clause": "...", "is_unfair": true/false, "reason": "...", "evidence_law": "..."}
    """
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key
    )
    
    # 벡터 스토어 로드
    vector_store = load_legal_precedent_store()
    
    # RAG 검색 체인
    def retrieve_legal_precedents(clause: str) -> tuple[str, str]:
        """법률 선례 검색 - 약관법 조항 우선 + Reranking
        
        Returns:
            tuple: (전체 컨텍스트, 가장 관련된 약관법 조항의 전체 내용)
        """
        if vector_store is None:
            return "법률 선례 벡터 스토어를 사용할 수 없습니다.", ""
        
        try:
            # 1단계: 벡터 유사도 검색 (더 많이 검색)
            initial_k = getattr(settings, 'initial_search_k', settings.top_k_retrieval * 3)
            vector_docs = vector_store.similarity_search(
                clause,
                k=initial_k  # reranking을 위해 더 많이 검색
            )
            
            # 2단계: 키워드 검색 (하이브리드 검색)
            # 모든 문서를 가져와서 키워드 검색 수행
            all_docs_in_store = []
            # FAISS에서 모든 문서를 가져올 수 없으므로, 벡터 검색 결과를 사용
            keyword_results = keyword_search(clause, vector_docs, top_k=settings.top_k_retrieval * 2)
            
            # 3단계: 하이브리드 검색으로 결과 결합
            combined_docs = hybrid_search(
                clause,
                vector_docs,
                keyword_results,
                vector_weight=0.7,
                keyword_weight=0.3
            )
            
            # 4단계: Reranking으로 관련성 재평가
            rerank_scores = []
            if settings.rerank_enabled:
                reranked_docs, rerank_scores = rerank_documents(
                    clause,
                    combined_docs,
                    top_k=settings.top_k_retrieval * 2,
                    threshold=settings.rerank_threshold
                )
            else:
                reranked_docs = combined_docs[:settings.top_k_retrieval * 2]
                rerank_scores = [0.0] * len(reranked_docs)
            
            # 3단계: 약관법 조항과 기타 법률 선례를 분리 (메타데이터 활용)
            terms_act_docs = []
            other_docs = []
            
            for doc in reranked_docs:
                # 메타데이터에서 약관법 조항인지 확인
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                content = doc.page_content
                
                if metadata.get("type") in ["terms_act", "terms_act_sub"]:
                    terms_act_docs.append(doc)
                elif "약관법" in content or any(f"제{i}조" in content for i in range(6, 15)):
                    # 메타데이터가 없어도 내용으로 판단
                    terms_act_docs.append(doc)
                else:
                    other_docs.append(doc)
            
            # 약관법 조항을 우선적으로 reranking
            terms_act_scores = []
            other_docs_scores = []
            if settings.rerank_enabled:
                if terms_act_docs:
                    terms_act_docs, terms_act_scores = rerank_documents(clause, terms_act_docs, top_k=5, threshold=settings.rerank_threshold)
                
                if other_docs:
                    other_docs, other_docs_scores = rerank_documents(clause, other_docs, top_k=3, threshold=settings.rerank_threshold)
            
            # 가장 관련된 약관법 조항의 전체 내용 추출
            primary_terms_act_content = ""
            if terms_act_docs:
                primary_terms_act_content = terms_act_docs[0].page_content
            
            # 약관법 조항을 먼저, 그 다음 기타 법률 선례를 배치
            context_parts = []
            
            if terms_act_docs:
                terms_act_texts = [doc.page_content for doc in terms_act_docs[:5]]
                context_parts.append("=== 약관법 핵심조항 (우선 참조) ===\n" + "\n\n".join(terms_act_texts))
            
            if other_docs:
                other_texts = [doc.page_content for doc in other_docs[:3]]
                context_parts.append("=== 기타 법률 선례 및 판례 ===\n" + "\n\n".join(other_texts))
            
            context = "\n\n".join(context_parts) if context_parts else "\n\n".join([doc.page_content for doc in reranked_docs[:5]])
            
            # 검색 품질 메트릭 로깅 (디버깅용)
            if settings.rerank_enabled and rerank_scores:
                avg_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0
                max_score = max(rerank_scores) if rerank_scores else 0.0
                min_score = min(rerank_scores) if rerank_scores else 0.0
                
                # 디버깅 모드에서만 출력
                import os
                if os.getenv("DEBUG", "false").lower() == "true":
                    print(f"[검색 품질] 평균 점수: {avg_score:.3f}, 최고: {max_score:.3f}, 최저: {min_score:.3f}")
                    print(f"[검색 결과] 약관법 조항: {len(terms_act_docs)}개, 기타: {len(other_docs)}개")
            
            return context, primary_terms_act_content
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return f"검색 중 오류 발생: {e}", ""
    
    # 위험 키워드 체크
    def check_risk_keywords(clause: str) -> Dict[str, Any]:
        """위험 키워드 확인"""
        has_risk = contains_risk_keywords(clause)
        matched_keywords = get_matched_keywords(clause) if has_risk else []
        return {
            "has_risk_keywords": has_risk,
            "matched_keywords": matched_keywords
        }
    
    # 메인 체인 구성
    def process_clause(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """조항 처리"""
        clause = input_dict.get("clause", "")
        
        # 병렬로 검색 및 키워드 체크
        legal_context, primary_terms_act_content = retrieve_legal_precedents(clause)
        risk_info = check_risk_keywords(clause)
        
        # Few-shot 예시 선택 및 포맷팅
        examples = load_few_shot_examples()
        relevant_examples = select_relevant_examples(clause, examples, top_k=3)
        
        # 예시를 문자열로 포맷팅
        formatted_examples = ""
        if relevant_examples:
            example_texts = []
            for i, ex in enumerate(relevant_examples, 1):
                example_text = f"""예시 {i}:
입력 조항: {ex.get('input_clause', '')}
불공정 여부: {ex.get('is_unfair', False)}
이유: {ex.get('reason', '')}
근거 법률: {ex.get('evidence_law', '')}"""
                example_texts.append(example_text)
            formatted_examples = "\n\n".join(example_texts)
        else:
            formatted_examples = "관련 예시 없음"
        
        return {
            "clause": clause,
            "legal_context": legal_context,
            "primary_terms_act_content": primary_terms_act_content,
            "risk_info": risk_info,
            "formatted_examples": formatted_examples
        }
    
    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 계약서의 불공정 약관을 분석하는 법률 전문가입니다.
주어진 계약서 조항과 관련 법률 선례, 위험 키워드 정보를 바탕으로 
해당 조항이 불공정한지 판단하고, 그 이유와 근거 법률을 제시하세요.

**중요한 근거 법률 선택 원칙:**
1. 약관법 핵심조항(제6조~제14조)을 가장 우선적으로 참조해야 합니다.
2. 약관법 제6조는 매우 일반적인 조항이므로, 더 구체적인 조항이 있으면 그것을 우선합니다.
   - 예: 손해배상 관련 → 약관법 제7조 또는 제8조
   - 예: 계약 해제/해지 관련 → 약관법 제9조
   - 예: 채무 이행 관련 → 약관법 제10조
   - 예: 고객 권익 관련 → 약관법 제11조
   - 예: 의사표시 관련 → 약관법 제12조
   - 예: 소송 제기 관련 → 약관법 제14조
3. 약관법 제6조는 다른 구체적인 조항이 없을 때만 사용합니다.
4. 약관법 조항이 없을 경우에만 다른 법률(민법, 상법 등)을 참조합니다.

**근거 법률 표기 규칙:**
- 약관법 조항은 '조'까지만 표기합니다 (예: "약관법 제7조")
- 세부 호(제1호, 제2호 등)는 표기하지 않습니다.

응답은 반드시 다음 JSON 형식으로 제공해야 합니다:
{{
    "is_unfair": true 또는 false,
    "reason": "불공정한 이유 또는 공정한 이유를 상세히 설명 (약관법 조항을 구체적으로 언급)",
    "evidence_law": "관련 법률 조항 (약관법 제X조 형식으로만 표기, 예: '약관법 제7조') 또는 '해당 없음'"
}}

위험 키워드가 발견되었거나 법률 선례에서 유사한 불공정 사례가 있다면 
is_unfair를 true로 설정하세요."""),
        ("human", """계약서 조항:
{clause}

관련 법률 선례 및 조항:
{legal_context}

가장 관련된 약관법 조항 전체 내용:
{primary_terms_act_content}

위험 키워드 정보:
{risk_info}

Few-shot 예시:
{formatted_examples}

위 정보를 바탕으로 JSON 형식으로 분석 결과를 제공하세요.
특히 약관법 핵심조항을 우선적으로 참조하고, 구체적인 조항이 있으면 제6조 대신 그것을 사용하세요.
근거 법률은 '조'까지만 표기하세요 (예: "약관법 제7조").
Few-shot 예시를 참고하여 유사한 패턴을 인식하세요.""")
    ])
    
    # 체인 구성
    chain = (
        RunnablePassthrough()
        | process_clause
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # JSON 파싱 래퍼 (클로저로 원본 입력 접근)
    def create_parse_output(original_input: Dict[str, Any]):
        """LLM 출력을 JSON으로 파싱하는 함수 생성"""
        def parse_output(output: str) -> Dict[str, Any]:
            """LLM 출력을 JSON으로 파싱"""
            try:
                # JSON 코드 블록 제거
                output = output.strip()
                if output.startswith("```json"):
                    output = output[7:]
                if output.startswith("```"):
                    output = output[3:]
                if output.endswith("```"):
                    output = output[:-3]
                output = output.strip()
                
                result = json.loads(output)
                
                # 원본 조항 추가
                result["clause"] = original_input.get("clause", "")
                
                # 해당 조의 전체 내용 추가
                primary_content = original_input.get("primary_terms_act_content", "")
                if primary_content and result.get("evidence_law", "") != "해당 없음":
                    result["evidence_law_content"] = primary_content
                else:
                    result["evidence_law_content"] = ""
                
                return result
            except json.JSONDecodeError as e:
                # 파싱 실패 시 기본값 반환
                return {
                    "clause": original_input.get("clause", ""),
                    "is_unfair": False,
                    "reason": f"JSON 파싱 오류: {e}",
                    "evidence_law": "해당 없음",
                    "evidence_law_content": ""
                }
        return parse_output
    
    # 최종 체인 구성 (동적 파싱 함수 생성)
    def full_chain_wrapper(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """전체 체인 래퍼"""
        parse_func = create_parse_output(input_dict)
        result = chain.invoke(input_dict)
        return parse_func(result)
    
    full_chain = full_chain_wrapper
    
    return full_chain


# 전역 체인 인스턴스 (재사용을 위해)
_unfair_term_detector_chain = None


def get_unfair_term_detector_chain():
    """불공정 약관 탐지 체인 싱글톤"""
    global _unfair_term_detector_chain
    if _unfair_term_detector_chain is None:
        _unfair_term_detector_chain = create_unfair_term_detector_chain()
    return _unfair_term_detector_chain

