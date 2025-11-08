"""전체 분석 파이프라인"""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from chains.unfair_term_detector import get_unfair_term_detector_chain
from chains.legal_term_translator import get_legal_term_translator_chain
from utils.text_splitter import split_contract_by_clauses
from config import settings
import json


def create_full_analysis_chain():
    """
    전체 분석 파이프라인 생성
    
    입력: {"contract_text": "계약서 전체 텍스트"}
    출력: {
        "overall_risk_assessment": "높음/중간/낮음",
        "summary": "종합 요약",
        "clauses": [
            {
                "original_clause": "...",
                "analysis": {
                    "is_unfair": true/false,
                    "reason": "...",
                    "evidence_law": "..."
                },
                "easy_translation": "...",
                "suggestion": "..."
            },
            ...
        ]
    }
    """
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key
    )
    
    # 서브 체인 로드
    unfair_detector = get_unfair_term_detector_chain()
    term_translator = get_legal_term_translator_chain()
    
    # 1단계: 계약서를 조항 단위로 분리
    def split_contract(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """계약서를 조항 단위로 분리"""
        contract_text = input_dict.get("contract_text", "")
        clauses = split_contract_by_clauses(contract_text)
        
        return {
            "contract_text": contract_text,
            "clauses": clauses,
            "num_clauses": len(clauses)
        }
    
    # 2단계: 각 조항에 대해 병렬로 분석 및 번역 수행
    async def analyze_clauses(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """모든 조항을 병렬로 분석 (비동기)"""
        clauses = input_dict.get("clauses", [])
        
        # 각 조항에 대해 병렬 처리
        clause_analyses = []
        
        for clause in clauses:
            # 병렬로 불공정 탐지와 용어 번역 수행
            parallel_chain = RunnableParallel({
                "analysis": unfair_detector,
                "translation": term_translator
            })
            
            # 비동기 호출
            parallel_results = await parallel_chain.ainvoke({"clause": clause})
            
            clause_analyses.append({
                "original_clause": clause,
                "analysis": parallel_results["analysis"],
                "easy_translation": parallel_results["translation"]
            })
        
        return {
            "contract_text": input_dict.get("contract_text", ""),
            "clause_analyses": clause_analyses
        }
    
    # 3단계: 종합 평가
    def synthesize_final_assessment(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """최종 종합 평가"""
        clause_analyses = input_dict.get("clause_analyses", [])
        
        # 불공정 조항 개수 계산
        unfair_count = sum(
            1 for ca in clause_analyses 
            if ca.get("analysis", {}).get("is_unfair", False)
        )
        
        # 위험도 평가 프롬프트
        assessment_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 계약서 분석 전문가입니다.
주어진 모든 조항의 분석 결과를 종합하여 전체 계약서의 위험도를 평가하고,
핵심 위험 조항을 요약하며, 가장 치명적인 위험 조항에 대한 대안을 제시하세요.

응답은 반드시 다음 JSON 형식으로 제공해야 합니다:
{{
    "overall_risk_assessment": "높음" 또는 "중간" 또는 "낮음",
    "summary": "종합 요약 (불공정 조항 개수, 주요 위험 사항 등)",
    "critical_suggestions": [
        {{
            "clause_index": 조항 번호 (0부터 시작),
            "suggestion": "구체적인 대안 조항 제안"
        }},
        ...
    ]
}}"""),
            ("human", """계약서 분석 결과:

불공정 조항 개수: {unfair_count}개
전체 조항 개수: {total_count}개

각 조항 분석 결과:
{clause_analyses}

위 정보를 바탕으로 JSON 형식으로 종합 평가를 제공하세요.""")
        ])
        
        # 조항 분석 결과를 문자열로 변환
        analyses_text = "\n\n".join([
            f"조항 {i+1}:\n"
            f"원본: {ca['original_clause'][:100]}...\n"
            f"불공정 여부: {ca['analysis'].get('is_unfair', False)}\n"
            f"이유: {ca['analysis'].get('reason', '')}\n"
            f"근거 법률: {ca['analysis'].get('evidence_law', '')}"
            for i, ca in enumerate(clause_analyses)
        ])
        
        # LLM으로 종합 평가 생성
        assessment_result = llm.invoke(
            assessment_prompt.format_messages(
                unfair_count=unfair_count,
                total_count=len(clause_analyses),
                clause_analyses=analyses_text
            )
        )
        
        # JSON 파싱
        try:
            assessment_text = assessment_result.content.strip()
            if assessment_text.startswith("```json"):
                assessment_text = assessment_text[7:]
            if assessment_text.startswith("```"):
                assessment_text = assessment_text[3:]
            if assessment_text.endswith("```"):
                assessment_text = assessment_text[:-3]
            assessment_text = assessment_text.strip()
            
            assessment_json = json.loads(assessment_text)
        except Exception as e:
            # 파싱 실패 시 기본값
            assessment_json = {
                "overall_risk_assessment": "중간" if unfair_count > 0 else "낮음",
                "summary": f"총 {unfair_count}개의 불공정 소지가 있는 조항이 발견되었습니다.",
                "critical_suggestions": []
            }
        
        # 최종 결과 구성
        final_result = {
            "overall_risk_assessment": assessment_json.get("overall_risk_assessment", "중간"),
            "summary": assessment_json.get("summary", ""),
            "clauses": []
        }
        
        # 각 조항에 대안 제안 추가
        critical_suggestions = assessment_json.get("critical_suggestions", [])
        suggestion_dict = {
            item.get("clause_index", -1): item.get("suggestion", "")
            for item in critical_suggestions
        }
        
        for i, ca in enumerate(clause_analyses):
            clause_result = {
                "original_clause": ca["original_clause"],
                "analysis": ca["analysis"],
                "easy_translation": ca["easy_translation"],
                "suggestion": suggestion_dict.get(i, "")
            }
            final_result["clauses"].append(clause_result)
        
        return final_result
    
    # 전체 체인 구성 (비동기 처리)
    async def full_chain_async(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """전체 분석 체인 (비동기)"""
        # 1단계: 계약서 분리
        split_result = split_contract(input_dict)
        
        # 2단계: 조항 분석
        analysis_result = await analyze_clauses(split_result)
        
        # 3단계: 종합 평가
        final_result = synthesize_final_assessment(analysis_result)
        
        return final_result
    
    return full_chain_async


# 전역 체인 인스턴스
_full_analysis_chain = None


def get_full_analysis_chain():
    """전체 분석 체인 싱글톤"""
    global _full_analysis_chain
    if _full_analysis_chain is None:
        _full_analysis_chain = create_full_analysis_chain()
    return _full_analysis_chain


# 비동기 버전
async def analyze_contract_async(contract_text: str) -> Dict[str, Any]:
    """계약서 분석 (비동기)"""
    chain = get_full_analysis_chain()
    result = await chain({"contract_text": contract_text})
    return result

