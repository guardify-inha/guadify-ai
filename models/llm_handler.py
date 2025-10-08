"""
LLM 핸들러 모듈

AI 모델을 활용한 불공정 약관 분석 및 판단을 위한 핵심 모듈
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalContext:
    """법적 맥락 정보"""
    content: str
    source: str
    document_type: str  # law, standard, reference
    confidence: float
    relevance_score: float

class UnfairClauseAnalysis(BaseModel):
    """불공정 조항 분석 결과"""
    is_unfair: bool = Field(description="불공정 여부")
    reason: str = Field(description="법적 근거를 포함한 불공정 판단 이유", min_length=10)
    risk_level: str = Field(description="위험도 레벨", pattern="^(High|Medium|Low)$")
    suggestion: str = Field(description="대안으로 제시하는 공정한 문구", min_length=5)
    confidence: float = Field(description="분석 신뢰도", ge=0.0, le=1.0)
    legal_basis: List[str] = Field(description="관련 법령 목록", default_factory=list)
    
    class Config:
        """Pydantic 설정"""
        validate_assignment = True
        use_enum_values = True

class LLMHandler:
    """LLM을 활용한 불공정 약관 분석 핸들러"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        LLM 핸들러 초기화
        
        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 로드)
            model: 사용할 LLM 모델명
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정하거나 api_key 파라미터를 제공하세요.")
        
        self.model = model
        # OpenAI 클라이언트 초기화 (최신 버전 호환)
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            # 대안: 환경 변수 설정 방식
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.client = openai.OpenAI()
        
        # 프롬프트 템플릿 설정
        self.system_prompt = self._build_system_prompt()
        self.user_prompt_template = self._build_user_prompt_template()
        
        logger.info(f"LLM 핸들러 초기화 완료 - 모델: {self.model}")
    
    def _build_system_prompt(self) -> str:
        """시스템 프롬프트 구성"""
        return """당신은 한국의 법률 전문가입니다. 계약 조항의 불공정성을 정확히 판단하는 것이 당신의 역할입니다.

중요 지침:
- 공정한 조항을 불공정으로 오탐하지 마세요
- 문맥을 고려하여 종합적으로 판단하세요  
- 소비자 보호 조항은 일반적으로 공정합니다

불공정성 판단 기준:

1. 일방적 불이익 조항
- 한 당사자에게만 불리한 조건
- 예: "회사는 언제든지 해지할 수 있으나, 이용자는 사전 통지 없이 해지할 수 없다"

2. 소비자 권리 제한
- 소비자의 기본 권리를 제한하는 조항
- 예: "이용자는 어떠한 경우에도 이의를 제기할 수 없다"

3. 정보 제공 의무 회피
- 중요한 정보를 제공하지 않는 조항
- 예: "회사는 정보를 제공할 의무가 없다"

4. 손해배상 제한
- 과도한 손해배상 제한
- 예: "회사는 어떠한 손해에 대해서도 배상하지 않는다"

5. 계약 해지 제한
- 불공정한 해지 조건
- 예: "이용자는 회사의 허락 없이는 해지할 수 없다"

공정한 조항의 예시:
- "이용자는 서비스 이용 중 발생한 피해에 대해 구제를 신청할 수 있습니다"
- "회사는 신청을 받은 날로부터 30일 이내에 처리 결과를 통지합니다"
- "양 당사자는 상호 합의하여 계약을 변경할 수 있습니다"

출력 형식:
반드시 다음 JSON 형식으로 답변하세요:
{
  "is_unfair": boolean,
  "reason": "법적 근거를 포함한 불공정 판단 이유",
  "risk_level": "High | Medium | Low",
  "suggestion": "대안으로 제시하는 공정한 문구",
  "confidence": 0.0-1.0,
  "legal_basis": ["관련 법령 목록"]
}"""
    
    def _build_user_prompt_template(self) -> str:
        """사용자 프롬프트 템플릿 구성"""
        return """다음 계약 조항을 분석해주세요:

## 분석할 조항
{clause_text}

## 관련 법적 맥락
{legal_context}

위 조항이 불공정한지 판단하고, 법적 근거와 함께 분석 결과를 JSON 형식으로 제공해주세요."""
    
    def analyze_clause(self, clause_text: str, legal_context: List[LegalContext]) -> UnfairClauseAnalysis:
        """
        조항의 불공정성 분석
        
        Args:
            clause_text: 분석할 조항 텍스트
            legal_context: 관련 법적 맥락 정보
            
        Returns:
            UnfairClauseAnalysis: 분석 결과
        """
        try:
            # 법적 맥락 포맷팅
            context_text = self._format_legal_context(legal_context)
            
            # 프롬프트 구성 (인코딩 안전 처리)
            user_prompt = self.user_prompt_template.format(
                clause_text=clause_text,
                legal_context=context_text
            )
            
            # 프롬프트 구성
            
            # LLM 호출 (최신 OpenAI API)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 낮은 온도로 일관성 확보
                max_tokens=1000
            )
            
            # 응답 파싱
            response_text = response.choices[0].message.content.strip()
            analysis_result = self._parse_response(response_text)
            
            logger.info(f"조항 분석 완료 - 불공정: {analysis_result.is_unfair}, 위험도: {analysis_result.risk_level}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"조항 분석 중 오류 발생: {e}")
            return self._get_default_analysis()
    
    def _format_legal_context(self, legal_context: List[LegalContext]) -> str:
        """법적 맥락을 텍스트로 포맷팅"""
        if not legal_context:
            return "관련 법적 맥락이 없습니다."
        
        context_parts = []
        for ctx in legal_context:
            doc_type_desc = {
                "law": "법령",
                "standard": "표준약관/지침", 
                "reference": "보도자료/사례"
            }.get(ctx.document_type, "기타")
            
            context_parts.append(
                f"- {doc_type_desc} ({ctx.confidence:.2f}): {ctx.content[:200]}...\n"
                f"  출처: {ctx.source}"
            )
        
        return "\n".join(context_parts)
    
    def _parse_response(self, response_text: str) -> UnfairClauseAnalysis:
        """LLM 응답을 파싱하여 분석 결과 객체로 변환"""
        try:
            # JSON 블록 추출
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            # JSON 파싱
            data = json.loads(json_text)
            
            # Pydantic 모델로 변환
            return UnfairClauseAnalysis(
                is_unfair=data.get("is_unfair", False),
                reason=data.get("reason", "분석 결과 없음"),
                risk_level=data.get("risk_level", "Low"),
                suggestion=data.get("suggestion", "개선 제안 없음"),
                confidence=data.get("confidence", 0.5),
                legal_basis=data.get("legal_basis", [])
            )
            
        except Exception as e:
            logger.error(f"응답 파싱 오류: {e}")
            logger.error(f"원본 응답: {response_text}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> UnfairClauseAnalysis:
        """기본 분석 결과 반환"""
        return UnfairClauseAnalysis(
            is_unfair=False,
            reason="AI 분석 실패로 인해 수동 검토가 필요합니다.",
            risk_level="Low",
            suggestion="수동 검토를 통해 불공정 여부를 판단하세요.",
            confidence=0.0,
            legal_basis=[]
        )
    
    def batch_analyze(self, clauses: List[str], legal_contexts: List[List[LegalContext]]) -> List[UnfairClauseAnalysis]:
        """
        여러 조항을 일괄 분석
        
        Args:
            clauses: 분석할 조항 리스트
            legal_contexts: 각 조항별 법적 맥락 리스트
            
        Returns:
            List[UnfairClauseAnalysis]: 분석 결과 리스트
        """
        results = []
        
        for i, (clause, context) in enumerate(zip(clauses, legal_contexts)):
            logger.info(f"조항 {i+1}/{len(clauses)} 분석 중...")
            result = self.analyze_clause(clause, context)
            results.append(result)
        
        return results

def create_llm_handler() -> LLMHandler:
    """LLM 핸들러 인스턴스 생성"""
    return LLMHandler()

# 테스트용 함수
def test_llm_handler():
    """LLM 핸들러 기본 기능 테스트"""
    try:
        handler = create_llm_handler()
        
        # 테스트 조항
        test_clause = "제10조 (소비자 보호) 이용자는 서비스 이용 중 발생한 피해에 대해 회사에 구제를 신청할 수 있습니다. 회사는 신청을 받은 날로부터 30일 이내에 처리 결과를 통지합니다."
        
        # 테스트 법적 맥락
        test_context = [
            LegalContext(
                content="소비자는 서비스 이용 중 발생한 피해에 대해 구제를 요청할 권리가 있습니다.",
                source="금융소비자 보호에 관한 법률",
                document_type="law",
                confidence=0.9,
                relevance_score=0.8
            )
        ]
        
        # 분석 실행
        result = handler.analyze_clause(test_clause, test_context)
        
        print("=== LLM 핸들러 테스트 결과 ===")
        print(f"조항: {test_clause}")
        print(f"불공정 여부: {result.is_unfair}")
        print(f"위험도: {result.risk_level}")
        print(f"신뢰도: {result.confidence}")
        print(f"판단 근거: {result.reason}")
        print(f"개선 제안: {result.suggestion}")
        print(f"법적 근거: {result.legal_basis}")
        
        return True
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_llm_handler()
