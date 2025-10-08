"""
LLM 핸들러 단위 테스트

AI 모델 기반 불공정 약관 탐지 시스템의 핵심 컴포넌트 테스트
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm_handler import LLMHandler, LegalContext, UnfairClauseAnalysis

class TestLLMHandler(unittest.TestCase):
    """LLM 핸들러 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.test_api_key = "test-api-key"
        self.test_model = "gpt-4o-mini"
        
        # 모킹된 OpenAI 응답
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message.content = json.dumps({
            "is_unfair": False,
            "reason": "이 조항은 소비자 보호를 위한 공정한 조항입니다.",
            "risk_level": "Low",
            "suggestion": "현재 조항이 적절합니다.",
            "confidence": 0.9,
            "legal_basis": ["금융소비자 보호에 관한 법률"]
        })
    
    @patch('models.llm_handler.openai.OpenAI')
    def test_llm_handler_initialization(self, mock_openai):
        """LLM 핸들러 초기화 테스트"""
        # Given
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # When
        handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
        
        # Then
        self.assertEqual(handler.api_key, self.test_api_key)
        self.assertEqual(handler.model, self.test_model)
        self.assertEqual(handler.client, mock_client)
        mock_openai.assert_called_once_with(api_key=self.test_api_key)
    
    def test_llm_handler_initialization_without_api_key(self):
        """API 키 없이 초기화 시 오류 테스트"""
        # Given
        with patch.dict(os.environ, {}, clear=True):
            # When & Then
            with self.assertRaises(ValueError) as context:
                LLMHandler()
            
            self.assertIn("OpenAI API 키가 설정되지 않았습니다", str(context.exception))
    
    @patch('models.llm_handler.openai.OpenAI')
    def test_analyze_clause_success(self, mock_openai):
        """조항 분석 성공 테스트"""
        # Given
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.mock_response
        mock_openai.return_value = mock_client
        
        handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
        
        test_clause = "제10조 (소비자 보호) 이용자는 서비스 이용 중 발생한 피해에 대해 회사에 구제를 신청할 수 있습니다."
        test_context = [
            LegalContext(
                content="소비자는 서비스 이용 중 발생한 피해에 대해 구제를 요청할 권리가 있습니다.",
                source="금융소비자 보호에 관한 법률",
                document_type="law",
                confidence=0.9,
                relevance_score=0.8
            )
        ]
        
        # When
        result = handler.analyze_clause(test_clause, test_context)
        
        # Then
        self.assertIsInstance(result, UnfairClauseAnalysis)
        self.assertFalse(result.is_unfair)
        self.assertEqual(result.risk_level, "Low")
        self.assertEqual(result.confidence, 0.9)
        self.assertIn("소비자 보호", result.reason)
        self.assertIn("금융소비자 보호에 관한 법률", result.legal_basis)
        
        # API 호출 확인
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], self.test_model)
        self.assertEqual(call_args[1]['temperature'], 0.1)
        self.assertEqual(call_args[1]['max_tokens'], 1000)
    
    @patch('models.llm_handler.openai.OpenAI')
    def test_analyze_clause_api_failure(self, mock_openai):
        """API 호출 실패 시 기본값 반환 테스트"""
        # Given
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API 오류")
        mock_openai.return_value = mock_client
        
        handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
        
        test_clause = "테스트 조항"
        test_context = []
        
        # When
        result = handler.analyze_clause(test_clause, test_context)
        
        # Then
        self.assertIsInstance(result, UnfairClauseAnalysis)
        self.assertFalse(result.is_unfair)
        self.assertEqual(result.risk_level, "Low")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.reason, "AI 분석 실패로 인해 수동 검토가 필요합니다.")
        self.assertEqual(result.suggestion, "수동 검토를 통해 불공정 여부를 판단하세요.")
        self.assertEqual(result.legal_basis, [])
    
    def test_parse_response_valid_json(self):
        """유효한 JSON 응답 파싱 테스트"""
        # Given
        with patch('models.llm_handler.openai.OpenAI'):
            handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
            
            valid_json_response = json.dumps({
                "is_unfair": True,
                "reason": "일방적 불이익 조항입니다.",
                "risk_level": "High",
                "suggestion": "양 당사자의 권리를 균형있게 설정하세요.",
                "confidence": 0.8,
                "legal_basis": ["약관 규제에 관한 법률"]
            })
        
        # When
        result = handler._parse_response(valid_json_response)
        
        # Then
        self.assertIsInstance(result, UnfairClauseAnalysis)
        self.assertTrue(result.is_unfair)
        self.assertEqual(result.risk_level, "High")
        self.assertEqual(result.confidence, 0.8)
        self.assertIn("일방적 불이익", result.reason)
        self.assertIn("약관 규제에 관한 법률", result.legal_basis)
    
    def test_parse_response_invalid_json(self):
        """유효하지 않은 JSON 응답 처리 테스트"""
        # Given
        with patch('models.llm_handler.openai.OpenAI'):
            handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
            
            invalid_response = "이것은 유효하지 않은 JSON입니다."
        
        # When
        result = handler._parse_response(invalid_response)
        
        # Then
        self.assertIsInstance(result, UnfairClauseAnalysis)
        self.assertFalse(result.is_unfair)
        self.assertEqual(result.risk_level, "Low")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.reason, "AI 분석 실패로 인해 수동 검토가 필요합니다.")
    
    def test_parse_response_with_code_blocks(self):
        """코드 블록이 포함된 응답 파싱 테스트"""
        # Given
        with patch('models.llm_handler.openai.OpenAI'):
            handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
            
            response_with_code_blocks = f"""```json
{json.dumps({
    "is_unfair": False,
    "reason": "공정한 조항입니다.",
    "risk_level": "Low",
    "suggestion": "현재 조항 유지",
    "confidence": 0.7,
    "legal_basis": []
})}
```"""
        
        # When
        result = handler._parse_response(response_with_code_blocks)
        
        # Then
        self.assertIsInstance(result, UnfairClauseAnalysis)
        self.assertFalse(result.is_unfair)
        self.assertEqual(result.risk_level, "Low")
        self.assertEqual(result.confidence, 0.7)
    
    def test_format_legal_context(self):
        """법적 맥락 포맷팅 테스트"""
        # Given
        with patch('models.llm_handler.openai.OpenAI'):
            handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
            
            legal_context = [
                LegalContext(
                    content="소비자 보호 관련 법령",
                    source="금융소비자 보호에 관한 법률",
                    document_type="law",
                    confidence=0.9,
                    relevance_score=0.8
                ),
                LegalContext(
                    content="불공정 약관 사례",
                    source="보도자료.pdf",
                    document_type="reference",
                    confidence=0.6,
                    relevance_score=0.5
                )
            ]
        
        # When
        formatted = handler._format_legal_context(legal_context)
        
        # Then
        self.assertIn("법령", formatted)
        self.assertIn("보도자료/사례", formatted)
        self.assertIn("금융소비자 보호에 관한 법률", formatted)
        self.assertIn("보도자료.pdf", formatted)
    
    def test_format_legal_context_empty(self):
        """빈 법적 맥락 포맷팅 테스트"""
        # Given
        with patch('models.llm_handler.openai.OpenAI'):
            handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
        
        # When
        formatted = handler._format_legal_context([])
        
        # Then
        self.assertEqual(formatted, "관련 법적 맥락이 없습니다.")
    
    def test_get_default_analysis(self):
        """기본 분석 결과 반환 테스트"""
        # Given
        with patch('models.llm_handler.openai.OpenAI'):
            handler = LLMHandler(api_key=self.test_api_key, model=self.test_model)
        
        # When
        result = handler._get_default_analysis()
        
        # Then
        self.assertIsInstance(result, UnfairClauseAnalysis)
        self.assertFalse(result.is_unfair)
        self.assertEqual(result.risk_level, "Low")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.reason, "AI 분석 실패로 인해 수동 검토가 필요합니다.")
        self.assertEqual(result.suggestion, "수동 검토를 통해 불공정 여부를 판단하세요.")
        self.assertEqual(result.legal_basis, [])

class TestUnfairClauseAnalysis(unittest.TestCase):
    """UnfairClauseAnalysis 모델 테스트"""
    
    def test_valid_analysis(self):
        """유효한 분석 결과 생성 테스트"""
        # Given & When
        analysis = UnfairClauseAnalysis(
            is_unfair=True,
            reason="일방적 불이익 조항으로 판단됩니다.",
            risk_level="High",
            suggestion="양 당사자의 권리를 균형있게 설정하세요.",
            confidence=0.8,
            legal_basis=["약관 규제에 관한 법률", "금융소비자 보호에 관한 법률"]
        )
        
        # Then
        self.assertTrue(analysis.is_unfair)
        self.assertEqual(analysis.risk_level, "High")
        self.assertEqual(analysis.confidence, 0.8)
        self.assertEqual(len(analysis.legal_basis), 2)
    
    def test_invalid_risk_level(self):
        """유효하지 않은 위험도 레벨 테스트"""
        # Given & When & Then
        with self.assertRaises(Exception):
            UnfairClauseAnalysis(
                is_unfair=True,
                reason="테스트 이유",
                risk_level="Invalid",  # 유효하지 않은 값
                suggestion="테스트 제안",
                confidence=0.5,
                legal_basis=[]
            )
    
    def test_confidence_out_of_range(self):
        """신뢰도 범위 초과 테스트"""
        # Given & When & Then
        with self.assertRaises(Exception):
            UnfairClauseAnalysis(
                is_unfair=True,
                reason="테스트 이유",
                risk_level="High",
                suggestion="테스트 제안",
                confidence=1.5,  # 범위 초과
                legal_basis=[]
            )

if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)
