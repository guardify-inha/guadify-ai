"""
AI 기반 불공정 약관 탐지 시스템

LLM 핸들러를 활용한 맥락 기반 불공정 조항 탐지
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm_handler import LLMHandler, LegalContext, UnfairClauseAnalysis
from db.neo4j_client import Neo4jClient
from db.graph_retriever import GraphRetriever
from db.entity_extractor import LegalEntityExtractor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """탐지 결과"""
    clause_text: str
    analysis: UnfairClauseAnalysis
    legal_context: List[LegalContext]
    clause_index: int

class AIUnfairDetector:
    """AI 기반 불공정 약관 탐지기"""
    
    def __init__(self):
        """초기화"""
        self.llm_handler = LLMHandler()
        self.neo4j_client = Neo4jClient()
        self.graph_retriever = GraphRetriever(self.neo4j_client)
        self.entity_extractor = LegalEntityExtractor()
        
        # 문서 타입별 우선순위
        self.document_priority = {
            "law": {"weight": 1.0, "description": "법령"},
            "standard": {"weight": 0.8, "description": "표준약관/지침"},
            "reference": {"weight": 0.3, "description": "보도자료/사례"}
        }
        
        logger.info("AI 기반 불공정 약관 탐지기 초기화 완료")
    
    def detect_unfair_clauses(self, contract_text: str) -> Dict[str, Any]:
        """AI 기반 불공정 조항 탐지"""
        logger.info(f"🤖 AI 기반 불공정 조항 탐지 시작: {len(contract_text)}자")
        
        # 문장 단위로 분할
        sentences = self._split_into_sentences(contract_text)
        
        detection_results = []
        total_risk_score = 0
        unfair_count = 0
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
                
            logger.info(f"조항 {i+1}/{len(sentences)} 분석 중...")
            
            # 1. 법적 맥락 수집
            legal_context = self._collect_legal_context(sentence)
            
            # 2. AI 분석
            analysis = self.llm_handler.analyze_clause(sentence, legal_context)
            
            # 3. 결과 저장
            result = DetectionResult(
                clause_text=sentence,
                analysis=analysis,
                legal_context=legal_context,
                clause_index=i
            )
            detection_results.append(result)
            
            if analysis.is_unfair:
                unfair_count += 1
                total_risk_score += analysis.confidence
        
        # 전체 위험도 계산
        overall_risk = total_risk_score / unfair_count if unfair_count > 0 else 0
        
        return {
            "input_text": contract_text,
            "detection_results": detection_results,
            "unfair_count": unfair_count,
            "total_clauses": len(detection_results),
            "overall_risk_score": overall_risk,
            "risk_level": self._get_risk_level(overall_risk),
            "summary": self._generate_summary(detection_results, overall_risk)
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        clauses = []
        current_clause = ""
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('제') and '조' in line:
                if current_clause:
                    clauses.append(current_clause)
                current_clause = line
            else:
                current_clause += " " + line
        
        if current_clause:
            clauses.append(current_clause)
        
        return clauses
    
    def _collect_legal_context(self, sentence: str) -> List[LegalContext]:
        """Graph RAG로 법적 맥락 수집"""
        legal_context = []
        
        try:
            # 엔티티 추출
            entities = self.entity_extractor.extract_entities(sentence, "input", "clause")
            
            for entity in entities:
                if entity.entity_type.value in ["Article", "Law"]:
                    # Graph RAG 검색
                    search_results = self.graph_retriever.search(entity.text, max_results=3)
                    
                    for result in search_results:
                        doc_type = self._get_document_type(result.source_file)
                        context = LegalContext(
                            content=result.context,
                            source=result.source_file,
                            document_type=doc_type,
                            confidence=result.confidence,
                            relevance_score=self._calculate_relevance(sentence, result.entity_text)
                        )
                        legal_context.append(context)
            
            # 가중치 기준으로 정렬
            legal_context.sort(key=lambda x: x.confidence * self.document_priority[x.document_type]["weight"], reverse=True)
            
        except Exception as e:
            logger.error(f"법적 맥락 수집 중 오류: {e}")
        
        return legal_context[:5]  # 상위 5개만 반환
    
    def _get_document_type(self, source_file: str) -> str:
        """파일 경로에서 문서 타입 추출"""
        if "law/" in source_file:
            return "law"
        elif "standard/" in source_file:
            return "standard"
        elif "reference/" in source_file:
            return "reference"
        return "reference"  # 기본값
    
    def _calculate_relevance(self, sentence: str, entity_text: str) -> float:
        """관련성 점수 계산"""
        relevance = 0.0
        
        # 키워드 매칭
        sentence_lower = sentence.lower()
        entity_lower = entity_text.lower()
        
        if any(keyword in sentence_lower for keyword in entity_lower.split()):
            relevance += 0.5
        
        # 법적 개념 관련성
        legal_keywords = ["불공정", "약관", "소비자", "권리", "보호", "손해", "배상", "해지"]
        if any(keyword in sentence_lower for keyword in legal_keywords):
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """위험도 레벨 결정"""
        if score >= 0.8:
            return "매우높음"
        elif score >= 0.6:
            return "높음"
        elif score >= 0.4:
            return "보통"
        else:
            return "낮음"
    
    def _generate_summary(self, results: List[DetectionResult], overall_risk: float) -> str:
        """요약 생성"""
        unfair_count = sum(1 for r in results if r.analysis.is_unfair)
        high_confidence_count = sum(1 for r in results if r.analysis.is_unfair and r.analysis.confidence >= 0.7)
        
        summary = f"총 {len(results)}개 조항 중 {unfair_count}개의 불공정 조항이 발견되었습니다. "
        summary += f"이 중 {high_confidence_count}개가 높은 신뢰도(≥0.7)로 탐지되었습니다. "
        summary += f"전체 위험도는 {overall_risk:.2f}입니다."
        
        return summary
    
    def close(self):
        """리소스 정리"""
        self.neo4j_client.close()

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI 기반 불공정 약관 탐지')
    parser.add_argument('--file', help='분석할 약관 파일 경로', default='test_inputs/sample_contract.txt')
    
    args = parser.parse_args()
    
    # 파일 읽기
    if not os.path.exists(args.file):
        print(f"❌ 파일을 찾을 수 없습니다: {args.file}")
        return
    
    with open(args.file, 'r', encoding='utf-8') as f:
        contract_text = f.read()
    
    # 탐지 실행
    detector = AIUnfairDetector()
    result = detector.detect_unfair_clauses(contract_text)
    detector.close()
    
    # 결과 출력
    print("=" * 80)
    print("🤖 AI 기반 불공정 약관 탐지 결과")
    print("=" * 80)
    print(f"📊 전체 위험도: {result['overall_risk_score']:.2f} ({result['risk_level']})")
    print(f"📝 요약: {result['summary']}")
    print()
    
    for i, detection in enumerate(result['detection_results'], 1):
        if detection.analysis.is_unfair:
            print(f"🚨 불공정 조항 {i}:")
            print(f"   조항: {detection.clause_text}")
            print(f"   위험도: {detection.analysis.risk_level} (신뢰도: {detection.analysis.confidence:.2f})")
            print(f"   판단 근거: {detection.analysis.reason}")
            
            if detection.analysis.legal_basis:
                print(f"   📚 법적 근거: {', '.join(detection.analysis.legal_basis)}")
            
            if detection.legal_context:
                print("   🔗 관련 법적 맥락:")
                for ctx in detection.legal_context[:2]:
                    doc_desc = detector.document_priority[ctx.document_type]["description"]
                    print(f"     - {doc_desc}: {ctx.content[:100]}...")
            
            if detection.analysis.suggestion:
                print(f"   💡 개선 제안: {detection.analysis.suggestion}")
            
            print()
    
    # 파일 출력 (자동으로 results 폴더에 저장)
    # 타임스탬프 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"결과 폴더 생성: {output_dir}")
    
    # 결과를 JSON으로 변환
    json_result = {
            "input_text": result["input_text"],
            "unfair_count": result["unfair_count"],
            "total_clauses": result["total_clauses"],
            "overall_risk_score": result["overall_risk_score"],
            "risk_level": result["risk_level"],
            "summary": result["summary"],
            "analysis_timestamp": timestamp,
            "unfair_clauses": [
                {
                    "clause_text": d.clause_text,
                    "is_unfair": d.analysis.is_unfair,
                    "risk_level": d.analysis.risk_level,
                    "confidence": d.analysis.confidence,
                    "reason": d.analysis.reason,
                    "legal_basis": d.analysis.legal_basis,
                    "suggestion": d.analysis.suggestion,
                    "legal_context": [
                        {
                            "content": ctx.content,
                            "source": ctx.source,
                            "document_type": ctx.document_type,
                            "confidence": ctx.confidence
                        }
                        for ctx in d.legal_context
                    ]
                }
                for d in result["detection_results"] if d.analysis.is_unfair
            ]
    }
    
    # JSON 파일 저장
    json_file = os.path.join(output_dir, "ai_detection_result.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    
    # 마크다운 보고서 생성
    markdown_file = os.path.join(output_dir, "analysis_report.md")
    _generate_markdown_report(json_result, markdown_file)
    
    print(f"📁 결과가 {output_dir} 폴더에 저장되었습니다:")
    print(f"   📄 JSON: {json_file}")
    print(f"   📋 마크다운: {markdown_file}")

def _generate_markdown_report(result: Dict[str, Any], output_file: str):
    """마크다운 보고서 생성"""
    
    # 위험도 레벨에 따른 이모지
    risk_emojis = {
        "Low": "🟢",
        "Medium": "🟡", 
        "High": "🔴",
        "Critical": "🚨"
    }
    
    # 문서 타입별 이모지
    doc_emojis = {
        "law": "⚖️",
        "standard": "📋",
        "reference": "📰"
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 🛡️ AI 기반 불공정 약관 탐지 보고서\n\n")
        f.write(f"**분석 일시**: {result['analysis_timestamp']}\n\n")
        
        # 전체 요약
        f.write(f"## 📊 전체 요약\n\n")
        f.write(f"- **총 조항 수**: {result['total_clauses']}개\n")
        f.write(f"- **불공정 조항**: {result['unfair_count']}개\n")
        f.write(f"- **전체 위험도**: {result['overall_risk_score']:.2f}\n")
        f.write(f"- **위험 레벨**: {risk_emojis.get(result['risk_level'], '❓')} {result['risk_level']}\n\n")
        f.write(f"**요약**: {result['summary']}\n\n")
        
        # 불공정 조항 상세 분석
        if result['unfair_clauses']:
            f.write(f"## 🚨 불공정 조항 상세 분석\n\n")
            
            for i, clause in enumerate(result['unfair_clauses'], 1):
                f.write(f"### {i}. {risk_emojis.get(clause['risk_level'], '❓')} 위험도: {clause['risk_level']} (신뢰도: {clause['confidence']:.2f})\n\n")
                f.write(f"**조항 내용**:\n```\n{clause['clause_text']}\n```\n\n")
                
                f.write(f"**불공정 판단 근거**:\n{clause['reason']}\n\n")
                
                if clause['legal_basis']:
                    f.write(f"**📚 법적 근거**:\n")
                    for basis in clause['legal_basis']:
                        f.write(f"- {basis}\n")
                    f.write("\n")
                
                if clause['legal_context']:
                    f.write(f"**🔗 관련 법적 맥락**:\n")
                    for ctx in clause['legal_context']:
                        doc_emoji = doc_emojis.get(ctx['document_type'], '📄')
                        f.write(f"- {doc_emoji} **{ctx['document_type'].upper()}**: {ctx['content'][:100]}...\n")
                        f.write(f"  - 출처: {ctx['source']}\n")
                        f.write(f"  - 신뢰도: {ctx['confidence']:.2f}\n")
                    f.write("\n")
                
                if clause['suggestion']:
                    f.write(f"**💡 개선 제안**:\n{clause['suggestion']}\n\n")
                
                f.write("---\n\n")
        else:
            f.write(f"## ✅ 불공정 조항 없음\n\n")
            f.write(f"분석된 모든 조항이 공정한 것으로 판단되었습니다.\n\n")
        
        # 분석 정보
        f.write(f"## 🔧 분석 정보\n\n")
        f.write(f"- **분석 모델**: OpenAI GPT-4o-mini\n")
        f.write(f"- **검색 시스템**: Vector RAG + Graph RAG\n")
        f.write(f"- **법적 데이터베이스**: 약관 규제법, 금융소비자보호법, 표준약관 등\n")
        f.write(f"- **생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"---\n")
        f.write(f"*이 보고서는 AI 기반 불공정 약관 탐지 시스템에 의해 자동 생성되었습니다.*\n")

if __name__ == "__main__":
    main()
