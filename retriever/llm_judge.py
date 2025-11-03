"""
LLM-as-Judge: 질의와 후보 조항을 LLM이 직접 비교하여 위반 가능성 판단

3단계: LLM-as-Judge 및 HyDE 기법 구현
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from utils.llm_client import get_llm_client


class LLMJudge:
    """
    LLM-as-Judge: 법률 조항 위반 가능성 판단자
    
    질의와 후보 조항을 비교하여 위반 가능성을 평가
    """
    
    def __init__(self):
        self.llm = get_llm_client()
        
        # 평가 기준
        self.evaluation_criteria = """
다음 기준으로 위반 가능성을 판단하세요:
1. 질의의 핵심 내용이 법률 조항에서 금지하는 내용과 일치하는가?
2. 질의가 해당 조항을 위반할 가능성이 얼마나 높은가? (0.0 = 위반 아님, 1.0 = 명백한 위반)
3. 구체적인 근거는 무엇인가?
"""
    
    def judge_single(self, query: str, candidate: Dict) -> Dict:
        """
        단일 후보 조항에 대한 위반 가능성 판단
        
        Args:
            query: 자연어 질의 (불공정 약관 조항)
            candidate: 후보 조항 정보 (id, content, node_type 등)
            
        Returns:
            판단 결과 딕셔너리
        """
        if not self.llm:
            return {
                "violation_probability": 0.0,
                "reasoning": "LLM을 사용할 수 없습니다.",
                "confidence": 0.0
            }
        
        node_type = candidate.get('node_type', '')
        content = candidate.get('content', '')
        article_id = candidate.get('article_id', candidate.get('id', ''))
        title = candidate.get('title', '')
        
        system_prompt = """당신은 약관법 전문가입니다. 
입력된 약관 조항이 특정 법률 조항을 위반할 가능성을 정확하게 판단해야 합니다.
각 조항의 고유한 특성을 정확히 파악하고, 질의 내용과의 관련성을 면밀히 분석해야 합니다.
한국어로 명확하고 전문적인 판단을 제공하세요."""

        user_prompt = f"""
다음 불공정 약관 조항이 특정 법률 조항을 위반할 가능성을 판단하세요.

**검사할 약관 조항:**
"{query}"

**법률 조항 정보:**
- 조항 ID: {article_id}
- 노드 타입: {node_type}
- 제목: {title}
- 법률 조항 내용:
{content}

**중요 지시사항:**
1. 법률 조항의 내용을 정확히 읽고 이해하세요.
2. 약관 조항이 이 법률 조항에서 금지하는 내용과 정확히 일치하는지 확인하세요.
3. 조항별 특성을 고려하세요:
   - 제6조: 일반원칙 (부당하게 불리한 조항)
   - 제7조: 면책조항의 금지
   - 제8조: 과중한 손해배상 금지
   - 제9조: 계약 해제/해지 제한 금지
   - 제10조: 채무 이행 관련 제한 금지
   - 제11조: 고객 권익 보호
   - 제12조: 의사표시 제한 금지
   - 제13조: 대리인 책임 가중 금지
   - 제14조: 소송 제기 금지 등 금지
4. 약관 조항의 핵심 내용과 법률 조항의 핵심 내용이 얼마나 일치하는지 평가하세요.
5. 위반 가능성 점수는 0.0(위반 아님)부터 1.0(명백한 위반)까지 정확하게 산정하세요.

{self.evaluation_criteria}

JSON 형식으로 응답하세요:
{{
    "violation_probability": 0.85,
    "reasoning": "이 약관은 명백한 면책 조항으로 제7조를 위반합니다. 제7조는 사업자의 고의 또는 중대한 과실로 인한 법률상의 책임을 배제하는 조항을 무효로 규정하고 있으며, 이 약관이 정확히 그에 해당합니다.",
    "confidence": 0.9,
    "key_violation_points": ["전면 면책", "책임 배제"]
}}
"""
        
        result = self.llm.generate_json(user_prompt, system_prompt)
        
        if result:
            return {
                "violation_probability": result.get("violation_probability", 0.0),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.5),
                "key_violation_points": result.get("key_violation_points", [])
            }
        
        # LLM 실패 시 기본값
        return {
            "violation_probability": 0.0,
            "reasoning": "LLM 판단 실패",
            "confidence": 0.0
        }
    
    def judge_batch(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        여러 후보 조항을 배치로 평가하고 순위화
        
        Args:
            query: 자연어 질의
            candidates: 후보 조항 리스트
            top_k: 최종 반환할 개수
            
        Returns:
            위반 가능성 점수 기준 정렬된 후보 리스트
        """
        if not self.llm or not candidates:
            return []
        
        # 배치 평가: 각 후보에 대해 판단 수행
        judged_candidates = []
        for candidate in candidates[:10]:  # 최대 10개만 평가 (비용 절감)
            judgment = self.judge_single(query, candidate)
            
            judged_candidate = candidate.copy()
            judged_candidate['llm_violation_probability'] = judgment.get('violation_probability', 0.0)
            judged_candidate['llm_reasoning'] = judgment.get('reasoning', '')
            judged_candidate['llm_confidence'] = judgment.get('confidence', 0.5)
            judged_candidate['key_violation_points'] = judgment.get('key_violation_points', [])
            
            # 기존 점수와 LLM 점수 결합 (LLM 점수가 더 중요)
            existing_score = candidate.get('final_score', 0.0)
            llm_score = judgment.get('violation_probability', 0.0)
            
            # 통합 점수: 하이브리드 검색 점수(40%) + LLM 판단 점수(60%)
            integrated_score = (existing_score * 0.4) + (llm_score * 0.6)
            judged_candidate['integrated_score'] = integrated_score
            
            judged_candidates.append(judged_candidate)
        
        # 통합 점수 기준 정렬
        judged_candidates.sort(key=lambda x: x.get('integrated_score', 0.0), reverse=True)
        
        return judged_candidates[:top_k]
    
    def compare_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        여러 후보를 비교하여 상대적 순위 결정
        
        LLM에게 모든 후보를 한 번에 제공하여 상대적 평가 수행
        """
        if not self.llm or len(candidates) < 2:
            return self.judge_batch(query, candidates)
        
        # 후보가 너무 많으면 일부만 비교
        candidates_to_compare = candidates[:5]
        
        system_prompt = """당신은 약관법 전문가입니다.
여러 법률 조항 중에서 입력된 약관 조항을 가장 잘 위반할 가능성이 높은 조항을 정확히 찾아야 합니다.
각 조항의 고유한 특성을 정확히 파악하고, 약관 조항과의 관련성을 면밀히 비교 분석해야 합니다.
한국어로 명확하고 전문적인 판단을 제공하세요."""

        candidates_text = ""
        for i, candidate in enumerate(candidates_to_compare, 1):
            article_id = candidate.get('article_id', candidate.get('id', ''))
            content = candidate.get('content', '')
            title = candidate.get('title', '')
            node_type = candidate.get('node_type', '')
            
            candidates_text += f"""
후보 {i}:
- 조항 ID: {article_id}
- 노드 타입: {node_type}
- 제목: {title}
- 법률 조항 내용:
{content}
"""
        
        user_prompt = f"""
다음 불공정 약관 조항을 검사합니다:
"{query}"

위 약관 조항이 아래 후보 법률 조항들 중 어떤 조항을 위반할 가능성이 가장 높은지 비교하고 순위를 매기세요.

{candidates_text}

**중요 비교 기준:**
1. 약관 조항의 핵심 내용과 각 법률 조항이 금지하는 내용의 일치도
2. 각 조항별 특성 고려:
   - 제6조: 일반원칙 (부당하게 불리한 조항, 공정성 잃은 조항)
   - 제7조: 면책조항의 금지 (책임 배제, 손해배상 범위 제한, 담보책임 배제 등)
   - 제8조: 과중한 손해배상금 금지 (지연 손해금 등)
   - 제9조: 계약 해제/해지 권한 제한 금지
   - 제10조: 채무 이행 관련 제한 금지 (급부 내용 일방 결정 등)
   - 제11조: 고객 권익 보호 (항변권, 상계권 등 고객 권리 제한 금지)
   - 제12조: 의사표시 제한 금지
   - 제13조: 대리인 책임 가중 금지
   - 제14조: 소송 제기 금지 등 금지
3. 약관 조항이 각 법률 조항의 핵심 내용과 얼마나 정확히 일치하는지 평가
4. 위반 가능성 점수는 0.0(위반 아님)부터 1.0(명백한 위반)까지 정확하게 산정
5. 각 후보에 대해 독립적으로 판단하되, 상대적 비교를 통해 순위 결정

{self.evaluation_criteria}

JSON 형식으로 응답하세요 (모든 후보에 대해 위반 가능성 점수와 순위 포함, 후보 개수만큼 반드시 포함):
{{
    "rankings": [
        {{
            "candidate_index": 1,
            "violation_probability": 0.95,
            "reasoning": "이 약관은 명백히 제7조를 위반합니다. 제7조는 사업자의 고의 또는 중대한 과실로 인한 법률상의 책임을 배제하는 조항을 무효로 규정하고 있으며, '어떠한 피해배상도 하지않는다'는 내용이 정확히 이에 해당합니다.",
            "rank": 1
        }},
        {{
            "candidate_index": 2,
            "violation_probability": 0.3,
            "reasoning": "이 약관은 제11조와도 관련이 있지만, 제7조에 비해 직접적 위반 정도가 낮습니다.",
            "rank": 2
        }}
    ]
}}
"""
        
        result = self.llm.generate_json(user_prompt, system_prompt)
        
        if result and result.get('rankings'):
            rankings = result.get('rankings', [])
            ranked_dict = {r.get('candidate_index', 0): r for r in rankings}
            
            # 원본 후보에 순위 및 점수 추가
            for i, candidate in enumerate(candidates_to_compare, 1):
                if i in ranked_dict:
                    rank_info = ranked_dict[i]
                    candidate['llm_violation_probability'] = rank_info.get('violation_probability', 0.0)
                    candidate['llm_reasoning'] = rank_info.get('reasoning', '')
                    candidate['llm_rank'] = rank_info.get('rank', i)
                    
                    # 통합 점수 계산
                    existing_score = candidate.get('final_score', 0.0)
                    llm_score = rank_info.get('violation_probability', 0.0)
                    candidate['integrated_score'] = (existing_score * 0.4) + (llm_score * 0.6)
            
            # 통합 점수 기준 재정렬
            candidates_to_compare.sort(key=lambda x: x.get('integrated_score', 0.0), reverse=True)
            return candidates_to_compare
        
        # 비교 실패 시 개별 판단으로 폴백
        return self.judge_batch(query, candidates)

