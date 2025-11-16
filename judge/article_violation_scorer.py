"""
조항별 위반도 점수 계산 모듈

약관법 제7조~제14조 각각의 위반 정도를 개별 점수로 계산
"""

from typing import Dict, List
from pathlib import Path
import json
import re


class ArticleViolationScorer:
    """조항별 위반도 점수 계산기"""

    ARTICLES = ['제7조', '제8조', '제9조', '제10조', '제11조',
                '제12조', '제13조', '제14조']

    def __init__(self):
        self.patterns = {}
        self._load_patterns()

    def _load_patterns(self):
        """patterns_by_article_v2.json 로드"""
        try:
            current_dir = Path(__file__).parent
            pattern_path = current_dir.parent / "data" / "contracts" / "reference" / "patterns_by_article_v2.json"

            if not pattern_path.exists():
                alternative_paths = [
                    Path("data/contracts/reference/patterns_by_article_v2.json"),
                    Path("../data/contracts/reference/patterns_by_article_v2.json"),
                ]
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        pattern_path = alt_path
                        break

            if pattern_path.exists():
                with open(pattern_path, 'r', encoding='utf-8') as f:
                    self.patterns = json.load(f)
                print(f"✅ 패턴 로드: {pattern_path}")
            else:
                print(f"⚠️ 패턴 파일 없음")
                self.patterns = {}
        except Exception as e:
            print(f"⚠️ 패턴 로드 실패: {e}")
            self.patterns = {}

    def calculate_article_scores(self, user_text: str) -> Dict:
        """조항별 위반도 점수 계산"""
        if not self.patterns:
            return {}

        result = {}
        for article in self.ARTICLES:
            score_data = self._calculate_single_article(article, user_text)
            result[article] = score_data

        return result

    def _calculate_single_article(self, article: str, user_text: str) -> Dict:
        """단일 조항 점수 계산"""
        if article not in self.patterns:
            return {'score': 0.0, 'details': {}}

        article_data = self.patterns[article]
        matched_keywords = []
        matched_high_risk = []
        matched_regex = []
        matched_combined = []
        matched_exceptions = []

        # 1. keywords 매칭 (각 0.1점)
        for pattern in article_data.get('patterns', []):
            for kw in pattern.get('keywords', []):
                if kw in user_text:
                    matched_keywords.append(kw)

        # 2. high_risk_keywords 매칭 (각 0.3점)
        for pattern in article_data.get('patterns', []):
            for kw in pattern.get('high_risk_keywords', []):
                if kw in user_text:
                    matched_high_risk.append(kw)

        # 3. regex_patterns 매칭 (각 0.3점)
        for pattern_info in article_data.get('regex_patterns', []):
            try:
                if re.search(pattern_info['regex'], user_text):
                    matched_regex.append(pattern_info['keyword'])
            except:
                pass

        # 4. combined_patterns 매칭 (각 0.5점)
        combined = self.patterns.get('combined_pattern_risks', {})
        for pattern in combined.get('patterns', []):
            if article in pattern.get('articles', []):
                keywords = pattern['combination']
                if all(kw in user_text for kw in keywords):
                    matched_combined.append(keywords)

        # 5. exception_patterns 매칭 (점수 무효화 또는 대폭 감소)
        exception_penalty = 0.0
        for exception in article_data.get('exception_patterns', []):
            # 키워드 기반 예외
            if 'keywords' in exception:
                if any(kw in user_text for kw in exception['keywords']):
                    matched_exceptions.append(exception.get('description', '예외'))
                    exception_penalty += exception.get('penalty', 0.5)

            # Regex 기반 예외
            if 'regex' in exception:
                try:
                    if re.search(exception['regex'], user_text):
                        matched_exceptions.append(exception.get('description', '예외'))
                        exception_penalty += exception.get('penalty', 0.5)
                except:
                    pass

        # 점수 계산 (예외 패널티 적용)
        base_score = (
            len(matched_keywords) * 0.1 +
            len(matched_high_risk) * 0.3 +
            len(matched_regex) * 0.3 +
            len(matched_combined) * 0.5
        )
        score = max(base_score - exception_penalty, 0.0)
        score = min(score, 1.0)

        return {
            'score': score,
            'details': {
                'matched_keywords': matched_keywords,
                'matched_high_risk': matched_high_risk,
                'matched_regex': matched_regex,
                'matched_combined': matched_combined,
                'matched_exceptions': matched_exceptions
            }
        }

    def get_primary_violation(self, scores: Dict) -> Dict:
        """최고점 조항 반환"""
        if not scores:
            return {'article': None, 'score': 0.0}

        max_article = max(scores.items(), key=lambda x: x[1]['score'])
        return {
            'article': max_article[0],
            'score': max_article[1]['score'],
            'details': max_article[1]['details']
        }

    def get_top_violations(self, scores: Dict, top_k: int = 3) -> List[Dict]:
        """상위 N개 조항 반환"""
        if not scores:
            return []

        sorted_scores = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return [
            {'article': art, 'score': data['score'], 'details': data['details']}
            for art, data in sorted_scores[:top_k]
        ]
