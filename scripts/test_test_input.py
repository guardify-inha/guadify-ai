"""
test_input.csv 데이터 테스트 - LLM 제외 버전
불공정 여부 단일 컬럼 기반 테스트

test_ai_csv.py랑동일한데 test_input.csv를 사용하는 테스트
"""
import pandas as pd
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Dict

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# Tokenizer 경고 제거
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from database.neo4j_connector import Neo4jConnector
from rag.hybrid_graphrag import HybridGraphRAG
from judge.graphrag_judge import GraphRAGJudge

# 로그 파일 경로
LOG_FILE = project_root / 'data' / 'test' / 'ai_csv_test_log.txt'


class TeeOutput:
    """콘솔 + 파일 출력"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class GraphRAGJudgeNoLLM(GraphRAGJudge):
    """LLM 제외 테스트용 Judge"""

    def _preprocess_user_input(self, user_text: str) -> Dict:
        """LLM 전처리 스킵 - 원본 텍스트 그대로 사용"""
        print(f"  ⏭️  LLM 전처리 스킵 (테스트 모드)")
        print(f"  ✅ 원본 텍스트 그대로 사용: {user_text[:50]}...\n")
        
        return {
            'is_multiple_clauses': False,
            'first_clause_raw': user_text,
            'needs_summary': False,
            'final_input': user_text,  # 원본 그대로 사용
            'processing_note': 'LLM 전처리 스킵 (테스트 모드)',
            'original_length': len(user_text),
            'final_length': len(user_text)
        }

    def _llm_semantic_reversal_check(self, *args, **kwargs):
        """Phase 6 스킵"""
        formula_score = kwargs.get('formula_score', args[1] if len(args) > 1 else 0.5)
        print(f"  ⏭️  LLM 의미 반전 검증 스킵")
        print(f"  ✅ 수식 점수 그대로 사용: {formula_score:.3f}\n")

        return {
            'adjusted_score': formula_score,
            'reasoning': "LLM 검증 스킵 (테스트 모드)",
            'is_reversed': False
        }

    def _generate_explanation(self, *args, **kwargs):
        """설명 생성 스킵"""
        confidence_expression = kwargs.get('confidence_expression', "판단 완료")
        pattern_analysis = kwargs.get('pattern_analysis', {})
        matched_count = len(pattern_analysis.get('matched_keywords', []))
        return f"{confidence_expression}. 패턴 분석: {matched_count}개 위험 키워드"

    def _generate_suggestion(self, *args, **kwargs):
        """수정 제안 스킵"""
        pattern_analysis = kwargs.get('pattern_analysis', {})
        risk_keywords = [kw['keyword'] for kw in pattern_analysis.get('matched_keywords', [])[:3]]
        return f"위험 키워드 제거/완화 권장: {', '.join(risk_keywords)}" if risk_keywords else "고의·중과실 책임 명시 권장"


def main():
    # 콘솔 + 파일 출력 준비
    tee = TeeOutput(LOG_FILE)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("="*80)
        print("test_input.csv 데이터 테스트 - LLM 제외 버전")
        print("="*80)
        print(f"📄 전체 로그: {LOG_FILE}\n")

        # 1. 데이터 로드
        csv_path = project_root / 'data' / 'test' / 'test_input.csv'

        if not csv_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
            return

        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.replace('\ufeff', '')

        print(f"📂 데이터 파일: {csv_path}")
        print(f"   전체 행 수: {len(df)}")
        print(f"   컬럼: {list(df.columns)}\n")

        # 유효 데이터만
        df_valid = df[df["입력 문장"].notna() & (df["입력 문장"].str.strip() != "")]
        df_valid = df_valid.reset_index(drop=True)

        print(f"✅ 유효한 테스트 데이터: {len(df_valid)}개")
        print(f"⏭️  LLM 사용 제외됨 (전처리, Phase 6, 설명, 수정 제안)\n")

        # 2. Judge 초기화
        print("GraphRAGJudge 초기화 중...")
        conn = Neo4jConnector()
        rag = HybridGraphRAG(
            driver=conn.driver,
            openai_api_key=os.getenv('OPENAI_API_KEY', '')
        )
        judge = GraphRAGJudgeNoLLM(rag, conn)
        print("초기화 완료\n")

        # 결과 저장용
        results = []

        print("="*80)
        print("단일 문장 테스트 시작")
        print("="*80)

        # 3. 테스트 반복
        for idx, row in df_valid.iterrows():
            text = row["입력 문장"]
            expected_label = row["위반여부"].strip().upper()  # O 또는 X

            expected_violation = True if expected_label == "O" else False

            print(f"\n[{idx+1}/{len(df_valid)}] 테스트 중...")
            print(f"   입력: {text[:80]}... (expected violation={expected_violation})")

            try:
                result = judge.judge_clause(text)
                predicted_violation = result.get("violation", False)
                confidence = result.get("confidence", 0.0)

                correct = (predicted_violation == expected_violation)

                status = "✅" if correct else "❌"
                print(f"{status} 결과: violation={predicted_violation}, conf={confidence:.3f}")

                results.append({
                    "index": int(idx),
                    "text": text[:120],
                    "expected_violation": expected_violation,
                    "predicted_violation": predicted_violation,
                    "confidence": confidence,
                    "correct": correct
                })

            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                results.append({
                    "index": int(idx),
                    "text": text[:120],
                    "error": str(e)
                })

        # 4. 집계
        print("\n" + "="*80)
        print("📊 최종 결과")
        print("="*80)

        valid = [r for r in results if "error" not in r]

        tp = sum(1 for r in valid if r["expected_violation"] is False and r["correct"])
        tn = sum(1 for r in valid if r["expected_violation"] is True and r["correct"])
        fp = sum(1 for r in valid if r["expected_violation"] is True and not r["correct"])
        fn = sum(1 for r in valid if r["expected_violation"] is False and not r["correct"])

        total = len(valid)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 신뢰도 점수 평균 계산
        unfair_confidences = [r['confidence'] for r in valid if r['expected_violation'] is True]
        fair_confidences = [r['confidence'] for r in valid if r['expected_violation'] is False]
        all_confidences = unfair_confidences + fair_confidences

        avg_all = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        avg_unfair = sum(unfair_confidences) / len(unfair_confidences) if unfair_confidences else 0
        avg_fair = sum(fair_confidences) / len(fair_confidences) if fair_confidences else 0

        print(f"\n📈 신뢰도 점수 평균:")
        print(f"  전체 평균:          {avg_all:.3f}")
        print(f"  불공정 문장 평균:   {avg_unfair:.3f}")
        print(f"  공정 문장 평균:     {avg_fair:.3f}")
        print(f"  차이 (불공정-공정): {avg_unfair - avg_fair:+.3f}")

        print(f"\n혼동 행렬:")
        print(f"           예측:공정   예측:불공정")
        print(f"실제:공정      {tp:3d}         {fn:3d}")
        print(f"실제:불공정    {fp:3d}         {tn:3d}")

        print(f"\n전체 성능 지표:")
        print(f"  정확도 Accuracy:  {accuracy*100:.1f}%")
        print(f"  정밀도 Precision: {precision*100:.1f}%")
        print(f"  재현율 Recall:    {recall*100:.1f}%")
        print(f"  F1 Score:         {f1:.3f}")

        # 5. json 저장
        output = {
            "metadata": {
                "test_date": datetime.now().isoformat(),
                "source": "test_input.csv",
                "total_tests": len(df_valid),
                "llm_excluded": ["전처리", "Phase 6", "설명", "수정 제안"],
            },
            "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "confidence_averages": {
                "overall": avg_all,
                "unfair_cases": avg_unfair,
                "fair_cases": avg_fair,
                "difference": avg_unfair - avg_fair
            },
            "results": results
        }

        output_path = project_root / 'data' / 'test' / 'ai_csv_test_results.json'
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("\n" + "="*80)
        print(f"💾 결과 저장 완료: {output_path}")
        print(f"📄 전체 로그: {LOG_FILE}")
        print("="*80)

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
