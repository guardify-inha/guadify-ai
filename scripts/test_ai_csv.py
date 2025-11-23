"""
AI.csv 데이터 테스트 - LLM 제외 버전
불공정 약관 원문 / 수정 후 약관 조항 테스트
"""
import pandas as pd
import sys
from pathlib import Path
import os
import json
from datetime import datetime

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
    """콘솔과 파일에 동시 출력"""
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
    """LLM 사용을 제외한 테스트용 Judge"""

    def _llm_semantic_reversal_check(self, *args, **kwargs):
        """Phase 6: LLM 의미 반전 검증 - 스킵"""
        formula_score = kwargs.get('formula_score', args[1] if len(args) > 1 else 0.5)

        print(f"  ⏭️  LLM 의미 반전 검증 스킵")
        print(f"  ✅ 수식 점수 그대로 사용: {formula_score:.3f}\n")

        return {
            'adjusted_score': formula_score,
            'reasoning': "LLM 검증 스킵 (테스트 모드)",
            'is_reversed': False
        }

    def _generate_explanation(self, *args, **kwargs):
        """Phase 8: 설명 생성 - 스킵"""
        confidence_expression = kwargs.get('confidence_expression', args[5] if len(args) > 5 else "판단 완료")
        pattern_analysis = kwargs.get('pattern_analysis', args[3] if len(args) > 3 else {})

        matched_count = len(pattern_analysis.get('matched_keywords', []))

        return f"{confidence_expression}. 패턴 분석: {matched_count}개 위험 키워드"

    def _generate_suggestion(self, *args, **kwargs):
        """수정 제안 생성 - 스킵"""
        pattern_analysis = kwargs.get('pattern_analysis', args[0] if len(args) > 0 else {})
        risk_keywords = [kw['keyword'] for kw in pattern_analysis.get('matched_keywords', [])[:3]]

        if risk_keywords:
            return f"위험 키워드 제거/완화 권장: {', '.join(risk_keywords)}"
        return "고의·중과실 책임 명시, 불가항력 구체화 권장"


def main():
    # 로그 파일과 콘솔에 동시 출력
    tee = TeeOutput(LOG_FILE)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("="*80)
        print("AI.csv 데이터 테스트 - LLM 제외 버전")
        print("="*80)
        print(f"📄 전체 로그: {LOG_FILE}\n")

        # 1. 데이터 로드
        csv_path = project_root / 'data' / 'contracts' / 'reference' / 'ai.csv'

        if not csv_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
            return

        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.replace('\ufeff', '')

        print(f"📂 데이터 파일: {csv_path}")
        print(f"   전체 행 수: {len(df)}")
        print(f"   컬럼: {list(df.columns)}\n")

        # 유효한 행만 (불공정 약관 원문과 수정 후 약관 조항이 모두 있는 행)
        df_valid = df[
            df['불공정 약관 원문'].notna() &
            (df['불공정 약관 원문'].str.strip() != '') &
            df['수정 후 약관 조항'].notna() &
            (df['수정 후 약관 조항'].str.strip() != '')
        ].reset_index(drop=True)

        # 100개로 제한
        df_sample = df_valid.head(100)

        print(f"✅ 유효한 데이터: {len(df_valid)}개")
        print(f"   샘플링: {len(df_sample)}개 사용")
        print(f"   총 테스트: {len(df_sample) * 2}개 (불공정 {len(df_sample)} + 공정 {len(df_sample)})")
        print(f"   ⏭️  LLM 사용 제외: Phase 6 (의미 반전), Phase 8 (설명), 수정 제안\n")

        # 2. GraphRAGJudge 초기화 (LLM 제외 버전)
        print("GraphRAGJudge 초기화 중 (LLM 제외 모드)...")
        conn = Neo4jConnector()
        rag = HybridGraphRAG(
            driver=conn.driver,
            openai_api_key=os.getenv('OPENAI_API_KEY', '')
        )
        judge = GraphRAGJudgeNoLLM(rag, conn)  # LLM 제외 버전 사용
        print("초기화 완료\n")

        # 결과 저장
        results = {
            'unfair': [],  # 불공정 원문 테스트
            'fair': []     # 공정 수정본 테스트
        }

        # 3. Part 1: 불공정 원문 테스트
        print("="*80)
        print(f"Part 1: 불공정 원문 테스트 ({len(df_sample)}개)")
        print("="*80)

        for idx, row in df_sample.iterrows():
            text = row['불공정 약관 원문']

            try:
                print(f"\n[{idx+1}/{len(df_sample)}] 테스트 중...")
                print(f"   입력: {text[:80]}...")

                result = judge.judge_clause(text)

                is_violation = result.get('violation', False)
                confidence = result.get('confidence', 0.0)
                detected_article = result.get('primary_evidence', {}).get('article_id', None)

                # TN: 불공정을 불공정으로 판단 (정답)
                correct = is_violation

                results['unfair'].append({
                    'index': int(idx),
                    'text': text[:100],
                    'expected': True,
                    'predicted': is_violation,
                    'confidence': confidence,
                    'detected_article': detected_article,
                    'correct': correct
                })

                status = "✅" if correct else "❌"
                print(f"\n{status} 결과: violation={is_violation}, conf={confidence:.3f}, article={detected_article}")

            except Exception as e:
                print(f"\n❌ 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                results['unfair'].append({
                    'index': int(idx),
                    'text': text[:100],
                    'error': str(e)
                })

        # 4. Part 2: 공정 수정본 테스트
        print("\n" + "="*80)
        print(f"Part 2: 공정 수정본 테스트 ({len(df_sample)}개)")
        print("="*80)

        for idx, row in df_sample.iterrows():
            text = row['수정 후 약관 조항']

            try:
                print(f"\n[{idx+1}/{len(df_sample)}] 테스트 중...")
                print(f"   입력: {text[:80]}...")

                result = judge.judge_clause(text)

                is_violation = result.get('violation', False)
                confidence = result.get('confidence', 0.0)

                # TP: 공정을 공정으로 판단 (정답)
                correct = not is_violation

                results['fair'].append({
                    'index': int(idx),
                    'text': text[:100],
                    'expected': False,
                    'predicted': is_violation,
                    'confidence': confidence,
                    'correct': correct
                })

                status = "✅" if correct else "❌"
                print(f"\n{status} 결과: violation={is_violation}, conf={confidence:.3f}")

            except Exception as e:
                print(f"\n❌ 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                results['fair'].append({
                    'index': int(idx),
                    'text': text[:100],
                    'error': str(e)
                })

        # 5. 결과 집계
        print("\n" + "="*80)
        print("📊 최종 결과")
        print("="*80)

        # 오류 제거
        unfair_valid = [r for r in results['unfair'] if 'error' not in r]
        fair_valid = [r for r in results['fair'] if 'error' not in r]

        unfair_errors = len(results['unfair']) - len(unfair_valid)
        fair_errors = len(results['fair']) - len(fair_valid)

        print(f"\n에러 발생:")
        print(f"  Part 1 (불공정): {unfair_errors}개")
        print(f"  Part 2 (공정): {fair_errors}개")

        # Confusion Matrix (Fair=Positive, Unfair=Negative)
        tn = sum(1 for r in unfair_valid if r['correct'])  # 불공정→불공정
        fp = len(unfair_valid) - tn                         # 불공정→공정
        tp = sum(1 for r in fair_valid if r['correct'])    # 공정→공정
        fn = len(fair_valid) - tp                           # 공정→불공정

        total = tp + tn + fp + fn

        # 신뢰도 점수 평균 계산
        unfair_confidences = [r['confidence'] for r in unfair_valid]
        fair_confidences = [r['confidence'] for r in fair_valid]
        all_confidences = unfair_confidences + fair_confidences

        avg_all = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        avg_unfair = sum(unfair_confidences) / len(unfair_confidences) if unfair_confidences else 0
        avg_fair = sum(fair_confidences) / len(fair_confidences) if fair_confidences else 0

        if total > 0:
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n📈 신뢰도 점수 평균:")
            print(f"  전체 평균:          {avg_all:.3f}")
            print(f"  불공정 원문 평균:   {avg_unfair:.3f}")
            print(f"  공정 수정본 평균:   {avg_fair:.3f}")
            print(f"  차이 (불공정-공정): {avg_unfair - avg_fair:+.3f}")

            print(f"\nPart 1: 불공정 원문 ({len(unfair_valid)}개)")
            print(f"  TN (불공정→불공정): {tn}개 ({tn/len(unfair_valid)*100:.1f}%)")
            print(f"  FP (불공정→공정): {fp}개 ({fp/len(unfair_valid)*100:.1f}%)")

            print(f"\nPart 2: 공정 수정본 ({len(fair_valid)}개)")
            print(f"  TP (공정→공정): {tp}개 ({tp/len(fair_valid)*100:.1f}%)")
            print(f"  FN (공정→불공정): {fn}개 ({fn/len(fair_valid)*100:.1f}%)")

            print(f"\n혼동 행렬 (Fair=Positive):")
            print(f"           예측:공정   예측:불공정")
            print(f"실제:공정      {tp:3d}         {fn:3d}")
            print(f"실제:불공정    {fp:3d}         {tn:3d}")

            print(f"\n전체 성능 지표:")
            print(f"  정확도 (Accuracy):  {accuracy*100:.1f}%")
            print(f"  정밀도 (Precision): {precision*100:.1f}%")
            print(f"  재현율 (Recall):    {recall*100:.1f}%")
            print(f"  F1 Score:          {f1:.3f}")
        else:
            print("\n⚠️ 유효한 테스트 결과가 없습니다.")
            accuracy = precision = recall = f1 = 0

        # 6. 결과 저장
        output = {
            'metadata': {
                'test_date': datetime.now().isoformat(),
                'source': 'ai.csv',
                'total_rows': len(df),
                'valid_rows': len(df_valid),
                'sample_size': len(df_sample),
                'total_tests': len(df_sample) * 2,
                'method': 'GraphRAGJudge (No LLM)',
                'threshold': 0.65,
                'llm_excluded': ['Phase 6 (의미 반전)', 'Phase 8 (설명)', '수정 제안']
            },
            'errors': {
                'unfair': unfair_errors,
                'fair': fair_errors
            },
            'confusion_matrix': {
                'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn
            },
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'confidence_averages': {
                'overall': avg_all,
                'unfair_cases': avg_unfair,
                'fair_cases': avg_fair,
                'difference': avg_unfair - avg_fair
            },
            'results': results
        }

        output_path = project_root / 'data' / 'test' / 'ai_csv_test_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*80}")
        print(f"💾 결과 저장: {output_path}")
        print(f"📄 전체 로그: {LOG_FILE}")
        print(f"{'='*80}")

    finally:
        # stdout 복원 및 로그 파일 닫기
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
