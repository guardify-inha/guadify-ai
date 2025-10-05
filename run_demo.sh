#!/bin/bash

# Vector RAG 시스템 실행 스크립트

echo "🚀 Vector RAG 시스템 실행 시작"

# 가상환경 활성화
source venv/bin/activate

# 메모리 충돌 방지를 위한 환경 변수 설정
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# .env 파일에서 환경 변수 로드
if [ -f ".env" ]; then
    echo "📄 .env 파일에서 환경 변수를 로드합니다..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ 환경 변수 로드 완료"
fi

# 환경 변수 확인 및 설정
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY가 설정되지 않았습니다."
    echo "🔑 API 키를 설정하세요:"
    echo "1. 터미널에서: export OPENAI_API_KEY=your_api_key_here"
    echo "2. 또는 Python 스크립트로: python set_api_key.py"
    echo ""
    echo "OpenAI API 키는 https://platform.openai.com/api-keys 에서 발급받을 수 있습니다."
    exit 1
fi

echo "✅ 환경 변수 확인 완료"

# 1단계: 문서 전처리 (이미 완료된 경우 스킵)
if [ ! -f "outputs/chunks.jsonl" ]; then
    echo "📄 1단계: PDF 문서 전처리 중..."
    python scripts/extract_and_chunk.py
    echo "✅ 문서 전처리 완료"
else
    echo "✅ 문서 전처리 이미 완료됨 (chunks.jsonl 존재)"
fi

# 2단계: 벡터 임베딩 (이미 완료된 경우 스킵)
if [ ! -f "outputs/faiss.index" ]; then
    echo "🔢 2단계: 벡터 임베딩 중..."
    python scripts/embed_and_index.py
    echo "✅ 벡터 임베딩 완료"
else
    echo "✅ 벡터 임베딩 이미 완료됨 (faiss.index 존재)"
fi

# 3단계: 테스트 계약서 분석
echo "🔍 3단계: 테스트 계약서 분석 중..."
cd scripts
python query_and_extract.py --file ../data/contracts/user/test.txt --openai
cd ..

# 결과 폴더 찾기 (가장 최근 생성된 analysis_* 폴더)
LATEST_RESULT_DIR=$(ls -td outputs/analysis_* 2>/dev/null | head -1)

if [ -z "$LATEST_RESULT_DIR" ]; then
    echo "❌ 분석 결과 폴더를 찾을 수 없습니다."
    exit 1
fi

echo "📁 분석 결과 폴더: $LATEST_RESULT_DIR"

# 4단계: 최종 위험도 평가
echo "⚖️ 4단계: 최종 위험도 평가 중..."
cd scripts
python query_pipeline.py "../$LATEST_RESULT_DIR/query_results.json"
cd ..

echo "🎉 Vector RAG 시스템 실행 완료!"
echo "📊 결과 파일 위치:"
echo "  - $LATEST_RESULT_DIR/query_results.json (불공정 조항 후보)"
echo "  - $LATEST_RESULT_DIR/all_results.json (최종 위험도 평가)"
echo "  - $LATEST_RESULT_DIR/summary.txt (사람이 읽기 좋은 요약)"
