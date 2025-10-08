#!/bin/bash

# AI 기반 불공정 약관 탐지 실행 스크립트

echo "🤖 AI 기반 불공정 약관 탐지 시스템 실행"

# --- 수정된 부분 시작 ---

# 가상환경 활성화 (운영체제 호환)
if [ -f "venv/Scripts/activate" ]; then
    # Windows (Git Bash)
    echo "🐍 Windows/Git Bash 환경을 감지했습니다. 가상 환경을 활성화합니다..."
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # macOS/Linux
    echo "🐍 macOS/Linux 환경을 감지했습니다. 가상 환경을 활성화합니다..."
    source venv/bin/activate
else
    echo "❌ 가상 환경 활성화 스크립트를 찾을 수 없습니다."
    echo "💡 venv/Scripts/activate 또는 venv/bin/activate 경로를 확인해주세요."
    exit 1
fi

# --- 수정된 부분 끝 ---


# 환경 변수 로드
if [ -f ".env" ]; then
    echo "📄 .env 파일에서 환경 변수를 로드합니다..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ 환경 변수 로드 완료"
fi

# 환경 변수 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY가 설정되지 않았습니다."
    echo "🔑 .env 파일에 API 키를 설정하세요"
    exit 1
fi

echo "✅ 환경 변수 확인 완료"

# 기본 입력 파일 설정
INPUT_FILE="test_inputs/sample_contract.txt"

# 명령행 인수로 입력 파일 지정 가능
if [ $# -gt 0 ]; then
    INPUT_FILE="$1"
fi

echo "📄 입력 파일: $INPUT_FILE"

# 입력 파일 존재 확인
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 입력 파일을 찾을 수 없습니다: $INPUT_FILE"
    echo "💡 사용 가능한 테스트 파일들:"
    ls -la test_inputs/
    exit 1
fi

echo "🚀 AI 기반 불공정 약관 탐지 시작..."

# AI 탐지 실행 (Windows 환경에서 인코딩 문제 해결)
if [ -f "venv/Scripts/activate" ]; then
    # Windows 환경
    python -X utf8 scripts/ai_unfair_detector.py --file "$INPUT_FILE" --output results
else
    # macOS/Linux 환경
    python scripts/ai_unfair_detector.py --file "$INPUT_FILE" --output results
fi

echo "🎉 분석 완료!"
echo "📁 결과는 results/analysis_YYYYMMDD_HHMMSS/ 폴더에 저장되었습니다."
echo "   📄 ai_detection_result.json - 원본 JSON 결과"
echo "   📋 analysis_report.md - 사람이 읽기 쉬운 마크다운 보고서"