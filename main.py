"""FastAPI 메인 애플리케이션"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from chains.full_analysis import analyze_contract_async
from utils.file_processor import extract_text_from_file

app = FastAPI(
    title="계약서 불공정 약관 분석 API",
    description="RAG/LangChain 기반 계약서 불공정 약관 분석 서비스",
    version="1.0.0"
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청/응답 모델
class TextAnalysisRequest(BaseModel):
    """텍스트 분석 요청"""
    text: str


class AnalysisResponse(BaseModel):
    """분석 결과 응답"""
    overall_risk_assessment: str
    summary: str
    clauses: list[dict]


@app.get("/")
async def root():
    """루트 엔드포인트 - 웹 UI 제공"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_contract(
    text: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """
    계약서 분석 엔드포인트
    
    두 가지 방식으로 사용 가능:
    1. 텍스트 직접 전송: JSON body에 "text" 필드 포함
    2. 파일 업로드: multipart/form-data로 파일 업로드
    
    지원 파일 형식: .txt, .pdf, .docx
    """
    contract_text = None
    
    # 텍스트 직접 전송
    if text:
        contract_text = text
    
    # 파일 업로드
    elif file:
        # 파일 확장자 확인
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".txt", ".pdf", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식: {file_extension}. 지원 형식: .txt, .pdf, .docx"
            )
        
        # 임시 파일로 저장
        file_content = await file.read()
        
        try:
            # 파일에서 텍스트 추출
            contract_text = extract_text_from_file(
                file_path=file.filename,
                file_content=file_content
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"파일 처리 중 오류 발생: {str(e)}"
            )
    
    else:
        raise HTTPException(
            status_code=400,
            detail="'text' 필드 또는 'file' 파일 중 하나는 필수입니다."
        )
    
    if not contract_text or not contract_text.strip():
        raise HTTPException(
            status_code=400,
            detail="계약서 텍스트가 비어있습니다."
        )
    
    try:
        # 계약서 분석 실행
        result = await analyze_contract_async(contract_text)
        
        return AnalysisResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류 발생: {str(e)}"
        )


@app.post("/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    """
    텍스트 분석 엔드포인트 (JSON body)
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="텍스트가 비어있습니다."
        )
    
    try:
        result = await analyze_contract_async(request.text)
        return AnalysisResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류 발생: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

