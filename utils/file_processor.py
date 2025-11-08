"""파일 처리 유틸리티"""
from typing import Optional
from pathlib import Path
import pypdf
from docx import Document


def extract_text_from_pdf(file_path: str) -> str:
    """PDF 파일에서 텍스트 추출"""
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """DOCX 파일에서 텍스트 추출"""
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def extract_text_from_txt(file_path: str) -> str:
    """TXT 파일에서 텍스트 읽기"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text_from_file(file_path: str, file_content: Optional[bytes] = None) -> str:
    """
    파일 경로 또는 파일 내용에서 텍스트 추출
    
    Args:
        file_path: 파일 경로 (확장자로 타입 판단)
        file_content: 파일 내용 (bytes, 선택적)
    
    Returns:
        추출된 텍스트
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if file_content:
        # 메모리에서 처리
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            if extension == ".pdf":
                text = extract_text_from_pdf(tmp_path)
            elif extension == ".docx":
                text = extract_text_from_docx(tmp_path)
            elif extension == ".txt":
                text = extract_text_from_txt(tmp_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {extension}")
        finally:
            os.unlink(tmp_path)
        
        return text
    else:
        # 파일 경로에서 직접 읽기
        if extension == ".pdf":
            return extract_text_from_pdf(file_path)
        elif extension == ".docx":
            return extract_text_from_docx(file_path)
        elif extension == ".txt":
            return extract_text_from_txt(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {extension}")



