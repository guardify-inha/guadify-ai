"""
개선사항:
1. 조문 경계 인식 강화 - 줄 시작 패턴으로 본문 내 조문 참조 무시
2. 문장 완성도 개선 - 항/호 단위로 완전한 문장 구성
3. 띄어쓰기 복원 개선 - PyKoSpacing 기반 (fallback: 규칙 기반)
4. 메타데이터 강화 - GraphRAG용 의미 라벨 추가

Outputs:
  1. ../outputs/chunks.jsonl - VectorRAG용 임베딩 대상
"""

import os
import json
import uuid
import re
import csv
import fitz  # PyMuPDF
from tqdm import tqdm

# 띄어쓰기 복원 라이브러리 (선택적)
try:
    from pykospacing import Spacing
    SPACING_AVAILABLE = True
    spacing = Spacing()
except ImportError:
    SPACING_AVAILABLE = False
    print("⚠️ pykospacing 미설치 - 규칙 기반 띄어쓰기 복원 사용")


# -------------------- 경로 설정 --------------------
DATA_ROOT = os.path.join("..", "data")
CONTRACTS_REF_DIR = os.path.join(DATA_ROOT, "contracts", "reference")
LAW_DIR = os.path.join(DATA_ROOT, "law")
STANDARD_DIR = os.path.join(DATA_ROOT, "standard")

OUT_DIR = os.path.join("..", "outputs")
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.jsonl")


# -------------------- 정규식 패턴 (개선) --------------------
# 🔥 핵심 개선: 줄 시작(^)에 있는 조문만 인식
# 본문에 "제16조 제1항" 같은 참조가 있어도 무시됨
ARTICLE_RE = re.compile(r"^제\s*\d+(?:의\d+)?\s*조(?:\s*\([^)]+\))?", re.MULTILINE)

# 관(款): 제X관 (중간 제목) - 줄 시작에만
SECTION_RE = re.compile(r"^제\s*\d+\s*관\s+[^\n]+", re.MULTILINE)

# 장: 제X장 - 줄 시작에만
CHAPTER_RE = re.compile(r"^제\s*\d+\s*장(?:\s+[^\n]+)?", re.MULTILINE)

# 부칙 패턴
APPENDIX_RE = re.compile(r"부\s*칙\s*<[^>]+>", re.MULTILINE)

# 항: ① ② ③ 등 원문자
HANG_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕]", re.MULTILINE)

# 호: 1. 2. 3. (줄 시작)
NUMBER_LIST_RE = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)


# -------------------- 띄어쓰기 복원 (개선) --------------------
def restore_spacing_advanced(text):
    """
    향상된 띄어쓰기 복원
    1순위: PyKoSpacing (있으면)
    2순위: 규칙 기반
    """
    if not text or len(text.strip()) < 3:
        return text
    
    # PyKoSpacing 사용 가능하면 사용
    if SPACING_AVAILABLE:
        try:
            # 너무 긴 텍스트는 나눠서 처리
            if len(text) > 500:
                sentences = re.split(r'([.!?]\s+)', text)
                spaced_sentences = []
                for sent in sentences:
                    if len(sent.strip()) > 2:
                        try:
                            spaced = spacing(sent)
                            spaced_sentences.append(spaced)
                        except:
                            spaced_sentences.append(sent)
                    else:
                        spaced_sentences.append(sent)
                return ''.join(spaced_sentences)
            else:
                return spacing(text)
        except Exception as e:
            # PyKoSpacing 실패 시 규칙 기반으로 폴백
            pass
    
    # 규칙 기반 띄어쓰기 복원
    text = restore_spacing_rule_based(text)
    return text


def restore_spacing_rule_based(text):
    """
    규칙 기반 띄어쓰기 복원 (개선)
    """
    if not text:
        return text
    
    # 1. 기본 정제
    text = re.sub(r'\s+', ' ', text)  # 연속 공백 제거
    
    # 2. 조사/어미 앞 띄어쓰기
    # "합니다", "습니다", "입니다" 등
    text = re.sub(r'([가-힣])(입니다|습니다|합니다|됩니다|있습니다|없습니다|하여야|하여서)', r'\1 \2', text)
    
    # "의", "에", "을", "를" 등 조사
    text = re.sub(r'([가-힣]{2,})(의|에|을|를|와|과|도|만|부터|까지|에서|으로|로써)', r'\1 \2', text)
    
    # 3. 숫자 뒤 단위
    text = re.sub(r'(\d+)(개월|년|일|원|건|명|개)', r'\1\2', text)
    
    # 4. 괄호 주변 공백 정리
    text = re.sub(r'\s*\(\s*', '(', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    
    # 5. 마침표, 쉼표 뒤 공백
    text = re.sub(r'([.!?])\s*([가-힣])', r'\1 \2', text)
    text = re.sub(r'([,])\s*([가-힣])', r'\1 \2', text)
    
    # 6. 연속 공백 다시 제거
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# -------------------- 텍스트 정제 (개선) --------------------
def clean_text_advanced(text):
    """
    PDF 추출 텍스트 정제 (개선)
    """
    if not text:
        return ""
    
    # 1. 줄바꿈 정규화
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    
    # 2. 조문 제목 내 불필요한 줄바꿈 제거
    text = re.sub(r'(제\s*\d+(?:의\d+)?)\s*\n\s*(조)', r'\1\2', text)
    text = re.sub(r'(제\s*\d+(?:의\d+)?\s*조)\s*\n\s*(\([^)]+\))', r'\1\2', text)
    
    # 3. 단어 중간 줄바꿈 제거 (하지만 조문 시작은 보존)
    # 조문이 아닌 일반 텍스트만 처리
    lines = text.split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        # 줄 시작이 조문 패턴이면 보존
        if re.match(r'^\s*제\s*\d+(?:의\d+)?\s*조', line):
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    # 4. 숫자 앞뒤 줄바꿈 정리 (단, 호 번호는 제외)
    text = re.sub(r'([가-힣])\s*\n\s*(\d+)\s*\n\s*([가-힣])', r'\1 \2 \3', text)
    
    # 5. 여러 줄바꿈을 2개로
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 6. 연속 공백 제거 (단, 줄바꿈은 보존)
    lines = text.split('\n')
    cleaned_lines = [re.sub(r' +', ' ', line) for line in lines]
    text = '\n'.join(cleaned_lines)
    
    return text.strip()


def remove_section_titles(text):
    """관(款) 제목 제거"""
    text = SECTION_RE.sub('', text)
    return text.strip()


# -------------------- 조문 분리 (대폭 개선) --------------------
def split_by_article_robust(text, is_standard=False):
    """
    강화된 조문 분리 로직 (v8 - 줄 시작 패턴 적용)
    - 줄 시작에 있는 조문만 인식 (본문 내 조문 참조 무시)
    - 위치 기반 정확한 경계 인식
    """
    text = clean_text_advanced(text)
    
    # 모든 조문 매치 찾기 (위치 정보 포함)
    article_matches = []
    for match in ARTICLE_RE.finditer(text):
        article_matches.append({
            'match': match,
            'title': match.group(0).strip(),
            'start': match.start(),
            'end': match.end()
        })
    
    if not article_matches:
        return []
    
    results = []
    
    for i, art_info in enumerate(article_matches):
        title = art_info['title']
        body_start = art_info['end']
        
        # 다음 조문까지의 본문 추출
        if i + 1 < len(article_matches):
            body_end = article_matches[i + 1]['start']
        else:
            # 마지막 조문 - 부칙 전까지
            appendix_matches = list(APPENDIX_RE.finditer(text[body_start:]))
            if appendix_matches:
                body_end = body_start + appendix_matches[0].start()
            else:
                body_end = len(text)
        
        body = text[body_start:body_end].strip()
        
        # 관(款) 제목 제거
        body = remove_section_titles(body)
        
        # 페이지 번호 제거 (- 2 -, - 3 - 같은 패턴)
        body = re.sub(r'-\s*\d+\s*-', '', body)
        
        # 최소 길이 필터
        if body and len(body) > 20:
            results.append({
                "title": title,
                "body": body,
                "is_appendix": False,
                "position": art_info['start']
            })
    
    # 부칙 처리
    for match in APPENDIX_RE.finditer(text):
        title = match.group(0).strip()
        body_start = match.end()
        
        # 부칙 본문 추출 (다음 조문 전까지)
        remaining_text = text[body_start:]
        next_article = ARTICLE_RE.search(remaining_text)
        
        if next_article:
            body = remaining_text[:next_article.start()].strip()
        else:
            body = remaining_text.strip()
        
        # 페이지 번호 제거
        body = re.sub(r'-\s*\d+\s*-', '', body)
        
        if body and len(body) > 20:
            results.append({
                "title": title,
                "body": body,
                "is_appendix": True,
                "position": match.start()
            })
    
    return results


# -------------------- 항/호 분리 (개선) --------------------
def split_into_hang_and_ho_advanced(text):
    """
    항(①②③)과 호(1. 2. 3.)로 분리 - 완전한 문장 보장
    """
    if not text:
        return []
    
    chunks = []
    
    # 항(①②③) 기준 분리
    hang_matches = list(HANG_RE.finditer(text))
    
    if hang_matches:
        for i, match in enumerate(hang_matches):
            start = match.start()
            
            # 다음 항까지
            if i + 1 < len(hang_matches):
                end = hang_matches[i + 1].start()
            else:
                end = len(text)
            
            hang_text = text[start:end].strip()
            
            # 페이지 번호 제거
            hang_text = re.sub(r'-\s*\d+\s*-', '', hang_text)
            
            # 호(1. 2. 3.) 기준으로 추가 분리
            ho_chunks = split_by_ho(hang_text)
            
            if ho_chunks:
                chunks.extend(ho_chunks)
            elif len(hang_text) > 15:
                chunks.append(hang_text)
    else:
        # 항이 없으면 호만으로 분리
        ho_chunks = split_by_ho(text)
        if ho_chunks:
            chunks.extend(ho_chunks)
        elif len(text) > 15:
            # 페이지 번호 제거
            text = re.sub(r'-\s*\d+\s*-', '', text)
            chunks.append(text)
    
    # 띄어쓰기 복원 적용
    restored_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) > 15:
            restored = restore_spacing_advanced(chunk)
            restored_chunks.append(restored)
    
    return restored_chunks


def split_by_ho(text):
    """
    호(1. 2. 3.) 기준 분리 - 완전한 문장 보장
    """
    if not text:
        return []
    
    # 호 패턴 찾기
    ho_matches = list(NUMBER_LIST_RE.finditer(text))
    
    if not ho_matches:
        return [text] if len(text) > 15 else []
    
    chunks = []
    
    # 첫 호 이전 텍스트
    before_first = text[:ho_matches[0].start()].strip()
    if len(before_first) > 15:
        chunks.append(before_first)
    
    # 각 호별로 분리
    for i, match in enumerate(ho_matches):
        start = match.start()
        
        if i + 1 < len(ho_matches):
            end = ho_matches[i + 1].start()
        else:
            end = len(text)
        
        ho_text = text[start:end].strip()
        
        # 페이지 번호 제거
        ho_text = re.sub(r'-\s*\d+\s*-', '', ho_text)
        
        if len(ho_text) > 15:
            chunks.append(ho_text)
    
    return chunks


# -------------------- 메타데이터 추출 (개선) --------------------
def extract_article_number(article_title):
    """조문 번호 추출: '제12조 (계약 전 알릴 의무)' -> '12'"""
    if not article_title:
        return None
    match = re.search(r"제\s*(\d+(?:의\d+)?)\s*조", article_title)
    return match.group(1) if match else None


def extract_article_title_detail(article_title):
    """조문 제목 상세 추출: '제12조 (계약 전 알릴 의무)' -> '계약 전 알릴 의무'"""
    if not article_title:
        return None
    match = re.search(r"\(([^)]+)\)", article_title)
    return match.group(1) if match else None


def extract_chapter_info(text, position):
    """장(章) 정보 추출 - 개선"""
    if position < 0 or position > len(text):
        return None, None
    
    before_text = text[:position]
    chapter_matches = list(CHAPTER_RE.finditer(before_text))
    
    if not chapter_matches:
        return None, None
    
    last_match = chapter_matches[-1]
    full_chapter = last_match.group(0).strip()
    
    # "제1장 총칙" 형태 파싱
    chapter_split = re.match(r"제\s*(\d+)\s*장(?:\s+(.+))?", full_chapter)
    if chapter_split:
        chapter_num = chapter_split.group(1)
        chapter_title = chapter_split.group(2).strip() if chapter_split.group(2) else None
        return chapter_num, chapter_title
    
    return None, None


def extract_section_info(text, position):
    """관(款) 정보 추출 - 개선"""
    if position < 0 or position > len(text):
        return None, None
    
    before_text = text[:position]
    section_matches = list(SECTION_RE.finditer(before_text))
    
    if not section_matches:
        return None, None
    
    last_match = section_matches[-1]
    full_section = last_match.group(0).strip()
    
    # "제1관 계약의 성립" 형태 파싱
    section_split = re.match(r"제\s*(\d+)\s*관\s+(.+)", full_section)
    if section_split:
        section_num = section_split.group(1)
        section_title = section_split.group(2).strip()
        return section_num, section_title
    
    return None, None


def infer_semantic_label(article_title_detail, chapter_title, section_title):
    """
    GraphRAG용 의미 라벨 추론
    조문 제목, 장 제목, 관 제목에서 대주제/소주제 추출
    """
    labels = []
    
    if chapter_title:
        labels.append(("대주제", chapter_title))
    
    if section_title:
        labels.append(("중주제", section_title))
    
    if article_title_detail:
        labels.append(("소주제", article_title_detail))
    
    return labels


# -------------------- CSV 기반 Reference 처리 (개선) --------------------
def process_reference_csv(csv_path):
    """CSV에서 불공정 약관 사례를 읽어서 청크 생성"""
    chunks = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            case_id = row.get("ID", "").strip()
            if not case_id:
                case_id = str(uuid.uuid4())
            
            chunk_id = str(uuid.uuid4())
            
            파일명 = row.get("파일명", "").strip()
            대주제 = row.get("대주제", "").strip()
            소주제 = row.get("소주제", "").strip()
            불공정약관원문 = row.get("불공정 약관 원문", "").strip()
            시정요청사유 = row.get("시정 요청 사유", "").strip()
            근거조항 = row.get("근거 조항", "").strip()
            수정후약관 = row.get("수정 후 약관 조항", "").strip()
            
            # VectorRAG용 통합 텍스트
            combined_text = f"""[대주제: {대주제}]
[소주제: {소주제}]

불공정 약관 원문:
{불공정약관원문}

시정 요청 사유:
{시정요청사유}

근거 조항:
{근거조항}"""

            if 수정후약관:
                combined_text += f"""

수정 후 약관:
{수정후약관}"""
            
            chunk = {
                "chunk_id": chunk_id,
                "source_file": 파일명,
                "source_tag": "reference",
                "level": "unfair_case",
                "text": combined_text,
                "metadata": {
                    "case_id": case_id,
                    "파일명": 파일명,
                    "대주제": 대주제,
                    "소주제": 소주제,
                    "근거조항": 근거조항,
                    # GraphRAG용 라벨
                    "semantic_labels": [
                        ("대주제", 대주제),
                        ("소주제", 소주제)
                    ]
                }
            }
            chunks.append(chunk)
    
    return chunks


# -------------------- PDF 법률 처리 (개선) --------------------
def process_law_pdf(path):
    """법률 PDF를 조문 -> 항/호 단위로 분리"""
    doc = fitz.open(path)
    fname = os.path.basename(path)
    
    full_text = "\n".join([p.get_text() for p in doc])
    if not full_text.strip():
        print(f"⚠️ {fname}: PDF에서 텍스트를 추출하지 못했습니다.")
        return []
    
    chunks = []
    articles = split_by_article_robust(full_text, is_standard=False)
    
    for art in articles:
        article_title = art.get("title", "")
        article_body = art.get("body", "")
        is_appendix = art.get("is_appendix", False)
        position = art.get("position", 0)
        
        if not article_title:
            continue
        
        article_num = extract_article_number(article_title)
        article_detail = extract_article_title_detail(article_title)
        
        if not article_num:
            continue
        
        # 장/관 정보 추출
        chapter_num, chapter_title = extract_chapter_info(full_text, position)
        section_num, section_title = extract_section_info(full_text, position)
        
        # 항/호 단위로 분리
        sentences = split_into_hang_and_ho_advanced(article_body)
        
        if not sentences:
            continue
        
        # 의미 라벨 추론
        semantic_labels = infer_semantic_label(article_detail, chapter_title, section_title)
        
        for idx, sent in enumerate(sentences):
            # 하위조항 식별
            hang_match = re.match(r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕])", sent.strip())
            ho_match = re.match(r"^(\d+\.)", sent.strip())
            
            하위조항 = None
            if hang_match:
                하위조항 = hang_match.group(1)
            elif ho_match:
                하위조항 = ho_match.group(1)
            
            chunk_id = str(uuid.uuid4())
            
            chunk = {
                "chunk_id": chunk_id,
                "source_file": fname,
                "source_tag": "law",
                "level": "law_appendix" if is_appendix else "law_clause",
                "text": sent,
                "metadata": {
                    "법령명": fname.replace(".pdf", ""),
                    "장": f"제{chapter_num}장" if chapter_num else None,
                    "장_제목": chapter_title,
                    "관": f"제{section_num}관" if section_num else None,
                    "관_제목": section_title,
                    "조문번호": article_num,
                    "조문제목": article_title,
                    "조문제목_상세": article_detail,
                    "항_호_순번": idx + 1,
                    "하위조항": 하위조항,
                    "부칙여부": is_appendix,
                    # GraphRAG용
                    "semantic_labels": semantic_labels
                }
            }
            chunks.append(chunk)
    
    return chunks


# -------------------- PDF 표준약관 처리 (개선) --------------------
def process_standard_pdf(path):
    """표준약관 PDF를 항 단위로 분리 - v8 개선 버전"""
    doc = fitz.open(path)
    fname = os.path.basename(path)
    
    full_text = "\n".join([p.get_text() for p in doc])
    if not full_text.strip():
        print(f"⚠️ {fname}: PDF에서 텍스트를 추출하지 못했습니다.")
        return []
    
    chunks = []
    articles = split_by_article_robust(full_text, is_standard=True)
    print(f"📄 {fname}: {len(articles)}개 조문 발견")
    
    for art in articles:
        article_title = art.get("title", "")
        article_body = art.get("body", "")
        is_appendix = art.get("is_appendix", False)
        position = art.get("position", 0)
        
        if not article_title:
            continue
        
        article_num = extract_article_number(article_title)
        article_detail = extract_article_title_detail(article_title)
        
        if not article_num:
            continue
        
        # 장/관 정보 추출
        chapter_num, chapter_title = extract_chapter_info(full_text, position)
        section_num, section_title = extract_section_info(full_text, position)
        
        # 항/호 단위로 분리
        sentences = split_into_hang_and_ho_advanced(article_body)
        
        if not sentences:
            continue
        
        # 의미 라벨 추론
        semantic_labels = infer_semantic_label(article_detail, chapter_title, section_title)
        
        for idx, sent in enumerate(sentences):
            # 하위조항 식별
            hang_match = re.match(r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕])", sent.strip())
            ho_match = re.match(r"^(\d+\.)", sent.strip())
            
            하위조항 = None
            if hang_match:
                하위조항 = hang_match.group(1)
            elif ho_match:
                하위조항 = ho_match.group(1)
            
            chunk_id = str(uuid.uuid4())
            
            chunk = {
                "chunk_id": chunk_id,
                "source_file": fname,
                "source_tag": "standard",
                "level": "standard_appendix" if is_appendix else "standard_clause",
                "text": sent,
                "metadata": {
                    "표준약관명": fname.replace(".pdf", ""),
                    "장": f"제{chapter_num}장" if chapter_num else None,
                    "장_제목": chapter_title,
                    "관": f"제{section_num}관" if section_num else None,
                    "관_제목": section_title,
                    "조문번호": article_num,
                    "조문제목": article_title,
                    "조문제목_상세": article_detail,
                    "항_호_순번": idx + 1,
                    "하위조항": 하위조항,
                    "부칙여부": is_appendix,
                    # GraphRAG용
                    "semantic_labels": semantic_labels
                }
            }
            chunks.append(chunk)
    
    return chunks


# -------------------- 메인 --------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    file_list = []
    
    # CSV reference
    if os.path.isdir(CONTRACTS_REF_DIR):
        for f in os.listdir(CONTRACTS_REF_DIR):
            if f.lower().endswith(".csv"):
                file_list.append((os.path.join(CONTRACTS_REF_DIR, f), "reference"))
    
    # PDF law / standard
    for d, tag in [(LAW_DIR, "law"), (STANDARD_DIR, "standard")]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".pdf"):
                    file_list.append((os.path.join(d, f), tag))
    
    if not file_list:
        print("❌ 처리할 파일이 없습니다.")
        return
    
    # 기존 출력 파일 삭제
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)
        print(f"🗑️  기존 파일 삭제: {CHUNKS_FILE}")
    
    all_chunks = []
    
    print(f"📦 처리 대상 파일: {len(file_list)}개")
    print(f"{'='*60}")
    
    for path, tag in tqdm(file_list, desc="Processing Files"):
        try:
            if tag == "reference":
                chunks = process_reference_csv(path)
            elif tag == "law":
                chunks = process_law_pdf(path)
            elif tag == "standard":
                chunks = process_standard_pdf(path)
            else:
                continue
            
            all_chunks.extend(chunks)
            print(f"✓ {os.path.basename(path)}: {len(chunks)}개 청크 생성")
            
        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")
            import traceback
            traceback.print_exc()
    
    # 파일 저장
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"✅ 완료!")
    print(f"  📝 총 청크: {len(all_chunks)}개")
    print(f"  💾 출력: {CHUNKS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()