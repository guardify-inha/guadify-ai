# scripts/extract_and_chunk.py
import os, json, uuid
import fitz  # pymupdf
from tqdm import tqdm

DATA_DIR = "../data/contracts/reference"
OUT_FILE = "../outputs/chunks.jsonl"
CHUNK_SIZE = 1200    # 문자 기준 (약 250~600 tokens)
CHUNK_OVERLAP = 200
MAX_PAGE_CHARS = 200_000   # 페이지당 최대 글자 수 제한 (필요시 조정)

def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end]
        chunks.append(chunk.strip())

        new_start = end - overlap
        # 무한루프 방지: overlap이 크거나 데이터가 이상한 경우
        if new_start <= start:
            new_start = start + size
        start = new_start
    return chunks

os.makedirs("../outputs", exist_ok=True)

with open(OUT_FILE, "w", encoding="utf-8") as outf:
    for fname in tqdm(os.listdir(DATA_DIR), desc="Processing PDFs"):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_DIR, fname)
        doc = fitz.open(path)

        for page_num, page in enumerate(doc, start=1):
            txt = page.get_text()
            if not txt:
                continue

            # 너무 큰 페이지는 잘라내거나 skip
            if len(txt) > MAX_PAGE_CHARS:
                print(f"⚠️ Skip or truncate large page {page_num} in {fname} ({len(txt)} chars)")
                txt = txt[:MAX_PAGE_CHARS]

            chunks = split_text(txt)
            for i, c in enumerate(chunks):
                rec = {
                    "chunk_id": str(uuid.uuid4()),
                    "source_file": fname,
                    "page": page_num,
                    "chunk_idx": i,
                    "text": c
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("✅ Done. Chunks written to", OUT_FILE)
