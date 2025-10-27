"""
VectorRAG 임베딩 생성 (메모리 최적화 버전)
- 모델: jhgan/ko-sroberta-multitask (가볍고 빠름!)
- 배치 크기 자동 조절
- 메모리 부족 시 자동 재시도

Inputs:  ../outputs/chunks.jsonl
Outputs: ../outputs/embeddings.npy
         ../outputs/faiss.index
         ../outputs/faiss_meta.pkl
"""

import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import gc

# 경로 설정
CHUNKS_FILE = os.path.join("..", "outputs", "chunks.jsonl")

# 가벼운 한국어 모델 (768차원, 빠름!)
EMBED_MODEL = "jhgan/ko-sroberta-multitask"

OUTPUT_DIR = os.path.join("..", "outputs")
EMB_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss.index")
META_FILE = os.path.join(OUTPUT_DIR, "faiss_meta.pkl")


def encode_with_memory_management(model, texts, initial_batch_size=32):
    """
    메모리 부족 시 배치 크기를 줄여가며 재시도
    """
    batch_size = initial_batch_size
    
    while batch_size >= 1:
        try:
            print(f"배치 크기 {batch_size}로 임베딩 생성 중...")
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device='cpu'
            )
            return embeddings
        
        except RuntimeError as e:
            if "out of memory" in str(e) or "not enough memory" in str(e):
                batch_size = batch_size // 2
                print(f"⚠️ 메모리 부족! 배치 크기를 {batch_size}로 줄입니다...")
                gc.collect()  # 가비지 컬렉션
                
                if batch_size < 1:
                    raise RuntimeError("배치 크기 1로도 메모리 부족. 모델을 더 작은 것으로 변경하세요.")
            else:
                raise e


def main():
    print(f"임베딩 모델 로드: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    
    # 청크 로드
    texts = []
    metadata = []
    
    print(f"청크 파일 로드: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="청크 로드 중"):
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            
            texts.append(text)
            metadata.append({
                "chunk_id": obj.get("chunk_id"),
                "source_file": obj.get("source_file"),
                "source_tag": obj.get("source_tag"),
                "level": obj.get("level"),
                "text": text,
                "metadata": obj.get("metadata", {})
            })
    
    print(f"총 {len(texts)}개 청크 로드 완료")
    
    # 임베딩 생성 (메모리 관리)
    embeddings = encode_with_memory_management(model, texts, initial_batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    
    print(f"임베딩 생성 완료: {embeddings.shape}")
    
    # 임베딩 저장
    np.save(EMB_FILE, embeddings)
    print(f"✅ 임베딩 저장: {EMB_FILE}")
    
    # FAISS 인덱스 생성
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ FAISS 인덱스 저장: {INDEX_FILE}")
    
    # 메타데이터 저장
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ 메타데이터 저장: {META_FILE}")
    
    print(f"\n🎉 완료! 총 {len(embeddings)}개 임베딩 생성")


if __name__ == "__main__":
    main()