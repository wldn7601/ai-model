import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import ast
import os
import time

"""
multilingual-e5-large 모델을 사용한 영화 임베딩 생성

입력 형식:
- 태그 있음: "tags: {태그}. {overview}"
- 태그 없음: "overview}"
- overview가 10자 미만이면 + Title

편향 제거 X -> run_whitening.py 파일 실행 X

나오는 결과 :
전처리 데이터 + 임베딩된 결과
"""

"""
multilingual-e5-large 임베딩 생성 (안전장치 강화)
"""

# ============================================================
# 설정
# ============================================================
# 1. 입력 경로 (방금 전처리한 파일)
input_pkl = '/home/ubuntu/ai-model/models/cbf/v2/data/pre_final_movies_processed.pkl'

# 2. 출력 경로 (FAISS 인덱싱에서 사용할 최종 파일명)
output_pkl = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl'

# 3. 모델 설정
MODEL_NAME = 'intfloat/multilingual-e5-large'
BATCH_SIZE = 128    # T4 GPU 기준 적절 (OOM 발생 시 64로 줄이세요)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 4. 텍스트 구성 설정
TAG_SCORE_THRESHOLD = 0.5   # 0.5점 이상인 태그만 사용
MAX_TAGS = 15               # 태그 최대 개수

print("="*60)
print(f"임베딩 생성 시작: {DEVICE} 모드")
print(f"입력 파일: {input_pkl}")
print("="*60)

# ============================================================
# 1. 데이터 로드
# ============================================================
if not os.path.exists(input_pkl):
    print("❌ 오류: 입력 파일이 없습니다. 전처리(run_preprocess.py)를 먼저 실행하세요.")
    exit()

print(">>> 데이터 로드 중...")
df = pd.read_pickle(input_pkl)
print(f"전체 영화 수: {len(df):,}")

# ============================================================
# 2. 텍스트 전처리 (태그 파싱 & 입력 문구 생성)
# ============================================================
print(">>> 텍스트 전처리 중...")

def parse_and_create_text(row):
    # --- 1. 태그 파싱 ---
    tags_str = ""
    tag_data = row.get('tag_genome')
    
    # 데이터가 문자열이면 딕셔너리로 변환, 이미 딕셔너리면 그대로 사용
    if isinstance(tag_data, str):
        try:
            tag_data = ast.literal_eval(tag_data)
        except:
            tag_data = {}
    elif not isinstance(tag_data, dict):
        tag_data = {}

    # 유효한 태그 추출
    if tag_data:
        valid_tags = [
            t for t, s in sorted(tag_data.items(), key=lambda x: x[1], reverse=True)
            if s >= TAG_SCORE_THRESHOLD
        ]
        valid_tags = valid_tags[:MAX_TAGS]
        if valid_tags:
            tags_str = ", ".join(valid_tags)

    # --- 2. 텍스트 조합 (Clean Logic) ---
    overview = str(row['overview']).strip() if pd.notnull(row['overview']) else ""
    title = str(row.get('title', '')).strip() # 컬럼명 title 또는 title_ko 확인 필요
    if not title and 'title_ko' in row:
        title = str(row['title_ko']).strip()

    parts = []

    # (1) 태그
    if tags_str:
        parts.append(f"tags: {tags_str}")

    # (2) 제목 (줄거리가 너무 짧을 때 문맥 보강용)
    # overview가 10자 미만이면 제목을 넣음 (빈 문자열인 경우도 포함됨)
    if len(overview) < 10 and title:
        parts.append(f"title: {title}")

    # (3) 줄거리
    if overview:
        parts.append(overview)

    # 결과 반환
    return ". ".join(parts) if parts else "empty"

# 적용
df['text_for_embedding'] = df.apply(parse_and_create_text, axis=1)

# "empty"가 된 데이터 확인 (로그용)
empty_count = (df['text_for_embedding'] == "empty").sum()
if empty_count > 0:
    print(f"⚠️ 경고: 텍스트를 만들 수 없는 영화가 {empty_count}개 있습니다. (empty로 임베딩됨)")

# ============================================================
# 3. 모델 로드 및 임베딩 생성
# ============================================================
print(f"\n>>> 모델 로드 중 ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

if DEVICE == 'cuda':
    model.half() # FP16 가속
    print("✅ FP16 적용 완료")

# 입력 데이터 준비 ("passage: " 접두어 필수)
input_texts = ["passage: " + t for t in df['text_for_embedding'].tolist()]

print(f">>> 임베딩 생성 시작 ({len(input_texts):,}개)...")
start_time = time.time()

embeddings = model.encode(
    input_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True # 코사인 유사도용 정규화
)

end_time = time.time()
print(f">>> 생성 완료. 소요 시간: {end_time - start_time:.1f}초")

# ============================================================
# 4. 저장
# ============================================================
print("\n>>> 결과 저장 중...")

# 임베딩 컬럼 추가
df['embedding'] = list(embeddings)

# 임시 텍스트 컬럼 삭제 (용량 절약)
if 'text_for_embedding' in df.columns:
    df = df.drop(columns=['text_for_embedding'])

# 저장
df.to_pickle(output_pkl)

print("="*60)
print(f"✅ 최종 완료: {output_pkl}")
print(f"파일 크기: {os.path.getsize(output_pkl) / (1024**2):.2f} MB")
print("이제 FAISS 인덱싱 코드(run_indexing.py)를 실행하시면 됩니다.")
print("="*60)