import pandas as pd
import numpy as np
import faiss
import pickle
import os

"""
[Step 3] FAISS 인덱싱 및 ID 매핑 생성
입력: movies_with_embeddings.pkl
출력: 
  1. movies.faiss (벡터 인덱스)
  2. movie_ids.pkl (순서대로 정렬된 tmdbId 리스트)
"""

# ============================================================
# 설정
# ============================================================
input_pkl = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl'
index_dir = '/home/ubuntu/ai-model/models/cbf/v2/index'

index_path = os.path.join(index_dir, 'movies.faiss')
mapping_path = os.path.join(index_dir, 'movie_ids.pkl')

# 디렉토리 생성
os.makedirs(index_dir, exist_ok=True)

print("="*60)
print("FAISS 인덱싱 작업 시작")
print("="*60)

# ============================================================
# 1. 데이터 로드 및 정렬 확인
# ============================================================
if not os.path.exists(input_pkl):
    print("❌ 오류: 임베딩 파일이 없습니다. (create_embeddings.py 먼저 실행)")
    exit()

print(">>> 데이터 로딩 중...")
df = pd.read_pickle(input_pkl)

# [중요] 인덱스 리셋: DataFrame의 행 번호와 FAISS의 ID(0,1,2...)를 1:1로 맞춤
df = df.reset_index(drop=True)

# 컬럼 확인 (tmdbId 사용)
target_id_col = 'tmdbId'

if target_id_col not in df.columns:
    print(f"❌ 오류: '{target_id_col}' 컬럼이 없습니다.")
    print(f"현재 컬럼 목록: {list(df.columns)}")
    exit()

print(f"전체 영화 수: {len(df):,}")
print(f"매핑 ID 컬럼: {target_id_col}")

# ============================================================
# 2. 임베딩 추출 및 정규화
# ============================================================
print("\n>>> 임베딩 추출 및 변환 중...")

# numpy 배열로 변환 (float32 필수)
embeddings = np.stack(df['embedding'].values).astype('float32')

# [안전장치] L2 정규화 (Inner Product를 코사인 유사도로 사용하기 위함)
# 임베딩 생성 시 normalize=True를 했지만, 여기서 한 번 더 확실하게 처리
faiss.normalize_L2(embeddings)

print(f"임베딩 Shape: {embeddings.shape}")

# ============================================================
# 3. FAISS 인덱스 생성 (GPU/CPU 자동 감지)
# ============================================================
print("\n>>> FAISS 인덱스 생성 중...")
dimension = embeddings.shape[1]

# GPU 사용 가능 여부 확인
if faiss.get_num_gpus() > 0:
    print(f"🚀 GPU 모드 활성화 (GPU 개수: {faiss.get_num_gpus()})")
    
    # GPU 리소스 초기화
    res = faiss.StandardGpuResources()
    
    # 인덱스 설정 (Flat Inner Product)
    index_config = faiss.IndexFlatIP(dimension)
    
    # GPU로 데이터 이동 및 추가
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_config)
    gpu_index.add(embeddings)
    
    # 저장을 위해 CPU로 다시 가져옴
    index = faiss.index_gpu_to_cpu(gpu_index)
else:
    print("🐢 CPU 모드 활성화")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

# ============================================================
# 4. 저장 (인덱스 + 매핑 파일)
# ============================================================
print("\n>>> 파일 저장 중...")

# 1. FAISS 인덱스 파일 저장
faiss.write_index(index, index_path)

# 2. ID 매핑 파일 저장 (tmdbId 리스트)
# FAISS 검색 결과(0, 1, 2...)를 실제 tmdbId(12345...)로 바꾸기 위함
with open(mapping_path, 'wb') as f:
    pickle.dump(df[target_id_col].tolist(), f)

print("="*60)
print(f"✅ 완료: {index.ntotal}개 벡터 인덱싱됨")
print(f"인덱스 파일: {index_path}")
print(f"매핑 파일  : {mapping_path}")
print("="*60)