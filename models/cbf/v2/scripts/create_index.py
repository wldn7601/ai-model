import pandas as pd
import numpy as np
import faiss
import pickle
import os

input_pkl = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl'
index_dir = '/home/ubuntu/ai-model/models/cbf/v2/index'
index_path = os.path.join(index_dir, 'movies.faiss')
mapping_path = os.path.join(index_dir, 'movie_ids.pkl')  # 안전장치

os.makedirs(index_dir, exist_ok=True)

print("데이터 로딩...")
df = pd.read_pickle(input_pkl)
df = df.reset_index(drop=True)  # 순서 보장

print("임베딩 추출...")
embeddings = np.stack(df['embedding'].values).astype('float32')

print("FAISS 인덱스 생성...")
dimension = embeddings.shape[1]

if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(dimension)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(embeddings)
    index = faiss.index_gpu_to_cpu(gpu_index)
else:
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

print("저장 중...")
faiss.write_index(index, index_path)

# [선택] 안전장치: tmdb_id 매핑 저장
with open(mapping_path, 'wb') as f:
    pickle.dump(df['tmdb_id'].tolist(), f)

print(f"✅ 완료: {index.ntotal}개 벡터 인덱싱")