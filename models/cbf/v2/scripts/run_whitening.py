import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import normalize

"""

안 쓸거 같음

movies_with_embeddings.pkl 파일에서 
모든 영화가 다 비슷해 보이는 문제(구조적 편향, Vector Space Collapse)를 해결하는 코드

코드의 핵심 역할 3가지
1. 중심점 이동 (Zero-centering):
    1.1 mu = vecs.mean(...), centered = vecs - mu
    1.2 현재 모든 영화 벡터가 우주의 한 구석(예: 좌표 100, 100 근처)에 몰려 있습니다.
    1.3 이 덩어리의 중심을 0, 0(원점)으로 끌고 옵니다.

2. 모양 펴기 (Decorrelation & Scaling):
    2.1 np.linalg.svd(cov), W = ...
    2.2 데이터가 길쭉한 타원형(특정 장르나 패턴에 치우침)으로 뭉쳐 있다면, 이를 동그란 구(Sphere) 모양으로 펴줍니다.
    2.3 수학적으로는 상관관계를 제거하고 분산을 1로 맞추는 과정입니다.

3. 검증 및 저장:
    3.1 Before와 After의 평균 유사도를 출력하여 수술이 성공했는지 바로 확인합니다.
    3.2 성공했다면(After 수치가 0에 가까우면) 그 결과를 pkl 파일로 저장하여, 이후 추천 시스템이 깨끗한 데이터를 쓰도록 합니다.

"""

# ============================================================
# Whitening 적용
# ============================================================
input_path = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl'
output_path = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_embeddings_whitened.pkl'

print("데이터 로드 중...")
df = pd.read_pickle(input_path)
embeddings = np.stack(df['embedding'].values)

print(f"Shape: {embeddings.shape}")

# ============================================================
# Whitening 함수 (제미나이 방식)
# ============================================================
def compute_whitening_transform(vecs):
    """Whitening 변환 행렬 계산"""
    print("Whitening 변환 행렬 계산 중...")
    
    mu = vecs.mean(axis=0, keepdims=True)
    centered = vecs - mu
    cov = np.cov(centered.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1.0 / np.sqrt(s + 1e-5)))
    
    return mu, W

def apply_whitening(vecs, mu, W):
    """Whitening 변환 적용"""
    centered = vecs - mu
    whitened = np.dot(centered, W)
    return normalize(whitened)

# ============================================================
# 실행
# ============================================================
mu, W = compute_whitening_transform(embeddings)
embeddings_whitened = apply_whitening(embeddings, mu, W)

# 검증
def get_avg_sim(vecs, n=2000):
    idx = np.random.choice(len(vecs), (n, 2), replace=True)
    sims = np.sum(vecs[idx[:, 0]] * vecs[idx[:, 1]], axis=1)
    return sims.mean()

print("\n" + "="*60)
print("Whitening 전후 비교")
print("="*60)
print(f"Before: {get_avg_sim(embeddings):.4f}")
print(f"After:  {get_avg_sim(embeddings_whitened):.4f}")

# 저장
df['embedding'] = list(embeddings_whitened)
df.to_pickle(output_path)

print(f"\n저장 완료: {output_path}")
print("\n다음 단계: 검증 스크립트로 재확인")