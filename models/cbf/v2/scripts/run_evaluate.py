import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# ============================================================
# 설정
# ============================================================

# run_whitening.py 실행 전 기존 모델 경로
input_path = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl'

# run_whitening.py 실행 후 보완 모델 경로
# input_path = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_embeddings_whitened.pkl'

print("="*60)
print("임베딩 품질 검증")
print("="*60)

# ============================================================
# 1. 데이터 로드
# ============================================================
if not os.path.exists(input_path):
    print("Error: 파일이 없습니다.")
    exit()

print("\n>>> 데이터 로드 중...")
df = pd.read_pickle(input_path)

# 임베딩 행렬 변환
embedding_matrix = np.stack(df['embedding'].values)

print(f"전체 영화 수: {len(df):,}")
print(f"임베딩 Shape: {embedding_matrix.shape}")
print(f"임베딩 차원: {embedding_matrix.shape[1]}")

# ============================================================
# 2. 구조적 편향 검증 (제미나이 방식)
# ============================================================
print("\n" + "="*60)
print("[검증 1] 랜덤 쌍 유사도 분포")
print("="*60)

n_samples = 2000
idx_a = np.random.randint(0, len(df), n_samples)
idx_b = np.random.randint(0, len(df), n_samples)

# 자기 자신 제외
mask = idx_a != idx_b
idx_a, idx_b = idx_a[mask], idx_b[mask]

# 유사도 계산
sims = np.sum(embedding_matrix[idx_a] * embedding_matrix[idx_b], axis=1)

print(f"평균 유사도: {sims.mean():.4f}")
print(f"표준편차: {sims.std():.4f}")
print(f"최소: {sims.min():.4f}")
print(f"최대: {sims.max():.4f}")

# 판단
if sims.mean() > 0.8:
    print("\n🔴 진단: 구조적 편향 심각")
    print("→ Whitening 필수")
    bias_status = "CRITICAL"
elif sims.mean() > 0.7:
    print("\n🟡 진단: 편향 존재")
    print("→ Whitening 권장")
    bias_status = "WARNING"
else:
    print("\n🟢 진단: 정상 범위")
    bias_status = "OK"

# ============================================================
# 3. 장르별 유사도 검증 (추가)
# ============================================================
print("\n" + "="*60)
print("[검증 2] 장르별 유사도 분포")
print("="*60)

def check_genre_similarity(genre_name, sample_size=100):
    """특정 장르 내 영화들의 평균 유사도"""
    genre_movies = df[df['genres'].str.contains(genre_name, na=False)]
    
    if len(genre_movies) < 10:
        print(f"{genre_name}: 샘플 부족")
        return None
    
    # 샘플링
    sample = genre_movies.sample(min(sample_size, len(genre_movies)))
    sample_embs = embedding_matrix[sample.index]
    
    # 유사도 행렬
    sim_matrix = cosine_similarity(sample_embs)
    
    # 자기 자신 제외
    mask = ~np.eye(len(sample_embs), dtype=bool)
    avg_sim = sim_matrix[mask].mean()
    
    print(f"{genre_name:10s}: 평균 {avg_sim:.4f} | 영화 수 {len(genre_movies):,}")
    return avg_sim

# 주요 장르 검증
genres_to_check = ['액션', '로맨스', '코미디', '공포', '드라마', 'SF']
genre_sims = {}

for genre in genres_to_check:
    sim = check_genre_similarity(genre)
    if sim:
        genre_sims[genre] = sim

# ============================================================
# 4. 실제 검색 테스트 (개선)
# ============================================================
print("\n" + "="*60)
print("[검증 3] 실제 유사 영화 검색")
print("="*60)

def search_similar_movies(query, top_k=5, by='title'):
    """
    query: 검색어
    by: 'title' 또는 'index'
    """
    if by == 'title':
        # 제목 검색
        candidates = df[df['title_ko'].str.contains(query, na=False, case=False)]
        
        if len(candidates) == 0:
            print(f"'{query}' 영화를 찾을 수 없습니다.\n")
            return
        
        target_idx = candidates.index[0]
    else:
        target_idx = query
    
    target_movie = df.loc[target_idx]
    target_vec = embedding_matrix[target_idx]
    
    print(f"\n쿼리: {target_movie['title_ko']}")
    print(f"장르: {target_movie['genres']}")
    print("-" * 40)
    
    # 유사도 계산
    sim_scores = embedding_matrix @ target_vec
    
    # Top K (자기 자신 제외)
    top_indices = np.argsort(sim_scores)[::-1][1:top_k+1]
    
    print("추천 결과:")
    for rank, idx in enumerate(top_indices, 1):
        rec_movie = df.loc[idx]
        score = sim_scores[idx]
        print(f"{rank}. [{score:.4f}] {rec_movie['title_ko']}")
        print(f"   장르: {rec_movie['genres']}")
    print()

# 테스트 케이스
test_cases = [
    "다크",      # 다크나이트 등
    "토이",      # 토이 스토리
    "어벤져스",
    "타이타닉"
]

for test in test_cases:
    try:
        search_similar_movies(test)
    except Exception as e:
        print(f"'{test}' 검색 실패: {e}\n")

# ============================================================
# 5. 장르 교차 검증 (추가)
# ============================================================
print("="*60)
print("[검증 4] 장르 교차 유사도")
print("="*60)

def cross_genre_similarity(genre1, genre2, n_samples=50):
    """두 장르 간 평균 유사도"""
    movies1 = df[df['genres'].str.contains(genre1, na=False)]
    movies2 = df[df['genres'].str.contains(genre2, na=False)]
    
    if len(movies1) < n_samples or len(movies2) < n_samples:
        return None
    
    sample1 = movies1.sample(n_samples)
    sample2 = movies2.sample(n_samples)
    
    embs1 = embedding_matrix[sample1.index]
    embs2 = embedding_matrix[sample2.index]
    
    # 교차 유사도
    cross_sim = cosine_similarity(embs1, embs2).mean()
    
    return cross_sim

# 액션 vs 로맨스 (반대 장르)
cross_sim = cross_genre_similarity('액션', '로맨스')
if cross_sim:
    print(f"액션 vs 로맨스: {cross_sim:.4f}")
    if cross_sim > 0.7:
        print("→ 경고: 다른 장르인데 유사도가 높음")
    else:
        print("→ 정상: 다른 장르는 낮은 유사도")

# ============================================================
# 6. 최종 판단
# ============================================================
print("\n" + "="*60)
print("최종 진단")
print("="*60)

if bias_status == "CRITICAL":
    print("❌ 임베딩 품질: 불량")
    print("→ 즉시 Whitening 적용 필요")
elif bias_status == "WARNING":
    print("⚠️  임베딩 품질: 보통")
    print("→ Whitening 권장")
else:
    print("✅ 임베딩 품질: 양호")
    print("→ 그대로 사용 가능")