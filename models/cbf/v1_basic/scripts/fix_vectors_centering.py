import pickle
import numpy as np
import os

"""
# 
# 저장된 임베딩을 불러와서 **Centering(보정)**을 수행하고, 결과가 얼마나 좋아졌는지 보여줍니다.
# movie_embeddings_centered 모델이 생성

"""

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
# 방금 만드신 파일 경로
INPUT_PKL = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_separate_e5.pkl')
# 보정된 결과를 저장할 파일
OUTPUT_PKL = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_centered.pkl')

# 가중치 설정 (태그 중심 추천)
W_GENRE = 0.2
W_TAG = 0.7
W_OVERVIEW = 0.1

def subtract_mean(vectors):
    """
    [핵심 로직] 평균 벡터 제거 (Centering)
    모든 벡터가 원점(0,0)을 기준으로 골고루 퍼지게 만듦
    """
    mean_vec = np.mean(vectors, axis=0)
    centered_vectors = vectors - mean_vec
    return centered_vectors

def check_similarity(name, vectors):
    """
    유사도 분포 확인 함수
    """
    # 1000개만 샘플링해서 빠르게 계산
    idx = np.random.choice(len(vectors), 1000, replace=False)
    sample = vectors[idx]
    
    # 정규화 (유사도 계산용)
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    sample = sample / norms
    
    # 내적 (Cosine Similarity)
    sims = np.dot(sample, sample.T)
    # 자기 자신(1.0) 제외
    sims = sims[~np.eye(sims.shape[0], dtype=bool)]
    
    avg = np.mean(sims)
    min_v = np.min(sims)
    max_v = np.max(sims)
    
    print(f"   [{name}] 평균: {avg:.4f}  (범위: {min_v:.2f} ~ {max_v:.2f})")
    return avg

def main():
    print("🚀 [Vector Centering] 임베딩 보정 작업 시작...")
    
    if not os.path.exists(INPUT_PKL):
        print("❌ 파일이 없습니다.")
        return

    # 1. 데이터 로드
    with open(INPUT_PKL, 'rb') as f:
        data = pickle.load(f)
    
    genre_vecs = data['genre_embeddings']
    tag_vecs = data['tag_embeddings']
    overview_vecs = data['overview_embeddings']
    
    print(f"   ✓ 데이터 로드 완료 ({len(genre_vecs):,}개)")

    # 2. 보정 전 상태 확인
    print("\n>>> [1] 보정 전 (Original) 상태")
    check_similarity("장르", genre_vecs)
    check_similarity("태그", tag_vecs)
    check_similarity("줄거리", overview_vecs)

    # 3. 보정 수행 (Centering)
    print("\n>>> [2] 평균 벡터 제거 수행...")
    genre_centered = subtract_mean(genre_vecs)
    tag_centered = subtract_mean(tag_vecs)
    overview_centered = subtract_mean(overview_vecs)

    print("\n>>> [3] 보정 후 (Centered) 상태")
    check_similarity("장르", genre_centered)
    check_similarity("태그", tag_centered)
    check_similarity("줄거리", overview_centered)

    # 4. 최종 결합
    print(f"\n>>> [4] 최종 결합 (G:{W_GENRE} / T:{W_TAG} / O:{W_OVERVIEW})")
    
    combined = (
        genre_centered * W_GENRE +
        tag_centered * W_TAG +
        overview_centered * W_OVERVIEW
    )
    
    # 최종 정규화
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / norms
    
    avg_final = check_similarity("최종 결과", combined)

    if avg_final < 0.3:
        print("\n✅ 성공! 유사도 분포가 아주 이상적입니다.")
    elif avg_final < 0.6:
        print("\n✅ 양호! 추천 시스템에 쓰기 적합합니다.")
    else:
        print("\n⚠️ 여전히 높음. 가중치를 더 조절해보세요.")

    # 5. 저장
    print(f"\n>>> [5] 결과 저장 중... ({OUTPUT_PKL})")
    
    # 기존 데이터 구조 유지하면서 임베딩 교체
    data['embeddings'] = combined
    data['genre_embeddings'] = genre_centered
    data['tag_embeddings'] = tag_centered
    data['overview_embeddings'] = overview_centered
    
    # 메타데이터에 '보정됨' 표시
    data['config']['method'] = 'centered_late_fusion'
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data, f, protocol=4)
        
    print("🎉 작업 완료! 이제 DB 적재 스크립트를 실행하세요.")

if __name__ == "__main__":
    main()