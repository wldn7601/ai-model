import pickle
import numpy as np
import os

"""
movie_embeddings_centered 모델 평가

이 코드는 **"센터링(Centering) 기법이 제대로 적용되었는지 검증하는 코드"**로서 아주 잘 작성되었습니다.
영화 검색 (for test in test_movies): **전체 데이터(100%)**를 사용하여 검색합니다.
통계 분석 (sample_idxs = ...): **샘플 데이터(1,000개)**를 사용하여 전체적인 분포를 추정합니다.
"""

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
PKL_PATH = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_centered.pkl')

def evaluate():
    print("🚀 [Centering 결과 평가] 로딩 중...")
    
    if not os.path.exists(PKL_PATH):
        print(f"❌ 파일이 없습니다: {PKL_PATH}")
        print("   먼저 fix_vectors_centering.py를 실행하세요.")
        return

    # 1. 데이터 로드
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
        
    embeddings = data['embeddings']  # 보정된 최종 벡터
    metadata = data['metadata']
    movie_ids = data['movie_ids']
    
    # ID 매핑 (ID -> Index)
    id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    
    print(f"   ✓ 데이터 로드 완료 ({len(embeddings):,}개)")
    
    # 2. 테스트할 영화 목록
    test_movies = [
        {'id': 1, 'keyword': '토이 스토리'},
        {'id': 2571, 'keyword': '매트릭스'},
        {'id': 1721, 'keyword': '타이타닉'}, # 타이타닉 (1997)
        {'id': 296, 'keyword': '펄프 픽션'}
    ]
    
    # 영화 ID가 없을 경우 제목으로 검색해서 보정
    for test in test_movies:
        if test['id'] not in id_to_idx:
            for m in metadata:
                if test['keyword'] in m['title']:
                    test['id'] = m['movieId']
                    break

    # 3. 유사도 검색 및 출력
    # (이미 Centering과 Normalize가 되어 있지만, 안전을 위해 한번 더 정규화)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    for test in test_movies:
        mid = test['id']
        keyword = test['keyword']
        
        if mid not in id_to_idx:
            print(f"\n❌ 영화 '{keyword}'를 찾을 수 없습니다.")
            continue
            
        idx = id_to_idx[mid]
        query_vec = embeddings[idx]
        query_meta = metadata[idx]
        
        # 내적 (Cosine Similarity)
        scores = np.dot(embeddings, query_vec)
        
        # 상위 10개 (자기 자신 제외)
        top_k_idx = np.argsort(scores)[::-1][1:11]
        
        print(f"\n🎥 기준 영화: {query_meta['title']} (장르: {', '.join(query_meta['genres'])})")
        print(f"{'순위':<4} {'유사도':<8} {'제목':<30} {'장르'}")
        print("-" * 70)
        
        for rank, i in enumerate(top_k_idx, 1):
            m = metadata[i]
            genres = ", ".join(m['genres'][:2])
            # 제목이 너무 길면 자르기
            title = m['title']
            if len(title) > 28: title = title[:25] + "..."
            
            print(f"{rank:<4} {scores[i]:.4f}   {title:<30} {genres}")
            
    # 4. 전체 통계 출력
    print("\n>>> 전체 유사도 분포 통계 (샘플 1000개)")
    sample_idxs = np.random.choice(len(embeddings), 1000, replace=False)
    sample_vecs = embeddings[sample_idxs]
    
    sims = np.dot(sample_vecs, sample_vecs.T)
    # 자기 자신(1.0) 제외하고 통계
    sims_no_diag = sims[~np.eye(sims.shape[0], dtype=bool)]
    
    avg_sim = np.mean(sims_no_diag)
    min_sim = np.min(sims_no_diag)
    max_sim = np.max(sims_no_diag)
    
    print(f"   📊 평균 유사도: {avg_sim:.4f}")
    print(f"   📉 최소/최대: {min_sim:.4f} / {max_sim:.4f}")
    
    if avg_sim < 0.3:
        print("   ✅ [Pass] 변별력이 매우 우수합니다.")
    elif avg_sim < 0.5:
        print("   ✅ [Pass] 추천 시스템에 적합한 분포입니다.")
    else:
        print("   ⚠️ [Warning] 여전히 유사도가 높습니다.")

if __name__ == "__main__":
    evaluate()