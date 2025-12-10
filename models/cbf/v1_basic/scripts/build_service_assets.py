import pickle
import json
import numpy as np
import os
from sklearn.cluster import KMeans

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
# 가장 품질이 좋았던 '보정된(Centered)' 임베딩 파일 사용
INPUT_PKL = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_centered.pkl')
# 저장할 서비스 자산 파일
OUTPUT_ASSETS = os.path.join(BASE_DIR, 'data', 'service_assets.json')

def build_assets():
    print("🚀 [Admin] 서비스 자산(장르, OTT, 대표영화) 구축 시작...")
    
    # 1. 데이터 로드
    if not os.path.exists(INPUT_PKL):
        print(f"❌ 파일이 없습니다: {INPUT_PKL}")
        return

    with open(INPUT_PKL, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    metadata = data['metadata']
    print(f"   ✓ 데이터 로드 완료 ({len(metadata):,}개)")

    # 2. 메타데이터 추출 (장르, OTT 목록)
    all_genres = set()
    all_providers = set()
    
    for m in metadata:
        # 장르 수집
        for g in m['genres']:
            all_genres.add(g)
        # OTT 수집
        for p in m.get('providers', []):
            all_providers.add(p)
            
    sorted_genres = sorted(list(all_genres))
    sorted_providers = sorted(list(all_providers))
    
    print(f"   ✓ 장르 목록 추출: {len(sorted_genres)}개")
    print(f"   ✓ OTT 목록 추출: {len(sorted_providers)}개")

    # 3. 대표 영화 선정 (Clustering for Diversity)
    # 태그/분위기(벡터) 기준으로 10개 그룹으로 나눔
    print("   ✓ 대표 영화 선정 중 (K-Means Clustering)...")
    
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    representative_movies = []
    
    for i in range(n_clusters):
        # 해당 클러스터에 속한 영화들의 인덱스
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_vecs = embeddings[cluster_indices]
        
        # 중심점과 가장 가까운 영화 찾기
        distances = np.linalg.norm(cluster_vecs - centers[i], axis=1)
        best_idx_local = np.argmin(distances)
        best_idx_global = cluster_indices[best_idx_local]
        
        movie = metadata[best_idx_global]
        
        # ★ 수정된 부분: movie.title() -> movie['title']
        representative_movies.append({
            'movieId': movie['movieId'],
            'title': movie['title'], 
            'genres': movie['genres'],
            'cluster_id': i
        })

    # 4. JSON 파일로 저장
    assets = {
        'available_genres': sorted_genres,
        'available_providers': sorted_providers,
        'onboarding_movies': representative_movies
    }
    
    with open(OUTPUT_ASSETS, 'w', encoding='utf-8') as f:
        json.dump(assets, f, ensure_ascii=False, indent=2)
        
    print(f"\n💾 저장 완료: {OUTPUT_ASSETS}")
    print("   이제 'run_terminal_app.py'를 실행할 수 있습니다.")

if __name__ == "__main__":
    build_assets()