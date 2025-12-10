import pickle
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Optional

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
PKL_PATH = f'{BASE_DIR}/outputs/movie_embeddings_centered.pkl'

class ServiceSimulator:
    def __init__(self):
        print(">>> [System] 서비스 초기화 중...")
        with open(PKL_PATH, 'rb') as f:
            data = pickle.load(f)
            
        self.embeddings = data['embeddings'] # (13043, 1024)
        self.metadata = data['metadata']
        self.movie_ids = data['movie_ids']
        
        # ID -> Index 매핑
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        print(f"   ✓ 데이터 로드 완료 ({len(self.movie_ids):,}개)")

    # ---------------------------------------------------------
    # [Step 1] 온보딩: 다양성을 가진 대표 영화 10개 선정
    # ---------------------------------------------------------
    def get_onboarding_movies(self, n_clusters=10) -> List[Dict]:
        print(f"\n>>> [Step 1] 온보딩용 대표 영화 {n_clusters}개 선정 (Clustering)...")
        
        # 1. K-Means로 전체 영화를 10개 그룹으로 나눔
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(self.embeddings)
        
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        representatives = []
        
        # 2. 각 그룹(Cluster)의 중심에 가장 가까운 영화 찾기
        for i in range(n_clusters):
            # 해당 그룹에 속한 영화들의 인덱스
            cluster_indices = np.where(labels == i)[0]
            
            # 중심점과의 거리 계산
            center_vec = centers[i]
            cluster_vecs = self.embeddings[cluster_indices]
            
            # 유클리드 거리 기준 가장 가까운 것
            distances = np.linalg.norm(cluster_vecs - center_vec, axis=1)
            best_idx_in_cluster = np.argmin(distances)
            real_idx = cluster_indices[best_idx_in_cluster]
            
            meta = self.metadata[real_idx]
            representatives.append({
                'movieId': meta['movieId'],
                'title': meta['title'],
                'genres': meta['genres'],
                'group_id': i
            })
            
        print("   ✓ 대표 영화 선정 완료:")
        for m in representatives:
            print(f"     [{m['group_id']}] {m['title']} ({', '.join(m['genres'][:2])})")
            
        return representatives

    # ---------------------------------------------------------
    # [Step 2] 사용자 벡터 생성 (평가 반영)
    # ---------------------------------------------------------
    def create_user_vector(self, user_ratings: Dict[int, float]) -> np.ndarray:
        print(f"\n>>> [Step 2] 사용자 취향 분석 (평가 {len(user_ratings)}건)...")
        
        weighted_sum = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        
        for mid, rating in user_ratings.items():
            if mid not in self.id_to_idx: continue
            
            idx = self.id_to_idx[mid]
            vec = self.embeddings[idx]
            
            # 가중치: (점수 - 3.0) / 2.0  ->  1점(-1.0) ~ 5점(+1.0)
            weight = (rating - 3.0) / 2.0
            
            weighted_sum += vec * weight
            total_weight += abs(weight)
            
        if total_weight == 0:
            return np.zeros(self.embeddings.shape[1])
            
        # 평균 및 정규화
        user_vec = weighted_sum / total_weight
        norm = np.linalg.norm(user_vec)
        if norm > 0: user_vec = user_vec / norm
            
        print("   ✓ User Vector 생성 완료")
        return user_vec

    # ---------------------------------------------------------
    # [Step 3] 필터링 + 추천 (서비스 로직)
    # ---------------------------------------------------------
    def recommend(self, user_vec, filter_opts: Dict, top_k=3):
        print(f"\n>>> [Step 3] 영화 추천 요청")
        print(f"   조건: {filter_opts}")
        
        # 1. 1차 필터링 (Hard Filter) - DB의 WHERE 절 역할
        candidates_idx = []
        
        target_ott = filter_opts.get('ott')
        target_genre = filter_opts.get('genre')
        max_runtime = filter_opts.get('runtime')
        
        for i, meta in enumerate(self.metadata):
            # OTT 체크
            if target_ott:
                # 데이터에 providers가 있는지 확인 (없으면 통과 or 제외 정책 결정)
                providers = meta.get('providers', [])
                # 간단한 문자열 매칭 (실제론 ID 매칭 권장)
                if not any(target_ott.lower() in p.lower() for p in providers):
                    continue
            
            # 장르 체크
            if target_genre:
                if target_genre not in meta['genres']:
                    continue
                    
            # 런타임 체크
            if max_runtime:
                if meta.get('runtime', 999) > max_runtime:
                    continue
                    
            candidates_idx.append(i)
            
        print(f"   ✓ 필터링 통과 후보: {len(candidates_idx)}개")
        
        if not candidates_idx:
            print("   ❌ 조건에 맞는 영화가 없습니다.")
            return []

        # 2. 2차 랭킹 (Soft Ranking) - 벡터 유사도
        candidate_vecs = self.embeddings[candidates_idx]
        
        # 내적 (코사인 유사도)
        scores = np.dot(candidate_vecs, user_vec)
        
        # 상위 K개 정렬
        # argsort는 오름차순이므로 뒤집어서 상위권 추출
        top_local_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for local_idx in top_local_indices:
            real_idx = candidates_idx[local_idx]
            meta = self.metadata[real_idx]
            score = scores[local_idx]
            
            recommendations.append({
                'title': meta['title'],
                'score': score,
                'genres': meta['genres'],
                'runtime': meta.get('runtime'),
                'providers': meta.get('providers', [])
            })
            
        return recommendations

# ==========================================
# 실행 시나리오
# ==========================================
def main():
    sim = ServiceSimulator()
    
    # 1. 온보딩 영화 제공
    onboarding_movies = sim.get_onboarding_movies()
    
    # [가정] 사용자가 이 중 3개를 평가했다고 가정
    # 예: 액션 영화(그룹 0)는 싫고(-), 로맨스(그룹 5)는 좋고(+)
    # 실제로는 onboarding_movies 리스트에서 ID를 보고 매핑해야 함
    
    # 시뮬레이션을 위해 임의의 ID와 평점 부여 (취향: 드라마/로맨스 선호, 액션/공포 불호)
    # 실제 환경에서는 프론트엔드에서 넘어온 데이터
    user_ratings = {
        1: 5.0,     # 토이 스토리 (애니) - 극호
        1721: 5.0,  # 타이타닉 (로맨스) - 극호
        2571: 1.0,  # 매트릭스 (SF) - 불호
        593: 1.0    # 양들의 침묵 (스릴러) - 불호
    }
    
    # 2. 사용자 벡터 생성
    user_vec = sim.create_user_vector(user_ratings)
    
    # 3. 추천 요청 (상황: 비행기 안, 2시간 이내, 넷플릭스, 가족 영화)
    filters = {
        'genre': '가족',       # 장르 필터
        'runtime': 120,       # 2시간 이내
        'ott': 'Netflix'      # 넷플릭스 구독
    }
    
    recs = sim.recommend(user_vec, filters, top_k=3)
    
    print("\n🎥 [최종 추천 결과]")
    for i, movie in enumerate(recs, 1):
        print(f"{i}위: {movie['title']} ({movie['score']:.4f})")
        print(f"     장르: {', '.join(movie['genres'])}")
        print(f"     시간: {movie['runtime']}분 | OTT: {movie['providers']}")

if __name__ == "__main__":
    main()