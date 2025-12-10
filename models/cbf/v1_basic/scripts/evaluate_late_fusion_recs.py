import pickle
import numpy as np
import os
from typing import List, Dict, Optional

"""
movie_embeddings_separate_e5 모델 가져와서

장르, 태그, overview 가중치 여러개 해서 평가
"""

class LateFusionRecommender:
    """
    Late Fusion 기반 가중치 조절 가능한 추천 시스템
    """
    
    def __init__(self, pkl_path: str):
        print(f">>> [Init] 데이터 로딩 중... ({pkl_path})")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 1. 개별 임베딩 로드 (핵심)
        self.genre_vecs = data['genre_embeddings']
        self.tag_vecs = data['tag_embeddings']
        self.overview_vecs = data['overview_embeddings']
        
        # 메타데이터
        self.movie_ids = data['movie_ids']
        self.metadata = data['metadata']
        
        # ID 매핑
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        
        # 현재 활성화된 임베딩 (초기값: None)
        self.current_embeddings = None
        
        print(f"   ✓ 영화 수: {len(self.movie_ids):,}개")
        print(f"   ✓ 개별 벡터 로드 완료 (장르, 태그, 줄거리)")

    def update_weights(self, w_genre: float, w_tag: float, w_overview: float):
        """
        실시간으로 가중치를 변경하고 임베딩을 재결합합니다.
        GPU 없이 CPU 연산만으로 순식간에 처리됩니다.
        """
        print(f"\n>>> [Weights Update] 가중치 변경: 장르({w_genre}) + 태그({w_tag}) + 줄거리({w_overview})")
        
        # 벡터 결합
        combined = (
            self.genre_vecs * w_genre +
            self.tag_vecs * w_tag +
            self.overview_vecs * w_overview
        )
        
        # L2 정규화 (필수)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        # 0으로 나누기 방지
        norms[norms == 0] = 1e-10
        self.current_embeddings = combined / norms
        
        print("   ✓ 임베딩 재결합 완료")

    def create_user_vector(self, user_ratings: Dict[int, float]) -> np.ndarray:
        """
        사용자 벡터 생성 (현재 활성화된 임베딩 기준)
        """
        if self.current_embeddings is None:
            raise ValueError("먼저 update_weights()를 호출하세요.")

        weighted_sum = np.zeros(self.current_embeddings.shape[1])
        total_weight = 0
        valid_count = 0
        
        for movie_id, rating in user_ratings.items():
            if movie_id not in self.id_to_idx:
                continue
            
            idx = self.id_to_idx[movie_id]
            # 3.0점 기준 가중치 (-1.0 ~ +1.0)
            weight = (rating - 3.0) / 2.0
            
            weighted_sum += self.current_embeddings[idx] * weight
            total_weight += abs(weight)
            valid_count += 1
            
        if valid_count == 0:
            return np.zeros(self.current_embeddings.shape[1])
            
        user_vector = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        # 정규화
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
            
        return user_vector

    def recommend(self, user_vector: np.ndarray, top_k: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
        """
        추천 실행
        """
        # 코사인 유사도
        scores = np.dot(self.current_embeddings, user_vector)
        
        # 정렬
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        count = 0
        
        for idx in top_indices:
            if count >= top_k:
                break
                
            mid = self.movie_ids[idx]
            if exclude_ids and mid in exclude_ids:
                continue
                
            meta = self.metadata[idx]
            results.append({
                'title': meta['title'],
                'similarity': float(scores[idx]),
                'genres': meta['genres'],
                'tags_sample': self._get_tags_sample(idx)
            })
            count += 1
            
        return results

    def _get_tags_sample(self, idx):
        # 태그 정보가 메타데이터에 없으면 genome_tags 등에서 가져와야 함 (생략 가능)
        return []

# ==========================================
# 테스트 시나리오
# ==========================================
def main():
    BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
    PKL_PATH = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_separate_e5.pkl')
    
    # 1. 초기화
    recommender = LateFusionRecommender(PKL_PATH)
    
    # 2. 테스트 사용자 (SF/액션 선호)
    user_ratings = {
        260: 5.0,   # 스타워즈
        2571: 5.0,  # 매트릭스
        296: 4.0    # 펄프 픽션 (범죄)
    }
    
    # 3. 실험 A: 기본 설정 (0.4 / 0.5 / 0.1)
    # ---------------------------------------------------------
    recommender.update_weights(w_genre=0.4, w_tag=0.5, w_overview=0.1)
    user_vec_a = recommender.create_user_vector(user_ratings)
    recs_a = recommender.recommend(user_vec_a, top_k=5, exclude_ids=list(user_ratings.keys()))
    
    print("\n[실험 A: 기본 설정] 결과:")
    for i, r in enumerate(recs_a, 1):
        print(f"  {i}. {r['title']} ({', '.join(r['genres'][:2])}) - Sim: {r['similarity']:.4f}")

    # 4. 실험 B: 줄거리 몰빵 (0.1 / 0.1 / 0.8) -> 유사도 인플레이션 확인용
    # ---------------------------------------------------------
    recommender.update_weights(w_genre=0.1, w_tag=0.1, w_overview=0.8)
    user_vec_b = recommender.create_user_vector(user_ratings) # 임베딩이 바뀌었으니 유저 벡터도 다시 생성해야 함!
    recs_b = recommender.recommend(user_vec_b, top_k=5, exclude_ids=list(user_ratings.keys()))
    
    print("\n[실험 B: 줄거리 중심 (비추천)] 결과:")
    for i, r in enumerate(recs_b, 1):
        print(f"  {i}. {r['title']} ({', '.join(r['genres'][:2])}) - Sim: {r['similarity']:.4f}")

    # 5. 실험 C: 장르/태그 중심 (0.5 / 0.5 / 0.0) -> 깔끔한 분류
    # ---------------------------------------------------------
    recommender.update_weights(w_genre=0.5, w_tag=0.5, w_overview=0.0)
    user_vec_c = recommender.create_user_vector(user_ratings)
    recs_c = recommender.recommend(user_vec_c, top_k=5, exclude_ids=list(user_ratings.keys()))
    
    print("\n[실험 C: 장르/태그 중심 (추천)] 결과:")
    for i, r in enumerate(recs_c, 1):
        print(f"  {i}. {r['title']} ({', '.join(r['genres'][:2])}) - Sim: {r['similarity']:.4f}")

if __name__ == "__main__":
    main()