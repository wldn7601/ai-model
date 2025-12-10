"""
e5-large 임베딩 기반 영화 추천 시스템 테스트
작업 위치: ~/ai-model/models/cbf/v1_basic/scripts/
"""

import pickle
import numpy as np
from typing import List, Dict, Optional


class MovieRecommender:
    """
    코사인 유사도 기반 영화 추천 시스템
    """
    
    def __init__(self, embeddings_path: str):
        """
        임베딩 로드
        """
        print(">>> 추천 시스템 초기화 중...")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']  # (13043, 1024)
        self.movie_ids = data['movie_ids']
        self.metadata = data['metadata']
        self.model_name = data['model_name']
        
        # movieId -> index 매핑
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        
        print(f"   ✓ 영화 수: {len(self.movie_ids):,}개")
        print(f"   ✓ 임베딩 차원: {self.embeddings.shape[1]}D")
        print(f"   ✓ 모델: {self.model_name}")
    
    
    def create_user_vector(
        self, 
        user_ratings: Dict[int, float]
    ) -> np.ndarray:
        """
        온보딩 평점으로 사용자 벡터 생성
        
        Args:
            user_ratings: {movieId: rating, ...}
                예: {1: 5.0, 260: 4.5, 527: 3.0}
        
        Returns:
            사용자 벡터 (1024,)
        """
        weighted_sum = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        
        valid_count = 0
        
        for movie_id, rating in user_ratings.items():
            if movie_id not in self.id_to_idx:
                print(f"   ⚠️  영화 ID {movie_id} 없음 (건너뜀)")
                continue
            
            idx = self.id_to_idx[movie_id]
            movie_vec = self.embeddings[idx]
            
            # 평점 → 가중치 변환
            # 5점 = +1.0, 4점 = +0.5, 3점 = 0, 2점 = -0.5, 1점 = -1.0
            weight = (rating - 3.0) / 2.0
            
            weighted_sum += movie_vec * weight
            total_weight += abs(weight)
            valid_count += 1
        
        if valid_count == 0:
            raise ValueError("유효한 영화가 없습니다")
        
        # 가중 평균
        user_vector = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        # L2 정규화
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
        
        print(f"   ✓ 사용자 벡터 생성 완료 (유효 영화: {valid_count}개)")
        
        return user_vector
    
    
    def recommend(
        self,
        user_vector: np.ndarray,
        genre_filter: Optional[List[str]] = None,
        runtime_min: Optional[int] = None,
        runtime_max: Optional[int] = None,
        ott_filter: Optional[List[str]] = None,
        exclude_movie_ids: Optional[List[int]] = None,
        top_k: int = 20
    ) -> List[Dict]:
        """
        코사인 유사도 기반 추천
        
        Args:
            user_vector: 사용자 벡터
            genre_filter: 장르 필터 (예: ['로맨스', '코미디'])
            runtime_min/max: 런타임 범위 (분)
            ott_filter: OTT 필터 (예: ['Netflix', 'Disney Plus'])
            exclude_movie_ids: 제외할 영화 ID
            top_k: 추천 개수
        
        Returns:
            추천 영화 리스트
        """
        # 1. 코사인 유사도 계산 (이미 정규화되어 있으므로 내적만)
        similarities = np.dot(self.embeddings, user_vector)
        
        # 2. 필터링
        valid_indices = []
        
        for i, meta in enumerate(self.metadata):
            movie_id = meta['movieId']
            
            # 이미 본 영화 제외
            if exclude_movie_ids and movie_id in exclude_movie_ids:
                continue
            
            # 장르 필터
            if genre_filter:
                if not any(g in meta['genres'] for g in genre_filter):
                    continue
            
            # 런타임 필터
            runtime = meta['runtime']
            if runtime_min and runtime < runtime_min:
                continue
            if runtime_max and runtime > runtime_max:
                continue
            
            # OTT 필터
            if ott_filter:
                if not any(ott in meta['providers'] for ott in ott_filter):
                    continue
            
            valid_indices.append(i)
        
        # 3. 상위 K개 선택
        valid_sims = [(i, similarities[i]) for i in valid_indices]
        valid_sims.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 결과 반환
        recommendations = []
        for idx, sim_score in valid_sims[:top_k]:
            meta = self.metadata[idx]
            recommendations.append({
                'movieId': meta['movieId'],
                'title': meta['title'],
                'similarity': float(sim_score),
                'runtime': meta['runtime'],
                'genres': meta['genres'],
                'providers': meta['providers'],
                'vote_average': meta['vote_average']
            })
        
        return recommendations


def test_basic_recommendation():
    """
    기본 추천 테스트
    """
    print("="*70)
    print(" e5-large 추천 시스템 테스트")
    print("="*70)
    print()
    
    # 1. 추천 시스템 초기화
    recommender = MovieRecommender(
        '../outputs/movie_embeddings_e5.pkl'
    )
    
    # 2. 테스트 사용자 (온보딩 시뮬레이션)
    print("\n>>> 사용자 온보딩 평점:")
    user_ratings = {
        1: 5.0,      # 토이 스토리 (애니메이션, 가족)
        260: 5.0,    # 스타워즈 (SF, 모험)
        527: 4.0,    # 쉰들러 리스트 (드라마, 역사)
        2571: 5.0,   # 매트릭스 (SF, 액션)
        593: 3.0,    # 사일런스 오브 더 램스 (스릴러)
    }
    
    for mid, rating in user_ratings.items():
        idx = recommender.id_to_idx.get(mid)
        if idx is not None:
            title = recommender.metadata[idx]['title']
            genres = ', '.join(recommender.metadata[idx]['genres'])
            print(f"  {title}: {rating}점 ({genres})")
    
    # 3. 사용자 벡터 생성
    print(f"\n>>> 사용자 벡터 생성:")
    user_vector = recommender.create_user_vector(user_ratings)
    print(f"   ✓ 벡터 shape: {user_vector.shape}")
    print(f"   ✓ 벡터 norm: {np.linalg.norm(user_vector):.6f}")
    
    # 4. 추천 실행
    print(f"\n>>> 추천 결과 (Top 10):")
    print(f"{'순위':<4} {'제목':<30} {'유사도':<8} {'장르':<30} {'런타임':<6} {'평점':<5}")
    print("="*90)
    
    recommendations = recommender.recommend(
        user_vector,
        exclude_movie_ids=list(user_ratings.keys()),
        top_k=10
    )
    
    for i, rec in enumerate(recommendations, 1):
        title = rec['title'][:28]
        genres = ', '.join(rec['genres'][:2])[:28]
        print(f"{i:<4} {title:<30} {rec['similarity']:.4f}   {genres:<30} {rec['runtime']:<6} {rec['vote_average']:.1f}")
    
    return recommender, user_vector, recommendations


def test_filtered_recommendation(recommender, user_vector):
    """
    필터링 테스트
    """
    print("\n\n" + "="*70)
    print(" 필터링 테스트")
    print("="*70)
    
    # 1. 장르 필터
    print("\n>>> 1. 장르 필터 (로맨스만):")
    romance_recs = recommender.recommend(
        user_vector,
        genre_filter=['로맨스'],
        top_k=5
    )
    
    for i, rec in enumerate(romance_recs, 1):
        print(f"  {i}. {rec['title']} ({', '.join(rec['genres'])}) - {rec['similarity']:.4f}")
    
    # 2. 런타임 필터
    print("\n>>> 2. 런타임 필터 (90~120분):")
    runtime_recs = recommender.recommend(
        user_vector,
        runtime_min=90,
        runtime_max=120,
        top_k=5
    )
    
    for i, rec in enumerate(runtime_recs, 1):
        print(f"  {i}. {rec['title']} ({rec['runtime']}분) - {rec['similarity']:.4f}")
    
    # 3. 장르 + 런타임 복합 필터
    print("\n>>> 3. 복합 필터 (액션, 100~150분):")
    combined_recs = recommender.recommend(
        user_vector,
        genre_filter=['액션'],
        runtime_min=100,
        runtime_max=150,
        top_k=5
    )
    
    for i, rec in enumerate(combined_recs, 1):
        print(f"  {i}. {rec['title']} ({rec['runtime']}분, {', '.join(rec['genres'])}) - {rec['similarity']:.4f}")


def test_different_users():
    """
    다양한 사용자 프로필 테스트
    """
    print("\n\n" + "="*70)
    print(" 다양한 사용자 프로필 테스트")
    print("="*70)
    
    recommender = MovieRecommender('../outputs/movie_embeddings_e5.pkl')
    
    # 사용자 1: 애니메이션 팬
    print("\n>>> 사용자 1: 애니메이션 팬")
    user1_ratings = {
        1: 5.0,     # 토이 스토리
        862: 5.0,   # 토이 스토리 2
        2355: 5.0,  # 몬스터 주식회사
    }
    
    user1_vec = recommender.create_user_vector(user1_ratings)
    user1_recs = recommender.recommend(user1_vec, top_k=5)
    
    for i, rec in enumerate(user1_recs, 1):
        print(f"  {i}. {rec['title']} ({', '.join(rec['genres'])}) - {rec['similarity']:.4f}")
    
    # 사용자 2: SF 마니아
    print("\n>>> 사용자 2: SF 마니아")
    user2_ratings = {
        260: 5.0,   # 스타워즈
        2571: 5.0,  # 매트릭스
        329: 5.0,   # 쥬라기 공원
    }
    
    user2_vec = recommender.create_user_vector(user2_ratings)
    user2_recs = recommender.recommend(user2_vec, top_k=5)
    
    for i, rec in enumerate(user2_recs, 1):
        print(f"  {i}. {rec['title']} ({', '.join(rec['genres'])}) - {rec['similarity']:.4f}")


def main():
    """
    전체 테스트 실행
    """
    try:
        # 1. 기본 추천
        recommender, user_vector, recs = test_basic_recommendation()
        
        # 2. 필터링 테스트
        test_filtered_recommendation(recommender, user_vector)
        
        # 3. 다양한 사용자
        test_different_users()
        
        print("\n\n" + "="*70)
        print("✅ 모든 테스트 완료!")
        print("="*70)
        print("\n다음 단계:")
        print("  1. 추천 품질 평가 (Precision, NDCG)")
        print("  2. 런타임 조합 최적화 (Knapsack)")
        print("  3. ALS와 하이브리드 앙상블")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()