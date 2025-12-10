"""
CBF 임베딩 품질 정성 평가
작업 위치: ~/ai-model/models/cbf/v1_basic/scripts/
"""

"""
# movie_embeddings_e5 모델 : 각각 임베딩과 보정 X

# "토이 스토리랑 비슷한 거 찾아줘" (test_popular_movies): 전체 100% 데이터를 다 뒤져서 찾습니다. (정확함)
# "전체적으로 유사도 평균이 몇이야?" (analyze_embedding_quality): 1,000개만 뽑아서 대략적인 추세를 봅니다. (빠름)

"""

import pickle
import numpy as np
from typing import List, Dict


class EmbeddingEvaluator:
    """
    임베딩 품질 평가
    """
    
    def __init__(self, embeddings_path: str):
        print(">>> 평가 시스템 로드 중...")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.movie_ids = data['movie_ids']
        self.metadata = data['metadata']
        
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        
        print(f"   ✓ 영화 수: {len(self.movie_ids):,}개")
    
    
    def find_similar_movies(
        self, 
        movie_id: int, 
        top_k: int = 10,
        show_details: bool = True
    ) -> List[Dict]:
        """
        특정 영화와 유사한 영화 찾기
        """
        if movie_id not in self.id_to_idx:
            print(f"❌ 영화 ID {movie_id} 없음")
            return []
        
        idx = self.id_to_idx[movie_id]
        query_vec = self.embeddings[idx]
        
        # 코사인 유사도 (이미 정규화됨)
        similarities = np.dot(self.embeddings, query_vec)
        
        # 자기 자신 제외하고 상위 K개
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            meta = self.metadata[idx]
            results.append({
                'rank': rank,
                'movieId': meta['movieId'],
                'title': meta['title'],
                'similarity': float(similarities[idx]),
                'genres': meta['genres'],
                'runtime': meta['runtime'],
                'vote_average': meta['vote_average']
            })
        
        if show_details:
            query_meta = self.metadata[self.id_to_idx[movie_id]]
            print(f"\n{'='*80}")
            print(f"기준 영화: {query_meta['title']}")
            print(f"장르: {', '.join(query_meta['genres'])}")
            print(f"런타임: {query_meta['runtime']}분")
            print(f"{'='*80}")
            print(f"\n유사한 영화 Top {top_k}:")
            print(f"{'순위':<4} {'제목':<35} {'유사도':<8} {'장르':<25} {'런타임':<6}")
            print("-"*80)
            
            for rec in results:
                title = rec['title'][:33]
                genres = ', '.join(rec['genres'][:2])[:23]
                print(f"{rec['rank']:<4} {title:<35} {rec['similarity']:.4f}   {genres:<25} {rec['runtime']:<6}")
        
        return results
    
    
    def evaluate_by_genre(self, movie_id: int, top_k: int = 10):
        """
        장르 일치도 평가
        """
        if movie_id not in self.id_to_idx:
            return
        
        idx = self.id_to_idx[movie_id]
        query_meta = self.metadata[idx]
        query_genres = set(query_meta['genres'])
        
        similar = self.find_similar_movies(movie_id, top_k, show_details=False)
        
        # 장르 일치 분석
        exact_match = 0
        partial_match = 0
        no_match = 0
        
        for rec in similar:
            rec_genres = set(rec['genres'])
            overlap = len(query_genres & rec_genres)
            
            if overlap == len(query_genres):
                exact_match += 1
            elif overlap > 0:
                partial_match += 1
            else:
                no_match += 1
        
        print(f"\n[장르 일치도 평가]")
        print(f"  완전 일치: {exact_match}/{top_k} ({exact_match/top_k*100:.1f}%)")
        print(f"  부분 일치: {partial_match}/{top_k} ({partial_match/top_k*100:.1f}%)")
        print(f"  불일치:   {no_match}/{top_k} ({no_match/top_k*100:.1f}%)")
    
    
    def evaluate_diversity(self, movie_id: int, top_k: int = 10):
        """
        추천 다양성 평가
        """
        similar = self.find_similar_movies(movie_id, top_k, show_details=False)
        
        # 고유 장르 수
        all_genres = set()
        for rec in similar:
            all_genres.update(rec['genres'])
        
        # 런타임 분포
        runtimes = [rec['runtime'] for rec in similar]
        runtime_std = np.std(runtimes)
        
        # 평점 분포
        ratings = [rec['vote_average'] for rec in similar]
        rating_std = np.std(ratings)
        
        print(f"\n[다양성 평가]")
        print(f"  고유 장르 수: {len(all_genres)}개")
        print(f"  런타임 표준편차: {runtime_std:.1f}분")
        print(f"  평점 표준편차: {rating_std:.2f}")


def test_popular_movies():
    """
    유명 영화로 테스트
    """
    evaluator = EmbeddingEvaluator('../outputs/movie_embeddings_e5.pkl')
    
    test_cases = [
        {'id': 1, 'name': '토이 스토리 (애니메이션/가족)'},
        {'id': 260, 'name': '스타워즈 (SF/모험)'},
        {'id': 2571, 'name': '매트릭스 (SF/액션)'},
        {'id': 527, 'name': '쉰들러 리스트 (드라마/역사)'},
        {'id': 318, 'name': '쇼생크 탈출 (드라마)'},
    ]
    
    print("="*80)
    print(" 임베딩 품질 정성 평가 - 유명 영화 테스트")
    print("="*80)
    
    for case in test_cases:
        evaluator.find_similar_movies(case['id'], top_k=10)
        evaluator.evaluate_by_genre(case['id'], top_k=10)
        evaluator.evaluate_diversity(case['id'], top_k=10)
        print("\n")


def test_edge_cases():
    """
    특수 케이스 테스트
    """
    evaluator = EmbeddingEvaluator('../outputs/movie_embeddings_e5.pkl')
    
    print("="*80)
    print(" 특수 케이스 테스트")
    print("="*80)
    
    # 1. 애니메이션
    print("\n>>> 테스트 1: 애니메이션 (토이 스토리)")
    evaluator.find_similar_movies(1, top_k=10)
    
    # 2. SF 블록버스터
    print("\n>>> 테스트 2: SF 블록버스터 (매트릭스)")
    evaluator.find_similar_movies(2571, top_k=10)
    
    # 3. 로맨스
    print("\n>>> 테스트 3: 로맨스")
    # 타이타닉 (영화 ID를 찾아야 함)
    for mid, meta in zip(evaluator.movie_ids, evaluator.metadata):
        if '타이타닉' in meta['title']:
            print(f"  타이타닉 찾음: ID {mid}")
            evaluator.find_similar_movies(mid, top_k=10)
            break


def analyze_embedding_quality():
    """
    전체 임베딩 품질 분석
    """
    print("="*80)
    print(" 전체 임베딩 품질 분석")
    print("="*80)
    
    with open('../outputs/movie_embeddings_e5.pkl', 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    
    # 1. 임베딩 통계
    print("\n>>> 1. 임베딩 통계:")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  평균 norm: {np.mean(norms):.6f}")
    print(f"  표준편차: {np.std(norms):.6f}")
    print(f"  최소/최대: {np.min(norms):.6f} / {np.max(norms):.6f}")
    
    # 2. 유사도 분포
    print("\n>>> 2. 유사도 분포 (샘플 1000쌍):")
    sample_size = 1000
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    
    similarities = []
    for i in range(0, sample_size, 2):
        if i+1 < sample_size:
            sim = np.dot(embeddings[indices[i]], embeddings[indices[i+1]])
            similarities.append(sim)
    
    similarities = np.array(similarities)
    print(f"  평균 유사도: {np.mean(similarities):.4f}")
    print(f"  표준편차: {np.std(similarities):.4f}")
    print(f"  최소/최대: {np.min(similarities):.4f} / {np.max(similarities):.4f}")
    
    # 3. 유사도 구간 분포
    print("\n>>> 3. 유사도 구간 분포:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(similarities, bins=bins)
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.1f} ~ {bins[i+1]:.1f}: {hist[i]:4}개 ({hist[i]/len(similarities)*100:.1f}%)")


def main():
    """
    전체 평가 실행
    """
    try:
        # 1. 유명 영화 테스트
        test_popular_movies()
        
        # 2. 특수 케이스
        test_edge_cases()
        
        # 3. 전체 품질 분석
        analyze_embedding_quality()
        
        print("\n" + "="*80)
        print("✅ 정성 평가 완료!")
        print("="*80)
        print("\n평가 기준:")
        print("  ✓ 장르 일치도 > 70% → 좋음")
        print("  ✓ 유사도 > 0.7 → 매우 유사")
        print("  ✓ 유사도 0.5~0.7 → 적당히 유사")
        print("  ✓ 유사도 < 0.5 → 다름")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()