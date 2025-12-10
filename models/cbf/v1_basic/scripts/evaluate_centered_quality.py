
"""
movie_embeddings_centered 모델 평가
# fix_vectors_centering.py 실행 후 평가
# 영화 추천/검색 (find_similar_movies): **전체 데이터(100%)**를 사용합니다.
# 품질 분석 통계 (analyze_quality): **샘플 데이터(1,000개)**를 사용합니다.

"""
import pickle
import numpy as np
import os
from typing import List, Dict

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
# Late Fusion으로 만든 파일 경로
PKL_PATH = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_centered.pkl')

# 가중치 설정 (이 비율로 합쳐서 평가합니다)
W_GENRE = 0.2
W_TAG = 0.7
W_OVERVIEW = 0.1

class EmbeddingEvaluator:
    """
    Late Fusion 임베딩 품질 평가
    """
    def __init__(self, embeddings_path: str):
        print(f">>> [Init] 데이터 로딩 중... ({embeddings_path})")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        # 1. 개별 임베딩 로드
        self.genre_vecs = data['genre_embeddings']
        self.tag_vecs = data['tag_embeddings']
        self.overview_vecs = data['overview_embeddings']
        
        self.movie_ids = data['movie_ids']
        self.metadata = data['metadata']
        
        # ID 매핑
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        
        # 2. 벡터 결합 (가중치 적용)
        print(f"   ✓ 벡터 결합 중... (G:{W_GENRE} / T:{W_TAG} / O:{W_OVERVIEW})")
        combined = (
            self.genre_vecs * W_GENRE +
            self.tag_vecs * W_TAG +
            self.overview_vecs * W_OVERVIEW
        )
        
        # 정규화
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        self.embeddings = combined / norms
        
        print(f"   ✓ 영화 수: {len(self.movie_ids):,}개")
        print(f"   ✓ 임베딩 준비 완료")

    def find_similar_movies(self, movie_id: int, top_k: int = 10, show_details: bool = True) -> List[Dict]:
        if movie_id not in self.id_to_idx:
            print(f"❌ 영화 ID {movie_id} 없음")
            return []
        
        idx = self.id_to_idx[movie_id]
        query_vec = self.embeddings[idx]
        
        # 유사도 계산
        similarities = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1] # 자기 자신 제외
        
        results = []
        for rank, i in enumerate(top_indices, 1):
            meta = self.metadata[i]
            results.append({
                'rank': rank,
                'title': meta['title'],
                'similarity': float(similarities[i]),
                'genres': meta['genres'],
                'runtime': meta.get('runtime', 0)
            })
        
        if show_details:
            query_meta = self.metadata[idx]
            print(f"\n{'='*80}")
            print(f"기준 영화: {query_meta['title']}")
            print(f"장르: {', '.join(query_meta['genres'])}")
            print(f"{'='*80}")
            print(f"{'순위':<4} {'제목':<35} {'유사도':<8} {'장르'}")
            print("-"*80)
            
            for rec in results:
                title = rec['title'][:33]
                genres = ', '.join(rec['genres'][:2])[:25]
                print(f"{rec['rank']:<4} {title:<35} {rec['similarity']:.4f}   {genres}")
        
        return results

    def analyze_quality(self):
        print("\n>>> 전체 품질 분석 (유사도 분포)")
        
        # 1000개 샘플링
        idx = np.random.choice(len(self.embeddings), 1000, replace=False)
        sample = self.embeddings[idx]
        
        sims = np.dot(sample, sample.T)
        sims = sims[~np.eye(sims.shape[0], dtype=bool)] # 대각선 제외
        
        avg = np.mean(sims)
        print(f"   📊 평균 유사도: {avg:.4f}")
        print(f"   📉 최소/최대: {np.min(sims):.4f} / {np.max(sims):.4f}")
        
        # 구간별 분포
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(sims, bins=bins)
        print("\n   [유사도 구간 분포]")
        for i in range(len(bins)-1):
            print(f"   {bins[i]:.1f} ~ {bins[i+1]:.1f}: {hist[i]:4}개 ({hist[i]/len(sims)*100:.1f}%)")

def main():
    try:
        evaluator = EmbeddingEvaluator(PKL_PATH)
        
        # 1. 유명 영화 테스트
        test_cases = [
            {'id': 1, 'name': '토이 스토리'},
            {'id': 2571, 'name': '매트릭스'},
            {'id': 1721, 'name': '타이타닉'} # ID 확인 필요
        ]
        
        # 타이타닉 ID 찾기 (없으면 검색)
        if 1721 not in evaluator.id_to_idx:
            for m in evaluator.metadata:
                if '타이타닉' in m['title']:
                    test_cases[2]['id'] = m['movieId']
                    break

        for case in test_cases:
            evaluator.find_similar_movies(case['id'])
            
        # 2. 전체 품질 분석
        evaluator.analyze_quality()
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()