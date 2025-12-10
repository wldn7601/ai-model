import pickle
import numpy as np
import os

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
PKL_PATH = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_separate_e5.pkl')

class MovieVectorReader:
    def __init__(self, pkl_path):
        print(f"📂 데이터 로딩 중... ({pkl_path})")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"파일이 없습니다: {pkl_path}")
            
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
            
        # 데이터 분리
        self.embeddings = self.data['embeddings']       # 최종 결합 벡터
        self.movie_ids = self.data['movie_ids']         # 영화 ID 리스트
        self.metadata = self.data['metadata']           # 영화 정보 리스트
        
        # 개별 임베딩 (필요 시 사용)
        self.genre_vecs = self.data['genre_embeddings']
        self.tag_vecs = self.data['tag_embeddings']
        self.overview_vecs = self.data['overview_embeddings']
        
        # ★ 핵심: ID를 입력하면 배열의 인덱스(순서)를 찾아주는 맵 생성
        # 예: { 1: 0, 2: 1, 10: 2, ... } -> ID 1번 영화는 0번째 칸에 있다.
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        
        print(f"✅ 로드 완료: {len(self.movie_ids):,}개 영화")

    def get_vector_by_id(self, movie_id):
        """
        영화 ID로 벡터 가져오기
        """
        if movie_id not in self.id_to_idx:
            print(f"❌ ID {movie_id}를 찾을 수 없습니다.")
            return None
        
        idx = self.id_to_idx[movie_id]
        
        return {
            'final_vector': self.embeddings[idx],      # 최종 벡터
            'genre_vector': self.genre_vecs[idx],      # 장르 벡터
            'tag_vector': self.tag_vecs[idx],          # 태그 벡터
            'overview_vector': self.overview_vecs[idx],# 줄거리 벡터
            'meta': self.metadata[idx]
        }

    def search_by_title(self, keyword):
        """
        제목으로 영화 검색해서 정보 가져오기
        """
        results = []
        for idx, meta in enumerate(self.metadata):
            if keyword.lower() in meta['title'].lower():
                results.append(meta)
                
        if not results:
            print(f"❌ '{keyword}' 검색 결과가 없습니다.")
            return []
            
        print(f"🔍 '{keyword}' 검색 결과 ({len(results)}개):")
        for m in results[:5]: # 최대 5개만 출력
            print(f"   - [{m['movieId']}] {m['title']} (장르: {', '.join(m['genres'][:2])})")
            
        return results

# ==========================================
# 실행 예시
# ==========================================
def main():
    reader = MovieVectorReader(PKL_PATH)
    
    # 1. 제목으로 검색해보기
    print("\n--- [1] '토이 스토리' 검색 ---")
    found_movies = reader.search_by_title("토이 스토리")
    
    if found_movies:
        target_id = found_movies[0]['movieId'] # 첫 번째 검색 결과 선택
        
        # 2. 해당 ID의 벡터 가져오기
        print(f"\n--- [2] ID {target_id}의 벡터 추출 ---")
        result = reader.get_vector_by_id(target_id)
        
        if result:
            vec = result['final_vector']
            meta = result['meta']
            
            print(f"영화: {meta['title']}")
            print(f"최종 벡터 크기: {len(vec)}") # 1024
            print(f"벡터 앞부분 5개: {vec[:5]}")
            
            # 개별 벡터 확인 (Late Fusion의 장점)
            print(f"\n--- [3] 개별 요소 확인 ---")
            print(f"장르 벡터 크기: {len(result['genre_vector'])}")
            print(f"태그 벡터 크기: {len(result['tag_vector'])}")
            print(f"줄거리 벡터 크기: {len(result['overview_vector'])}")

if __name__ == "__main__":
    main()