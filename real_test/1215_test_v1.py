import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import faiss

"""

필터링 X

"""

class HybridRecommender:
    def __init__(
        self,
        sbert_embeddings_path: str,
        lightgcn_model_path: str,
        lightgcn_data_path: str,
        metadata_path: str,
        sbert_weight: float = 0.7,
        lightgcn_weight: float = 0.3,
        device: str = None
    ):
        """
        Args:
            sbert_embeddings_path: SBERT 임베딩 pkl 파일 경로
            lightgcn_model_path: LightGCN best 모델 체크포인트 경로
            lightgcn_data_path: LightGCN 전처리 데이터 폴더 경로
            metadata_path: 영화 메타데이터 CSV 파일 경로
            sbert_weight: SBERT 가중치 (default: 0.7)
            lightgcn_weight: LightGCN 가중치 (default: 0.3)
            device: 연산 디바이스
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sbert_weight = sbert_weight
        self.lightgcn_weight = lightgcn_weight
        
        # SBERT 데이터 로드
        self._load_sbert_data(sbert_embeddings_path)
        
        # LightGCN 데이터 및 모델 로드
        self._load_lightgcn_data(lightgcn_data_path)
        self._load_lightgcn_model(lightgcn_model_path)
        
        # 메타데이터 로드
        self._load_metadata(metadata_path)
        
        # 스케일러 초기화
        self.scaler = MinMaxScaler()

    def _load_metadata(self, path: str):
        """CSV 메타데이터 로드 및 딕셔너리 변환"""
        print(f"Loading metadata from {path}")
        try:
            df = pd.read_csv(path)
            
            # movieId가 float일 경우를 대비해 int로 변환
            if 'movieId' in df.columns:
                df['movieId'] = df['movieId'].astype(int)
            
            # 검색 속도 향상을 위해 DataFrame을 Dictionary로 변환
            # {movieId: {'title_ko': '...', ...}} 형태
            self.metadata_map = df.set_index('movieId').to_dict('index')
            print(f"Loaded metadata for {len(self.metadata_map)} movies")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata_map = {}
        
    def _load_sbert_data(self, embeddings_path: str):
        """SBERT 임베딩 데이터 로드"""
        print(f"Loading SBERT embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        self.sbert_movie_ids = data['movieId'].tolist()
        self.sbert_embeddings = data['embedding'].tolist()
        self.sbert_embeddings = np.array(self.sbert_embeddings).astype('float32')
        
        # movieId to index 매핑
        self.sbert_movie_to_idx = {mid: idx for idx, mid in enumerate(self.sbert_movie_ids)}
        
        print(f"Loaded {len(self.sbert_movie_ids)} SBERT embeddings")
        
    def _load_lightgcn_data(self, data_path: str):
        """LightGCN 전처리 데이터 로드"""
        data_path = Path(data_path)
        
        # id_mappings.pkl 로드
        mapping_path = data_path / 'id_mappings.pkl'
        print(f"Loading LightGCN mappings from {mapping_path}")
        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)
        
        # item = movie
        self.lightgcn_movie_to_idx = mappings['item2id']
        self.lightgcn_idx_to_movie = mappings['id2item']
        self.n_items = mappings['n_items']
        
        print(f"Loaded {len(self.lightgcn_movie_to_idx)} LightGCN movie mappings")
        
    def _load_lightgcn_model(self, model_path: str):
        """LightGCN 모델 로드"""
        print(f"Loading LightGCN model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # model_state_dict 구조에 맞게 안전하게 로드
        if isinstance(checkpoint, dict):
            if 'item_embeddings' in checkpoint:
                self.lightgcn_item_embeddings = checkpoint['item_embeddings'].cpu().numpy()
            elif 'model_state_dict' in checkpoint:
                # 사용자가 확인한 키 경로
                self.lightgcn_item_embeddings = checkpoint['model_state_dict']['item_embedding.weight'].cpu().numpy()
            else:
                # 혹시 다른 키일 경우를 대비해 Fallback
                if 'item_embedding.weight' in checkpoint:
                    self.lightgcn_item_embeddings = checkpoint['item_embedding.weight'].cpu().numpy()
        
        # 메타데이터 저장 (검증용)
        self.embedding_dim = checkpoint.get('embedding_dim', 'Unknown')
        self.n_layers = checkpoint.get('n_layers', 'Unknown')
        
        print(f"Loaded LightGCN embeddings: {self.lightgcn_item_embeddings.shape}")
        
    def _compute_sbert_scores(self, user_movie_ids: list) -> np.ndarray:
        """SBERT 기반 유사도 점수 계산"""
        # 사용자가 선호하는 영화들의 임베딩 추출
        user_embeddings = []
        valid_movie_ids = []
        
        for movie_id in user_movie_ids:
            if movie_id in self.sbert_movie_to_idx:
                idx = self.sbert_movie_to_idx[movie_id]
                user_embeddings.append(self.sbert_embeddings[idx])
                valid_movie_ids.append(movie_id)
            else:
                print(f"Warning: Movie {movie_id} not found in SBERT embeddings")
        
        if len(user_embeddings) == 0:
            raise ValueError("No valid movies found in SBERT embeddings")
        
        user_embeddings = np.array(user_embeddings)
        
        # 각 선호 영화에 대해 전체 영화와의 코사인 유사도 계산
        # L2 정규화
        user_embeddings_norm = user_embeddings / np.linalg.norm(
            user_embeddings, axis=1, keepdims=True
        )
        all_embeddings_norm = self.sbert_embeddings / np.linalg.norm(
            self.sbert_embeddings, axis=1, keepdims=True
        )
        
        # 코사인 유사도 계산 (각 선호 영화별)
        similarities = user_embeddings_norm @ all_embeddings_norm.T  # (n_user_movies, n_all_movies)
        
        # 평균 유사도 계산
        avg_similarities = np.mean(similarities, axis=0)
        
        return avg_similarities
    
    def _compute_lightgcn_scores(self, user_movie_ids: list) -> np.ndarray:
        """LightGCN 기반 점수 계산"""
        # 사용자가 선호하는 영화들의 임베딩 추출
        user_embeddings = []
        valid_movie_ids = []
        
        for movie_id in user_movie_ids:
            if movie_id in self.lightgcn_movie_to_idx:
                idx = self.lightgcn_movie_to_idx[movie_id]
                user_embeddings.append(self.lightgcn_item_embeddings[idx])
                valid_movie_ids.append(movie_id)
            else:
                print(f"Warning: Movie {movie_id} not found in LightGCN embeddings")
        
        if len(user_embeddings) == 0:
            raise ValueError("No valid movies found in LightGCN embeddings")
        
        user_embeddings = np.array(user_embeddings)
        
        # 선호 영화 임베딩의 평균 계산
        avg_user_embedding = np.mean(user_embeddings, axis=0, keepdims=True)
        
        # 전체 영화와의 내적 계산
        scores = avg_user_embedding @ self.lightgcn_item_embeddings.T
        scores = scores.squeeze()
        
        return scores
    
    def _align_scores(self, sbert_scores: np.ndarray, lightgcn_scores: np.ndarray) -> tuple:
        """
        SBERT와 LightGCN 점수를 정렬하고 정규화
        """
        # 두 모델 모두에서 사용 가능한 영화 찾기
        common_movie_ids = []
        aligned_sbert = []
        aligned_lightgcn = []
        
        for lightgcn_idx, movie_id in self.lightgcn_idx_to_movie.items():
            if movie_id in self.sbert_movie_to_idx:
                sbert_idx = self.sbert_movie_to_idx[movie_id]
                
                common_movie_ids.append(movie_id)
                aligned_sbert.append(sbert_scores[sbert_idx])
                aligned_lightgcn.append(lightgcn_scores[lightgcn_idx])
        
        aligned_sbert = np.array(aligned_sbert)
        aligned_lightgcn = np.array(aligned_lightgcn)
        
        print(f"Common movies between SBERT and LightGCN: {len(common_movie_ids)}")
        
        return aligned_sbert, aligned_lightgcn, common_movie_ids
    
    def recommend(
        self,
        user_movie_ids: list,
        top_k: int = 20,
        exclude_seen: bool = True
    ) -> list:
        """
        하이브리드 추천 실행
        """
        print(f"\nStarting hybrid recommendation for {len(user_movie_ids)} user movies")
        
        # 1. SBERT 점수 계산
        print("Computing SBERT scores...")
        sbert_scores = self._compute_sbert_scores(user_movie_ids)
        
        # 2. LightGCN 점수 계산
        print("Computing LightGCN scores...")
        lightgcn_scores = self._compute_lightgcn_scores(user_movie_ids)
        
        # 3. 점수 정렬
        print("Aligning scores...")
        aligned_sbert, aligned_lightgcn, common_movie_ids = self._align_scores(
            sbert_scores, lightgcn_scores
        )
        
        # 4. 정규화 (0~1 범위로)
        print("Normalizing scores...")
        normalized_sbert = self.scaler.fit_transform(aligned_sbert.reshape(-1, 1)).squeeze()
        normalized_lightgcn = self.scaler.fit_transform(aligned_lightgcn.reshape(-1, 1)).squeeze()
        
        # 5. 가중 평균
        final_scores = (
            self.sbert_weight * normalized_sbert +
            self.lightgcn_weight * normalized_lightgcn
        )
        
        # 6. 사용자가 본 영화 제외
        if exclude_seen:
            for movie_id in user_movie_ids:
                if movie_id in common_movie_ids:
                    idx = common_movie_ids.index(movie_id)
                    final_scores[idx] = -np.inf
        
        # 7. Top-K 추출
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # 결과 반환 시 메타데이터 결합
        recommendations = []
        for idx in top_indices:
            mid = common_movie_ids[idx]
            meta = self.metadata_map.get(mid, {})
            
            recommendations.append({
                'movie_id': mid,
                'hybrid_score': final_scores[idx],
                'sbert_score': normalized_sbert[idx],
                'lightgcn_score': normalized_lightgcn[idx],
                'title_ko': meta.get('title_ko', 'Unknown Title'),
                'genres': meta.get('genres', ''),
                'overview': meta.get('overview', ''),
                'poster_path': meta.get('poster_path', ''),
                'release_date': meta.get('release_date', ''),
                'runtime': meta.get('runtime', 0),
                'popularity': meta.get('popularity', 0),
                'adult': meta.get('adult', False),
            })
        
        return recommendations

# 실행 예시
if __name__ == "__main__":
    # 경로 설정
    SBERT_EMBEDDINGS_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl"
    LIGHTGCN_MODEL_PATH = "/home/ubuntu/ai-model/models/light_gcn/checkpoints/best_model.pt"
    LIGHTGCN_DATA_PATH = "/home/ubuntu/ai-model/models/light_gcn/data"
    METADATA_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/pre_final_movies_processed.csv"
    
    # 추천 시스템 초기화
    recommender = HybridRecommender(
        sbert_embeddings_path=SBERT_EMBEDDINGS_PATH,
        lightgcn_model_path=LIGHTGCN_MODEL_PATH,
        lightgcn_data_path=LIGHTGCN_DATA_PATH,
        metadata_path=METADATA_PATH,
        sbert_weight=0.7,
        lightgcn_weight=0.3
    )
    
    # 더미 사용자 데이터 (선호하는 영화 3개)
    user_preferred_movies = [1, 296, 356]  # 실제 movieId
    
    print("\n" + "="*120)
    print("USER PREFERRED MOVIES (INPUT)")
    print("="*120)
    print(f"{'ID':<6} | {'Title (KR)':<30} | {'Genres'}")
    print("-" * 120)
    
    # 사용자 영화 정보 출력
    for mid in user_preferred_movies:
        # metadata_map에서 정보 조회
        info = recommender.metadata_map.get(mid, {})
        title = str(info.get('title_ko', 'Unknown Title'))
        genres = str(info.get('genres', 'Unknown'))
        
        # 출력 포맷팅 (긴 제목 자르기)
        if len(title) > 28: title = title[:25] + "..."
        if len(genres) > 50: genres = genres[:47] + "..."
            
        print(f"{mid:<6} | {title:<30} | {genres}")

    # 추천 실행
    recommendations = recommender.recommend(
        user_movie_ids=user_preferred_movies,
        # top_k=20,
        top_k=100,
        exclude_seen=True
    )
    
    # 결과 출력
    print("\n" + "="*150)
    print("RECOMMENDATION RESULTS (OUTPUT)")
    print("="*150)
    print(f"SBERT weight: {recommender.sbert_weight}, LightGCN weight: {recommender.lightgcn_weight}")
    print("\nTop 100 Recommendations:")
    print("-" * 150)
    # [수정] 헤더에 Year 추가 (Runtime 옆)
    print(f"{'Rank':<4} | {'ID':<6} | {'Score':<6} | {'Title (KR)':<25} | {'Year':<4} | {'Runtime':<7} | {'Adult':<5} | {'Pop':<8} | {'Genres'}")
    print("-" * 150)
    
    for i, rec in enumerate(recommendations, 1):
        # 긴 제목과 장르 텍스트 자르기
        title = str(rec['title_ko'])
        if len(title) > 23: title = title[:20] + "..."
            
        genres = str(rec['genres'])
        if len(genres) > 30: genres = genres[:27] + "..."
        
        # 데이터 포맷팅
        runtime = str(rec['runtime'])
        adult = str(rec['adult'])
        popularity = float(rec['popularity'])
        
        # [추가] release_date에서 연도만 추출 (YYYY-MM-DD -> YYYY)
        release_date = str(rec.get('release_date', ''))
        year = release_date[:4] if len(release_date) >= 4 else "Unk"

        # [수정] 출력 포맷에 year 추가
        print(f"{i:<4} | {rec['movie_id']:<6} | {rec['hybrid_score']:.4f} | {title:<25} | {year:<4} | {runtime:<7} | {adult:<5} | {popularity:<8.2f} | {genres}")