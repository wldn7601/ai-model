import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple
from itertools import combinations
from math import comb
import time
import ast

"""
Hybrid Recommender with Genre & Runtime Filtering
- SBERT (70%) + LightGCN (30%)
- 장르, 런타임 필터링 지원
- 240분 미만: 단일 영화 추천 (Track A, B)
- 240분 이상: 영화 조합 추천
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
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sbert_weight = sbert_weight
        self.lightgcn_weight = lightgcn_weight
        
        print("Initializing Hybrid Recommender...")
        
        # 1. 데이터 로드
        self._load_sbert_data(sbert_embeddings_path)
        self._load_lightgcn_data(lightgcn_data_path)
        self._load_lightgcn_model(lightgcn_model_path)
        self._load_metadata(metadata_path)
        
        # 2. Pre-alignment
        print("Pre-aligning models for fast inference...")
        
        common_ids = set(self.sbert_movie_to_idx.keys()) & set(self.lightgcn_movie_to_idx.keys())
        self.common_movie_ids = sorted(list(common_ids))
        
        self.target_sbert_matrix = []
        self.target_lightgcn_matrix = []
        
        for mid in self.common_movie_ids:
            s_idx = self.sbert_movie_to_idx[mid]
            self.target_sbert_matrix.append(self.sbert_embeddings[s_idx])
            
            l_idx = self.lightgcn_movie_to_idx[mid]
            self.target_lightgcn_matrix.append(self.lightgcn_item_embeddings[l_idx])
        
        self.target_sbert_matrix = np.array(self.target_sbert_matrix)
        self.target_lightgcn_matrix = np.array(self.target_lightgcn_matrix)
        
        self.target_sbert_norm = self.target_sbert_matrix / (
            np.linalg.norm(self.target_sbert_matrix, axis=1, keepdims=True) + 1e-10
        )
        
        print(f"Pre-alignment complete. Target movies: {len(self.common_movie_ids)}")
        
        self.scaler = MinMaxScaler()
        self.recommendation_history = []

    def _load_sbert_data(self, embeddings_path: str):
        print(f"Loading SBERT embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        self.sbert_movie_ids = data['tmdbId'].tolist()
        self.sbert_embeddings = np.array(data['embedding'].tolist(), dtype='float32')
        self.sbert_movie_to_idx = {mid: idx for idx, mid in enumerate(self.sbert_movie_ids)}
        
        print(f"  SBERT movies: {len(self.sbert_movie_ids):,}")
    
    def _load_lightgcn_data(self, data_path: str):
        data_path = Path(data_path)
        with open(data_path / 'id_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        self.lightgcn_movie_to_idx = mappings['tmdb2id']
        self.lightgcn_idx_to_movie = mappings['id2tmdb']
        
        print(f"  LightGCN movies: {len(self.lightgcn_movie_to_idx):,}")
    
    def _load_lightgcn_model(self, model_path: str):
        print(f"Loading LightGCN model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.lightgcn_item_embeddings = checkpoint['model_state_dict']['item_embedding.weight'].cpu().numpy()
            elif 'item_embeddings' in checkpoint:
                self.lightgcn_item_embeddings = checkpoint['item_embeddings'].cpu().numpy()
            else:
                self.lightgcn_item_embeddings = checkpoint['item_embedding.weight'].cpu().numpy()
    
    def _load_metadata(self, path: str):
        print(f"Loading metadata from {path}")
        df = pd.read_csv(path)
        
        # tmdb_id 컬럼 확인 및 중복 제거
        if 'tmdb_id' in df.columns:
            df['tmdb_id'] = df['tmdb_id'].astype(int)
            
            # 중복 확인
            duplicates = df[df.duplicated(subset=['tmdb_id'], keep=False)]
            if not duplicates.empty:
                print(f"  Warning: Found {len(duplicates)} duplicate tmdb_ids")
                print(f"  Keeping first occurrence for each duplicate")
                df = df.drop_duplicates(subset=['tmdb_id'], keep='first')
            
            self.metadata_map = df.set_index('tmdb_id').to_dict('index')
        else:
            print("Warning: tmdb_id column not found")
            self.metadata_map = {}
        
        # 장르 리스트 추출
        all_genres = set()
        for movie_data in self.metadata_map.values():
            genres = movie_data.get('genres', '')
            if isinstance(genres, str) and genres:
                try:
                    # 문자열이 리스트 형태인 경우 파싱
                    if genres.startswith('['):
                        genre_list = ast.literal_eval(genres)
                    else:
                        genre_list = [g.strip() for g in genres.split(',')]
                    
                    # 공백 제거 및 유효한 장르만 추가
                    genre_list = [g for g in genre_list if g.strip()]  # 추가
                    all_genres.update(genre_list)
                except:
                    pass
        
        # 공백 제거 및 정렬
        self.all_genres = sorted([g for g in all_genres if g.strip()])  # 수정
        
        print(f"  Metadata loaded: {len(self.metadata_map):,} movies")
        print(f"  Available genres: {len(self.all_genres)}")

    def _get_movie_runtime(self, movie_id: int) -> int:
        """영화 런타임 반환 (분)"""
        meta = self.metadata_map.get(movie_id, {})
        runtime = meta.get('runtime', 0)
        try:
            return int(float(runtime)) if runtime else 0
        except:
            return 0

    def _apply_filters(
        self,
        movie_ids: List[int],
        preferred_genres: Optional[List[str]] = None,
        max_runtime: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        필터링 적용 (장르, 런타임)
        
        Returns:
            (filtered_ids, filtered_indices)
        """
        filtered_indices = []
        filtered_ids = []
        
        for i, movie_id in enumerate(movie_ids):
            meta = self.metadata_map.get(movie_id, {})
            if not meta:
                continue
            
            # 1. 런타임 필터링
            if max_runtime is not None:
                runtime = meta.get('runtime', 0)
                try:
                    runtime = float(runtime) if runtime else 0
                except:
                    runtime = 0
                
                if runtime <= 0 or runtime > max_runtime:
                    continue
            
            # 2. 장르 필터링
            if preferred_genres:
                genres = meta.get('genres', '')
                if not genres:
                    continue
                
                # 문자열 파싱
                try:
                    if isinstance(genres, str):
                        if genres.startswith('['):
                            genre_list = ast.literal_eval(genres)
                        else:
                            genre_list = [g.strip() for g in genres.split(',')]
                    else:
                        genre_list = []
                    
                    # 선호 장르와 교집합 확인
                    if not any(g in genre_list for g in preferred_genres):
                        continue
                except:
                    continue
            
            filtered_indices.append(i)
            filtered_ids.append(movie_id)
        
        return filtered_ids, filtered_indices

    def _find_movie_combinations(
        self,
        movie_ids: List[int],
        scores: np.ndarray,
        available_time: int,
        top_k: int = 1
    ) -> List[dict]:
        """
        시간에 맞는 영화 조합 찾기 (Knapsack)
        """
        print(f"\nFinding movie combinations...")
        print(f"  Available time: {available_time} min")
        print(f"  Candidate movies: {len(movie_ids)}")
        
        # 영화 데이터 준비
        movie_data = []
        for i, mid in enumerate(movie_ids):
            runtime = self._get_movie_runtime(mid)
            if runtime > 0 and runtime <= available_time:
                movie_data.append({
                    'id': mid,
                    'runtime': runtime,
                    'score': scores[i]
                })
        
        if not movie_data:
            print("  No valid movies for combination")
            return []
        
        print(f"  Valid movies after runtime filter: {len(movie_data)}")
        
        # 점수 순 정렬
        movie_data.sort(key=lambda x: x['score'], reverse=True)
        
        # 동적 후보 수 조절
        max_combinations_limit = 1_000_000
        max_candidates = min(len(movie_data), 60)
        
        for n in range(20, min(len(movie_data), 100)):
            max_combo_size = min(5, n // 3)
            total_combos = sum(comb(n, k) for k in range(2, max_combo_size + 1))
            if total_combos > max_combinations_limit:
                max_candidates = n - 1
                break
        
        movie_data = movie_data[:max_candidates]
        print(f"  Using top {len(movie_data)} candidates")
        
        # 조합 생성 (2~5개 영화)
        valid_combinations = []
        time_tolerance = 30  # ±30분 허용
        
        for combo_size in range(2, min(6, len(movie_data) + 1)):
            for combo in combinations(movie_data, combo_size):
                total_runtime = sum(m['runtime'] for m in combo)
                
                if available_time - time_tolerance <= total_runtime <= available_time + time_tolerance:
                    avg_score = np.mean([m['score'] for m in combo])
                    valid_combinations.append({
                        'movies': [m['id'] for m in combo],
                        'total_runtime': total_runtime,
                        'avg_score': avg_score
                    })
                    
                    if len(valid_combinations) >= 1:
                        break
            
            if len(valid_combinations) >= 1:
                break
        
        print(f"  Found {len(valid_combinations)} valid combination(s)")
        
        if not valid_combinations:
            return []
        
        valid_combinations.sort(key=lambda x: x['avg_score'], reverse=True)
        return valid_combinations[:top_k]

    def recommend(
        self,
        user_movie_ids: List[int],
        available_time: int,
        top_k: int = 20,
        exclude_seen: bool = True,
        preferred_genres: Optional[List[str]] = None
    ) -> Tuple[str, dict]:
        """
        하이브리드 추천
        
        Args:
            user_movie_ids: 사용자가 본 영화 tmdb_id 리스트
            available_time: 이용 가능 시간 (분)
            top_k: 추천할 영화 개수
            exclude_seen: 본 영화 제외 여부
            preferred_genres: 선호 장르 리스트
        
        Returns:
            (recommendation_type, result)
        """
        print(f"\nStarting hybrid recommendation...")
        print(f"Available time: {available_time} min")
        
        start_time = time.time()
        
        # 1. 사용자 프로필 생성
        user_sbert_vecs = []
        for mid in user_movie_ids:
            if mid in self.sbert_movie_to_idx:
                user_sbert_vecs.append(self.sbert_embeddings[self.sbert_movie_to_idx[mid]])
        
        if not user_sbert_vecs:
            return 'single', {'recommendations': {'track_a': [], 'track_b': []}}
        
        user_sbert_profile = np.mean(user_sbert_vecs, axis=0)
        user_sbert_profile = user_sbert_profile / (np.linalg.norm(user_sbert_profile) + 1e-10)
        
        user_gcn_vecs = []
        for mid in user_movie_ids:
            if mid in self.lightgcn_movie_to_idx:
                user_gcn_vecs.append(self.lightgcn_item_embeddings[self.lightgcn_movie_to_idx[mid]])
        
        if not user_gcn_vecs:
            return 'single', {'recommendations': {'track_a': [], 'track_b': []}}
        
        user_gcn_profile = np.mean(user_gcn_vecs, axis=0)
        
        # 2. 전체 점수 계산
        sbert_scores = self.target_sbert_norm @ user_sbert_profile
        lightgcn_scores = self.target_lightgcn_matrix @ user_gcn_profile
        
        # 3. 추천 타입 결정
        recommendation_type = 'combination' if available_time >= 240 else 'single'
        max_runtime = None if recommendation_type == 'combination' else available_time
        
        # 4. Track A 필터링 (장르 적용)
        filtered_ids_a, filtered_indices_a = self._apply_filters(
            self.common_movie_ids, preferred_genres, max_runtime
        )
        
        # 5. Track B 필터링 (장르 무시)
        filtered_ids_b, filtered_indices_b = self._apply_filters(
            self.common_movie_ids, None, max_runtime
        )
        
        if recommendation_type == 'single':
            # === 단일 영화 추천 ===
            
            # Track A
            if filtered_ids_a:
                filtered_sbert_a = sbert_scores[filtered_indices_a]
                filtered_lightgcn_a = lightgcn_scores[filtered_indices_a]
                
                norm_sbert_a = self.scaler.fit_transform(filtered_sbert_a.reshape(-1, 1)).squeeze()
                norm_lightgcn_a = self.scaler.fit_transform(filtered_lightgcn_a.reshape(-1, 1)).squeeze()
                
                final_scores_a = self.sbert_weight * norm_sbert_a + self.lightgcn_weight * norm_lightgcn_a
                
                if exclude_seen:
                    for i, mid in enumerate(filtered_ids_a):
                        if mid in user_movie_ids:
                            final_scores_a[i] = -np.inf
                
                top_indices_a = np.argsort(final_scores_a)[::-1][:3]
                track_a = self._build_recommendations(filtered_ids_a, final_scores_a, top_indices_a)
            else:
                track_a = []
            
            # Track B: 장르 다양성 확보
            if filtered_ids_b:
                filtered_sbert_b = sbert_scores[filtered_indices_b]
                filtered_lightgcn_b = lightgcn_scores[filtered_indices_b]
                
                norm_sbert_b = self.scaler.fit_transform(filtered_sbert_b.reshape(-1, 1)).squeeze()
                norm_lightgcn_b = self.scaler.fit_transform(filtered_lightgcn_b.reshape(-1, 1)).squeeze()
                
                # Track B는 협업 필터링 강화
                final_scores_b = 0.4 * norm_sbert_b + 0.6 * norm_lightgcn_b
                
                if exclude_seen:
                    for i, mid in enumerate(filtered_ids_b):
                        if mid in user_movie_ids:
                            final_scores_b[i] = -np.inf
                
                # Track A 영화 제외
                track_a_ids = [m['tmdb_id'] for m in track_a]
                for i, mid in enumerate(filtered_ids_b):
                    if mid in track_a_ids:
                        final_scores_b[i] = -np.inf
                
                # 이전 추천 이력 제외 (추가)
                for i, mid in enumerate(filtered_ids_b):
                    if mid in self.recommendation_history[-9:]:  # 최근 9개 제외
                        final_scores_b[i] = -np.inf
                
                # 장르 다양성 부스팅
                track_a_genres = set()
                if preferred_genres:
                    track_a_genres.update(preferred_genres)
                
                for i, mid in enumerate(filtered_ids_b):
                    if final_scores_b[i] == -np.inf:  # 이미 제외된 영화는 스킵
                        continue
                        
                    meta = self.metadata_map.get(mid, {})
                    genres = meta.get('genres', '')
                    
                    try:
                        if isinstance(genres, str) and genres.startswith('['):
                            genre_list = ast.literal_eval(genres)
                        else:
                            genre_list = [g.strip() for g in str(genres).split(',')]
                        
                        if track_a_genres and not any(g in track_a_genres for g in genre_list):
                            final_scores_b[i] *= 1.3
                    except:
                        pass
                
                # 상위 10개 중 랜덤 3개 선택 (추가)
                valid_indices = [i for i, score in enumerate(final_scores_b) if score != -np.inf]
                if len(valid_indices) >= 10:
                    top_10_indices = sorted(valid_indices, key=lambda i: final_scores_b[i], reverse=True)[:10]
                    selected_indices = np.random.choice(top_10_indices, size=min(3, len(top_10_indices)), replace=False)
                elif len(valid_indices) >= 3:
                    top_indices = sorted(valid_indices, key=lambda i: final_scores_b[i], reverse=True)[:3]
                    selected_indices = top_indices
                else:
                    selected_indices = valid_indices
                
                track_b = self._build_recommendations(filtered_ids_b, final_scores_b, selected_indices)
                
                # 추천 이력에 추가 (추가)
                for rec in track_b:
                    self.recommendation_history.append(rec['tmdb_id'])
            else:
                track_b = []
            
            result = {
                'recommendations': {
                    'track_a': {
                        'label': '선호 장르 맞춤 추천',
                        'movies': track_a
                    },
                    'track_b': {
                        'label': '장르 확장 추천',
                        'movies': track_b
                    }
                },
                'elapsed_time': time.time() - start_time
            }
            
            return recommendation_type, result
        
        else:
            # === 조합 추천 ===
            
            # Track A: 장르 적용 조합
            if filtered_ids_a:
                filtered_sbert_a = sbert_scores[filtered_indices_a]
                filtered_lightgcn_a = lightgcn_scores[filtered_indices_a]
                
                norm_sbert_a = self.scaler.fit_transform(filtered_sbert_a.reshape(-1, 1)).squeeze()
                norm_lightgcn_a = self.scaler.fit_transform(filtered_lightgcn_a.reshape(-1, 1)).squeeze()
                
                final_scores_a = self.sbert_weight * norm_sbert_a + self.lightgcn_weight * norm_lightgcn_a
                
                if exclude_seen:
                    for i, mid in enumerate(filtered_ids_a):
                        if mid in user_movie_ids:
                            final_scores_a[i] = -np.inf
                
                combination_a = self._find_movie_combinations(
                    filtered_ids_a, final_scores_a, available_time, top_k=1
                )
                
                if combination_a:
                    combo = combination_a[0]
                    combo_movies = []
                    for mid in combo['movies']:
                        meta = self.metadata_map.get(mid, {})
                        combo_movies.append({
                            'tmdb_id': mid,
                            'title': meta.get('title', 'Unknown'),
                            'runtime': meta.get('runtime', 0),
                            'genres': meta.get('genres', ''),
                            'overview': meta.get('overview', ''),
                            'release_date': meta.get('release_date', '')
                        })
                    
                    track_a_combo = {
                        'combination_score': combo['avg_score'],
                        'total_runtime': combo['total_runtime'],
                        'movies': combo_movies
                    }
                else:
                    track_a_combo = None
            else:
                track_a_combo = None
            
            # Track B: 장르 무시 조합 (수정)
            if filtered_ids_b:
                filtered_sbert_b = sbert_scores[filtered_indices_b]
                filtered_lightgcn_b = lightgcn_scores[filtered_indices_b]
                
                norm_sbert_b = self.scaler.fit_transform(filtered_sbert_b.reshape(-1, 1)).squeeze()
                norm_lightgcn_b = self.scaler.fit_transform(filtered_lightgcn_b.reshape(-1, 1)).squeeze()
                
                # Track B는 협업 필터링 강화
                final_scores_b = 0.4 * norm_sbert_b + 0.6 * norm_lightgcn_b
                
                if exclude_seen:
                    for i, mid in enumerate(filtered_ids_b):
                        if mid in user_movie_ids:
                            final_scores_b[i] = -np.inf
                
                # Track A 조합 영화 제외
                exclude_ids = []
                if track_a_combo:
                    exclude_ids = [m['tmdb_id'] for m in track_a_combo['movies']]
                
                for i, mid in enumerate(filtered_ids_b):
                    if mid in exclude_ids:
                        final_scores_b[i] = -np.inf
                
                # 이전 추천 이력 제외
                for i, mid in enumerate(filtered_ids_b):
                    if mid in self.recommendation_history[-9:]:
                        final_scores_b[i] = -np.inf
                
                # Track B도 조합으로 추천 (변경)
                combination_b = self._find_movie_combinations(
                    filtered_ids_b, final_scores_b, available_time, top_k=1
                )
                
                if combination_b:
                    combo = combination_b[0]
                    combo_movies = []
                    for mid in combo['movies']:
                        meta = self.metadata_map.get(mid, {})
                        combo_movies.append({
                            'tmdb_id': mid,
                            'title': meta.get('title', 'Unknown'),
                            'runtime': meta.get('runtime', 0),
                            'genres': meta.get('genres', ''),
                            'overview': meta.get('overview', ''),
                            'release_date': meta.get('release_date', '')
                        })
                    
                    track_b_combo = {
                        'combination_score': combo['avg_score'],
                        'total_runtime': combo['total_runtime'],
                        'movies': combo_movies
                    }
                    
                    # 추천 이력에 추가
                    for movie in combo_movies:
                        self.recommendation_history.append(movie['tmdb_id'])
                else:
                    track_b_combo = None
            else:
                track_b_combo = None
            
            result = {
                'recommendations': {
                    'track_a': {
                        'label': '선호 장르 영화 조합',
                        'combination': track_a_combo
                    },
                    'track_b': {
                        'label': '장르 확장 영화 조합',  # 레이블 변경
                        'combination': track_b_combo  # movies → combination
                    }
                },
                'elapsed_time': time.time() - start_time
            }
            
            return recommendation_type, result

    def _build_recommendations(self, movie_ids, scores, indices):
        """추천 결과 생성"""
        recommendations = []
        for idx in indices:
            mid = movie_ids[idx]
            meta = self.metadata_map.get(mid, {})
            recommendations.append({
                'tmdb_id': mid,
                'hybrid_score': float(scores[idx]),
                'title': meta.get('title', 'Unknown'),
                'overview': meta.get('overview', ''),
                'runtime': meta.get('runtime', 0),
                'genres': meta.get('genres', ''),
                'release_date': meta.get('release_date', '')
            })
        return recommendations


# ============================================================
# 사용자 입력 함수
# ============================================================
def get_user_input(recommender: HybridRecommender):
    """사용자 필터 입력"""
    
    print("\n" + "="*80)
    print("FILTER SELECTION")
    print("="*80)
    
    # 1. 시간 입력
    print("\n[1] 이용 가능 시간 입력")
    print("-" * 80)
    print("영화를 볼 수 있는 시간을 분 단위로 입력하세요.")
    print("※ 240분 이상: 영화 조합 추천")
    
    while True:
        time_input = input("\n시간(분): ").strip()
        try:
            available_time = int(time_input)
            if available_time > 0:
                break
            else:
                print("양수를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    # 2. 장르 선택
    print("\n[2] 선호 장르 선택")
    print("-" * 80)
    for i, genre in enumerate(recommender.all_genres, 1):
        print(f"{i:2d}. {genre}")
    
    genre_input = input("\n장르 번호 입력 (쉼표로 구분, 엔터로 스킵): ").strip()
    
    selected_genres = []
    if genre_input:
        try:
            genre_indices = [int(x.strip()) for x in genre_input.split(',')]
            selected_genres = [
                recommender.all_genres[i-1] 
                for i in genre_indices 
                if 1 <= i <= len(recommender.all_genres)
            ]
        except (ValueError, IndexError):
            print("잘못된 입력입니다. 장르 필터를 건너뜁니다.")
    
    print("\n" + "="*80)
    print("선택된 필터:")
    print(f"시간: {available_time}분")
    print(f"장르: {selected_genres if selected_genres else '전체'}")
    print("="*80)
    
    return {
        'available_time': available_time,
        'preferred_genres': selected_genres if selected_genres else None
    }


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    SBERT_EMBEDDINGS_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl"
    LIGHTGCN_MODEL_PATH = "/home/ubuntu/ai-model/models/light_gcn/checkpoints/best_model.pt"
    LIGHTGCN_DATA_PATH = "/home/ubuntu/ai-model/models/light_gcn/data"
    METADATA_PATH = "/home/ubuntu/ai-model/datas/data/insert_movies_updated.csv"
    
    print("\n" + "="*80)
    print("INITIALIZING HYBRID RECOMMENDER")
    print("="*80)
    
    recommender = HybridRecommender(
        sbert_embeddings_path=SBERT_EMBEDDINGS_PATH,
        lightgcn_model_path=LIGHTGCN_MODEL_PATH,
        lightgcn_data_path=LIGHTGCN_DATA_PATH,
        metadata_path=METADATA_PATH,
        sbert_weight=0.7,
        lightgcn_weight=0.3
    )
    
    print("\n" + "="*80)
    print("INITIALIZATION COMPLETE")
    print("="*80)
    
    # 테스트 사용자
    # user_movies = [862, 8844, 15602]
    user_movies = [854, 138843]
    
    while True:
        print("\n" + "="*80)
        print("USER INPUT MOVIES")
        print("="*80)
        for mid in user_movies:
            meta = recommender.metadata_map.get(mid, {})
            print(f"  {mid}: {meta.get('title', 'Unknown')}")
        
        # 첫 실행 체크 (수정)
        if hasattr(recommender, '_last_filters'):
            # 두 번째 이후 실행
            print("\n1. 새로운 조건으로 추천받기")
            print("2. 같은 조건으로 다른 영화 추천받기")
            mode_choice = input("선택 (1/2): ").strip()
            
            if mode_choice == "2":
                # 이전 필터 재사용
                filters = recommender._last_filters
                print("\n이전 조건으로 추천합니다.")
                print(f"시간: {filters['available_time']}분")
                print(f"장르: {filters['preferred_genres'] if filters['preferred_genres'] else '전체'}")
            else:
                # 새로운 필터 입력
                filters = get_user_input(recommender)
                recommender._last_filters = filters
        else:
            # 첫 실행 - 바로 필터 입력
            filters = get_user_input(recommender)
            recommender._last_filters = filters
        
        # 추천 실행
        rec_type, result = recommender.recommend(
            user_movie_ids=user_movies,
            top_k=20,
            exclude_seen=True,
            **filters
        )
        
        # 결과 출력
        print("\n" + "="*160)
        print(f"RECOMMENDATIONS ({'SINGLE' if rec_type == 'single' else 'COMBINATION'})")
        print(f"Elapsed: {result['elapsed_time']:.2f}s")
        print("="*160)
        
        if rec_type == 'single':
            # Track A
            track_a = result['recommendations']['track_a']
            print(f"\n[{track_a['label']}]")
            print("-" * 160)
            print(f"{'Rank':<4} | {'ID':<8} | {'Score':<6} | {'Title':<30} | {'Runtime':<7} | {'Genres'}")
            print("-" * 160)
            
            for i, rec in enumerate(track_a['movies'], 1):
                title = rec['title']
                if len(title) > 28:
                    title = title[:25] + "..."
                
                runtime = rec.get('runtime', 0)
                runtime_str = f"{runtime}분" if runtime else "N/A"
                
                # 장르 파싱
                genres = rec.get('genres', '')
                try:
                    if isinstance(genres, str) and genres.startswith('['):
                        genre_list = ast.literal_eval(genres)
                        genres_str = ', '.join(genre_list[:3])  # 최대 3개
                        if len(genre_list) > 3:
                            genres_str += f" +{len(genre_list)-3}"
                    else:
                        genres_str = str(genres)[:30]
                except:
                    genres_str = str(genres)[:30]
                
                if len(genres_str) > 30:
                    genres_str = genres_str[:27] + "..."
                
                print(f"{i:<4} | {rec['tmdb_id']:<8} | {rec['hybrid_score']:.4f} | {title:<30} | {runtime_str:<7} | {genres_str}")
                
                # 줄거리
                overview = rec['overview']
                if len(overview) > 120:
                    overview = overview[:117] + "..."
                print(f"       → {overview}")
                print()
            
            # Track B
            track_b = result['recommendations']['track_b']
            print(f"\n[{track_b['label']}]")
            print("-" * 160)
            print(f"{'Rank':<4} | {'ID':<8} | {'Score':<6} | {'Title':<30} | {'Runtime':<7} | {'Genres'}")
            print("-" * 160)
            
            for i, rec in enumerate(track_b['movies'], 1):
                title = rec['title']
                if len(title) > 28:
                    title = title[:25] + "..."
                
                runtime = rec.get('runtime', 0)
                runtime_str = f"{runtime}분" if runtime else "N/A"
                
                # 장르 파싱
                genres = rec.get('genres', '')
                try:
                    if isinstance(genres, str) and genres.startswith('['):
                        genre_list = ast.literal_eval(genres)
                        genres_str = ', '.join(genre_list[:3])
                        if len(genre_list) > 3:
                            genres_str += f" +{len(genre_list)-3}"
                    else:
                        genres_str = str(genres)[:30]
                except:
                    genres_str = str(genres)[:30]
                
                if len(genres_str) > 30:
                    genres_str = genres_str[:27] + "..."
                
                print(f"{i:<4} | {rec['tmdb_id']:<8} | {rec['hybrid_score']:.4f} | {title:<30} | {runtime_str:<7} | {genres_str}")
                
                # 줄거리
                overview = rec['overview']
                if len(overview) > 120:
                    overview = overview[:117] + "..."
                print(f"       → {overview}")
                print()
        
        else:
            # Track A: 조합
            track_a = result['recommendations']['track_a']
            print(f"\n[{track_a['label']}]")
            print("-" * 160)
            
            if track_a['combination']:
                combo = track_a['combination']
                print(f"Total Runtime: {combo['total_runtime']}분 | Score: {combo['combination_score']:.4f}")
                print("-" * 160)
                print(f"{'#':<2} | {'ID':<8} | {'Year':<4} | {'Title':<33} | {'Runtime':<7} | {'Genres'}")
                print("-" * 160)
                
                for j, movie in enumerate(combo['movies'], 1):
                    title = movie['title']
                    if len(title) > 31:
                        title = title[:28] + "..."
                    
                    release_date = movie.get('release_date', '')
                    year = release_date[:4] if release_date else 'N/A'
                    
                    runtime = movie.get('runtime', 0)
                    runtime_str = f"{runtime}분"
                    
                    genres = movie.get('genres', '')
                    try:
                        if isinstance(genres, str) and genres.startswith('['):
                            genre_list = ast.literal_eval(genres)
                            genres_str = ', '.join(genre_list[:3])
                            if len(genre_list) > 3:
                                genres_str += f" +{len(genre_list)-3}"
                        else:
                            genres_str = str(genres)[:40]
                    except:
                        genres_str = str(genres)[:40]
                    
                    if len(genres_str) > 40:
                        genres_str = genres_str[:37] + "..."
                    
                    print(f"{j:<2} | {movie['tmdb_id']:<8} | {year:<4} | {title:<33} | {runtime_str:<7} | {genres_str}")
                    
                    overview = movie.get('overview', '')
                    if len(overview) > 100:
                        overview = overview[:97] + "..."
                    if overview:
                        print(f"     → {overview}")
                    print()
            else:
                print("조건에 맞는 조합을 찾지 못했습니다.")
            
            # Track B: 조합 (수정)
            track_b = result['recommendations']['track_b']
            print(f"\n[{track_b['label']}]")
            print("-" * 160)
            
            if track_b['combination']:
                combo = track_b['combination']
                print(f"Total Runtime: {combo['total_runtime']}분 | Score: {combo['combination_score']:.4f}")
                print("-" * 160)
                print(f"{'#':<2} | {'ID':<8} | {'Year':<4} | {'Title':<33} | {'Runtime':<7} | {'Genres'}")
                print("-" * 160)
                
                for j, movie in enumerate(combo['movies'], 1):
                    title = movie['title']
                    if len(title) > 31:
                        title = title[:28] + "..."
                    
                    release_date = movie.get('release_date', '')
                    year = release_date[:4] if release_date else 'N/A'
                    
                    runtime = movie.get('runtime', 0)
                    runtime_str = f"{runtime}분"
                    
                    genres = movie.get('genres', '')
                    try:
                        if isinstance(genres, str) and genres.startswith('['):
                            genre_list = ast.literal_eval(genres)
                            genres_str = ', '.join(genre_list[:3])
                            if len(genre_list) > 3:
                                genres_str += f" +{len(genre_list)-3}"
                        else:
                            genres_str = str(genres)[:40]
                    except:
                        genres_str = str(genres)[:40]
                    
                    if len(genres_str) > 40:
                        genres_str = genres_str[:37] + "..."
                    
                    print(f"{j:<2} | {movie['tmdb_id']:<8} | {year:<4} | {title:<33} | {runtime_str:<7} | {genres_str}")
                    
                    overview = movie.get('overview', '')
                    if len(overview) > 100:
                        overview = overview[:97] + "..."
                    if overview:
                        print(f"     → {overview}")
                    print()
            else:
                print("조건에 맞는 조합을 찾지 못했습니다.")
        
        # 계속
        continue_input = input("\n다시 추천받으시겠습니까? (y/n): ").strip().lower()
        if continue_input == 'n':
            print("\n프로그램을 종료합니다.")
            break