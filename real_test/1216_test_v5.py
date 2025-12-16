import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple
from itertools import combinations
from math import comb

"""
[최종 수정 버전]
1. 내부 로직 최적화:
   - OTT 로딩 속도 개선 (groupby 사용)
   - 모델 초기화 시 교집합 사전 정렬 (추천 속도 향상)
   - 영화 조합 추천 시 중복 제거 및 CPU 연산 안전 장치 적용
2. 실행 로직 복구:
   - 사용자 입력(input) 받는 인터페이스 유지
"""

class HybridRecommender:
    def __init__(
        self,
        sbert_embeddings_path: str,
        lightgcn_model_path: str,
        lightgcn_data_path: str,
        metadata_path: str,
        ott_path: str,
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
        self._load_ott_data(ott_path) # [최적화됨]
        
        # 2. [최적화] 모델 간 교집합 사전 정렬 (Pre-alignment)
        print("Pre-aligning models for fast inference...")
        
        # 교집합 ID 추출
        common_ids = set(self.sbert_movie_to_idx.keys()) & set(self.lightgcn_movie_to_idx.keys())
        self.common_movie_ids = sorted(list(common_ids))
        
        # 교집합 영화들의 임베딩 행렬을 미리 구축
        self.target_sbert_matrix = []
        self.target_lightgcn_matrix = []
        
        for mid in self.common_movie_ids:
            # SBERT
            s_idx = self.sbert_movie_to_idx[mid]
            self.target_sbert_matrix.append(self.sbert_embeddings[s_idx])
            # LightGCN
            l_idx = self.lightgcn_movie_to_idx[mid]
            self.target_lightgcn_matrix.append(self.lightgcn_item_embeddings[l_idx])
            
        self.target_sbert_matrix = np.array(self.target_sbert_matrix)
        self.target_lightgcn_matrix = np.array(self.target_lightgcn_matrix)
        
        # 미리 정규화 (SBERT Cosine Similarity용)
        self.target_sbert_norm = self.target_sbert_matrix / (np.linalg.norm(self.target_sbert_matrix, axis=1, keepdims=True) + 1e-10)
        
        print(f"Pre-alignment complete. Target movies: {len(self.common_movie_ids)}")
        
        self.scaler = MinMaxScaler()

    def _generate_session_id(self, user_id: int = None) -> str:
        """
        Session ID 생성 (피드백 추적용)
        형식: rec_YYYYMMDDHHMMSS_userID_random6
        """
        import time
        import random
        import string
        
        timestamp = time.strftime("%Y%m%d%H%M%S")
        user_part = f"user{user_id}" if user_id else "guest"
        random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        
        return f"rec_{timestamp}_{user_part}_{random_part}"
    
    def _load_ott_data(self, path: str):
        """OTT 데이터 로드 (속도 최적화: groupby 사용)"""
        print(f"Loading OTT data from {path}")
        try:
            df = pd.read_csv(path)
            # iterrows 대신 groupby 사용 -> 로딩 속도 획기적 개선
            ott_dict = df.groupby('movie_id')['provider_name'].unique().apply(list).to_dict()
            self.ott_map = ott_dict
            
            print(f"Loaded OTT info for {len(self.ott_map)} movies")
            self.all_ott_providers = sorted(df['provider_name'].dropna().unique().tolist())
            
        except Exception as e:
            print(f"Error loading OTT data: {e}")
            self.ott_map = {}
            self.all_ott_providers = []

    def _load_metadata(self, path: str):
        print(f"Loading metadata from {path}")
        try:
            df = pd.read_csv(path)
            if 'movieId' in df.columns:
                df['movieId'] = df['movieId'].astype(int)
            self.metadata_map = df.set_index('movieId').to_dict('index')
            
            all_genres = set()
            for movie_data in self.metadata_map.values():
                genres = movie_data.get('genres', '')
                if isinstance(genres, str) and genres:
                    g_list = [g.strip() for g in genres.replace('|', ',').split(',')]
                    all_genres.update(g_list)
            self.all_genres = sorted(list(all_genres))
            print(f"Loaded metadata for {len(self.metadata_map)} movies")
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata_map = {}
            self.all_genres = []
        
    def _load_sbert_data(self, embeddings_path: str):
        print(f"Loading SBERT embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        self.sbert_movie_ids = data['movieId'].tolist()
        self.sbert_embeddings = np.array(data['embedding'].tolist(), dtype='float32')
        self.sbert_movie_to_idx = {mid: idx for idx, mid in enumerate(self.sbert_movie_ids)}
        
    def _load_lightgcn_data(self, data_path: str):
        data_path = Path(data_path)
        with open(data_path / 'id_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        self.lightgcn_movie_to_idx = mappings['item2id']
        self.lightgcn_idx_to_movie = mappings['id2item']
        
    def _load_lightgcn_model(self, model_path: str):
        print(f"Loading LightGCN model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        # 키 에러 방지 로직
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.lightgcn_item_embeddings = checkpoint['model_state_dict']['item_embedding.weight'].cpu().numpy()
            elif 'item_embeddings' in checkpoint:
                self.lightgcn_item_embeddings = checkpoint['item_embeddings'].cpu().numpy()
            else:
                 self.lightgcn_item_embeddings = checkpoint['item_embedding.weight'].cpu().numpy()
        
    def _get_movie_runtime(self, movie_id: int) -> int:
        meta = self.metadata_map.get(movie_id, {})
        runtime = meta.get('runtime', 0)
        try: return int(float(runtime))
        except: return 0

    def _apply_filters(self, movie_ids, preferred_genres=None, preferred_ott=None, max_runtime=None, allow_adult=True):
        """필터링 적용 (결과는 ID 리스트와 해당 인덱스 리스트 반환)"""
        filtered_indices = []
        filtered_ids = []
        
        for i, movie_id in enumerate(movie_ids):
            meta = self.metadata_map.get(movie_id, {})
            if not meta: continue
            
            # 성인물
            is_adult = str(meta.get('adult', False)).lower() == 'true'
            if is_adult and not allow_adult: continue
            
            # 런타임
            runtime = meta.get('runtime', 0)
            try: runtime = float(runtime)
            except: runtime = 0
            if max_runtime is not None and (runtime <= 0 or runtime > max_runtime): continue
            
            # 장르
            if preferred_genres:
                genres = meta.get('genres', '')
                if not genres or not isinstance(genres, str): continue
                g_list = [g.strip() for g in genres.replace('|', ',').split(',')]
                if not any(g in g_list for g in preferred_genres): continue
            
            # OTT
            if preferred_ott:
                movie_ott = self.ott_map.get(movie_id, [])
                if not movie_ott or not any(ott in movie_ott for ott in preferred_ott): continue
            
            filtered_indices.append(i)
            filtered_ids.append(movie_id)
            
        return filtered_ids, filtered_indices

    def _find_movie_combinations(
        self,
        movie_ids: List[int],
        scores: np.ndarray,
        available_time: int,
        top_k: int = 1  # 기본값 1개로 변경
    ) -> List[dict]:
        """
        주어진 시간에 맞는 영화 조합 찾기 (Knapsack)
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
        max_combinations_limit = 1_000_000  # 100만 회 제한
        max_candidates = min(len(movie_data), 60)
        
        for n in range(20, min(len(movie_data), 100)):
            max_combo_size = min(5, n // 3)
            total_combos = sum(comb(n, k) for k in range(2, max_combo_size + 1))
            if total_combos > max_combinations_limit:
                max_candidates = n - 1
                break
        
        movie_data = movie_data[:max_candidates]
        print(f"  Using top {len(movie_data)} candidates to avoid combinatorial explosion")
        
        # 조합 생성 (2~5개 영화)
        valid_combinations = []
        time_tolerance = 30  # ±30분 허용
        
        for combo_size in range(2, min(6, len(movie_data) + 1)):
            for combo in combinations(movie_data, combo_size):
                total_runtime = sum(m['runtime'] for m in combo)
                
                # 시간 조건: available_time ± 30분
                if available_time - time_tolerance <= total_runtime <= available_time + time_tolerance:
                    avg_score = np.mean([m['score'] for m in combo])
                    valid_combinations.append({
                        'movies': [m['id'] for m in combo],
                        'total_runtime': total_runtime,
                        'avg_score': avg_score
                    })
                    
                    # 1개만 찾으면 종료
                    if len(valid_combinations) >= 1:
                        break
            
            if len(valid_combinations) >= 1:
                break
        
        print(f"  Found {len(valid_combinations)} valid combination(s)")
        
        if not valid_combinations:
            return []
        
        # 점수 순 정렬 후 상위 top_k 반환
        valid_combinations.sort(key=lambda x: x['avg_score'], reverse=True)
        return valid_combinations[:top_k]

    def _get_track_a_recommendations(
        self,
        movie_ids: List[int],
        scores: np.ndarray,
        top_k: int = 3
    ) -> List[dict]:
        """
        Track A: 개인화 추천 (하이브리드 점수 기반)
        """
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            mid = movie_ids[idx]
            meta = self.metadata_map.get(mid, {})
            recommendations.append({
                'movie_id': mid,
                'hybrid_score': float(scores[idx]),
                'title_ko': meta.get('title_ko', 'Unknown'),
                'genres': meta.get('genres', ''),
                'runtime': meta.get('runtime', 0),
                'release_date': meta.get('release_date', ''),
                'popularity': meta.get('popularity', 0),
                'vote_average': meta.get('vote_average', 0),
                'adult': meta.get('adult', False),
                'ott_providers': self.ott_map.get(mid, []),
                'track': 'A'
            })
        
        return recommendations
    
    def _get_track_b_recommendations(
        self,
        movie_ids: List[int],
        scores: np.ndarray,
        top_k: int = 3,
        exclude_ids: List[int] = None,
        max_runtime: int = None,
        allow_adult: bool = True
    ) -> List[dict]:
        """
        Track B: 인기 추천 (장르 무관, 시간+성인물만 필터링, 하이브리드 점수 기반)
        """
        if exclude_ids is None:
            exclude_ids = []
        
        print(f"\n[Track B Debug]")
        print(f"  Input movies: {len(movie_ids)}")
        print(f"  Excluded by Track A: {len(exclude_ids)} movies")
        print(f"  Max runtime: {max_runtime}")
        print(f"  Allow adult: {allow_adult}")
        
        # 인기 점수 계산
        popularity_scores = []
        valid_movies = []
        valid_indices = []
        
        runtime_filtered = 0
        adult_filtered = 0
        popularity_filtered = 0
        
        for i, mid in enumerate(movie_ids):
            if mid in exclude_ids:
                continue
                
            meta = self.metadata_map.get(mid, {})
            
            # 1. 시간 필터링
            if max_runtime is not None:
                runtime = meta.get('runtime', 0)
                try:
                    runtime = int(runtime) if runtime else 0
                except (ValueError, TypeError):
                    runtime = 0
                
                if runtime <= 0 or runtime > max_runtime:
                    runtime_filtered += 1
                    continue
            
            # 2. 성인물 필터링
            if not allow_adult:
                adult = meta.get('adult', False)
                if adult:
                    adult_filtered += 1
                    continue
            
            # 3. popularity 체크 (TMDB popularity 기반)
            popularity = meta.get('popularity', 0)
            
            try:
                popularity = float(popularity) if popularity else 0
            except (ValueError, TypeError):
                popularity = 0
            
            # popularity 5.0 이상인 영화만 (인기 영화)
            if popularity < 5.0:
                popularity_filtered += 1
                continue
            
            # 4. 인기 점수 계산
            release_date = meta.get('release_date', '')
            try:
                year = int(release_date[:4]) if release_date and len(release_date) >= 4 else 0
                is_recent = year >= 2022
            except (ValueError, TypeError):
                is_recent = False
            
            # 인기 점수 = 하이브리드 점수(0.5) + popularity 정규화(0.4) + 신작 보너스(0.1)
            hybrid_score = scores[i]
            # popularity는 보통 0~100 범위, 10으로 나눠서 0~10 범위로 변환
            normalized_popularity = min(popularity / 10.0, 10.0) / 10.0  # 0~1 범위
            popularity_score = (hybrid_score * 0.5) + (normalized_popularity * 0.4) + (0.1 if is_recent else 0)
            
            popularity_scores.append(popularity_score)
            valid_movies.append(mid)
            valid_indices.append(i)
        
        print(f"  Filtered by runtime: {runtime_filtered}")
        print(f"  Filtered by adult: {adult_filtered}")
        print(f"  Filtered by popularity: {popularity_filtered}")
        print(f"  Valid movies after first pass: {len(valid_movies)}")
        
        # Fallback: 조건 완화
        if len(valid_movies) < top_k:
            print(f"  -> Relaxing popularity criteria (1.0 이상)...")
            
            for i, mid in enumerate(movie_ids):
                if mid in exclude_ids or mid in valid_movies:
                    continue
                    
                meta = self.metadata_map.get(mid, {})
                
                # 시간 필터링
                if max_runtime is not None:
                    runtime = meta.get('runtime', 0)
                    try:
                        runtime = int(runtime) if runtime else 0
                    except (ValueError, TypeError):
                        runtime = 0
                    
                    if runtime <= 0 or runtime > max_runtime:
                        continue
                
                # 성인물 필터링
                if not allow_adult:
                    adult = meta.get('adult', False)
                    if adult:
                        continue
                
                # popularity 1.0 이상만 (매우 완화)
                popularity = meta.get('popularity', 0)
                try:
                    popularity = float(popularity) if popularity else 0
                except (ValueError, TypeError):
                    popularity = 0
                
                if popularity < 1.0:
                    continue
                
                release_date = meta.get('release_date', '')
                try:
                    year = int(release_date[:4]) if release_date and len(release_date) >= 4 else 0
                    is_recent = year >= 2022
                except (ValueError, TypeError):
                    is_recent = False
                
                hybrid_score = scores[i]
                normalized_popularity = min(popularity / 10.0, 10.0) / 10.0
                popularity_score = (hybrid_score * 0.5) + (normalized_popularity * 0.4) + (0.1 if is_recent else 0)
                
                popularity_scores.append(popularity_score)
                valid_movies.append(mid)
                valid_indices.append(i)
                
                if len(valid_movies) >= top_k:
                    break
            
            print(f"  Valid movies after fallback: {len(valid_movies)}")
        
        # 2차 Fallback: popularity 무시, 하이브리드 점수만 사용
        if len(valid_movies) < top_k:
            print(f"  -> 2nd fallback: Using hybrid score only (no popularity filter)...")
            
            for i, mid in enumerate(movie_ids):
                if mid in exclude_ids or mid in valid_movies:
                    continue
                    
                meta = self.metadata_map.get(mid, {})
                
                # 시간 필터링
                if max_runtime is not None:
                    runtime = meta.get('runtime', 0)
                    try:
                        runtime = int(runtime) if runtime else 0
                    except (ValueError, TypeError):
                        runtime = 0
                    
                    if runtime <= 0 or runtime > max_runtime:
                        continue
                
                # 성인물 필터링
                if not allow_adult:
                    adult = meta.get('adult', False)
                    if adult:
                        continue
                
                # popularity 무시, 하이브리드 점수만 사용
                hybrid_score = scores[i]
                popularity_score = hybrid_score  # 점수 그대로 사용
                
                popularity_scores.append(popularity_score)
                valid_movies.append(mid)
                valid_indices.append(i)
                
                if len(valid_movies) >= top_k * 2:  # 충분한 후보 확보
                    break
            
            print(f"  Valid movies after 2nd fallback: {len(valid_movies)}")
        
        if not valid_movies:
            print(f"  -> No valid movies found for Track B!")
            return []
        
        # 상위 top_k 선정
        popularity_scores = np.array(popularity_scores)
        top_indices = np.argsort(popularity_scores)[::-1][:top_k]
        
        print(f"  Final Track B movies: {len(top_indices)}")
        
        recommendations = []
        for idx in top_indices:
            mid = valid_movies[idx]
            original_idx = valid_indices[idx]
            meta = self.metadata_map.get(mid, {})
            
            print(f"    - Movie {mid}: {meta.get('title_ko', 'Unknown')} (pop_score={popularity_scores[idx]:.4f}, popularity={meta.get('popularity', 0)})")
            
            recommendations.append({
                'movie_id': mid,
                'popularity_score': float(popularity_scores[idx]),
                'hybrid_score': float(scores[original_idx]),
                'title_ko': meta.get('title_ko', 'Unknown'),
                'genres': meta.get('genres', ''),
                'runtime': meta.get('runtime', 0),
                'release_date': meta.get('release_date', ''),
                'popularity': meta.get('popularity', 0),
                'vote_average': meta.get('vote_average', 0),
                'vote_count': meta.get('vote_count', 0),
                'adult': meta.get('adult', False),
                'ott_providers': self.ott_map.get(mid, []),
                'track': 'B'
            })
        
        return recommendations

    def _merge_tracks(
        self,
        track_a: List[dict],
        track_b: List[dict]
    ) -> dict:
        """
        Track A와 Track B 병합 (중복 제거)
        """
        # Track A 영화 ID 추출
        track_a_ids = {movie['movie_id'] for movie in track_a}
        
        # Track B에서 중복 제거
        track_b_filtered = [
            movie for movie in track_b 
            if movie['movie_id'] not in track_a_ids
        ]
        
        return {
            'track_a': {
                'label': '당신을 위한 추천',
                'description': '구독 중인 OTT에서 볼 수 있는 맞춤 추천',
                'movies': track_a
            },
            'track_b': {
                'label': '지금 HOT한 영화',
                'description': '많은 사람들이 보고 있는 인기 영화',
                'movies': track_b_filtered
            }
        }

    def recommend(
        self,
        user_movie_ids: list,
        available_time: int,
        top_k: int = 20,
        exclude_seen: bool = True,
        preferred_genres: Optional[List[str]] = None,
        preferred_ott: Optional[List[str]] = None,
        allow_adult: bool = True,
        user_id: int = None
    ) -> Tuple[str, dict]:
        
        print(f"\nStarting hybrid recommendation...")
        print(f"Available time: {available_time} min")
        
        # Session ID 생성
        session_id = self._generate_session_id(user_id)
        
        # 1. 사용자 프로필 생성
        user_sbert_vecs = []
        for mid in user_movie_ids:
            if mid in self.sbert_movie_to_idx:
                user_sbert_vecs.append(self.sbert_embeddings[self.sbert_movie_to_idx[mid]])
        
        if not user_sbert_vecs:
            return 'single', {
                'session_id': session_id,
                'recommendations': {'track_a': {'movies': []}, 'track_b': {'movies': []}},
                'total_results': 0
            }
        
        user_sbert_profile = np.mean(user_sbert_vecs, axis=0)
        user_sbert_profile = user_sbert_profile / (np.linalg.norm(user_sbert_profile) + 1e-10)
        
        user_gcn_vecs = []
        for mid in user_movie_ids:
            if mid in self.lightgcn_movie_to_idx:
                user_gcn_vecs.append(self.lightgcn_item_embeddings[self.lightgcn_movie_to_idx[mid]])
        
        if not user_gcn_vecs:
            return 'single', {
                'session_id': session_id,
                'recommendations': {'track_a': {'movies': []}, 'track_b': {'movies': []}},
                'total_results': 0
            }
        
        user_gcn_profile = np.mean(user_gcn_vecs, axis=0)
        
        # 2. 전체 점수 계산 (Pre-aligned Matrix 사용)
        sbert_scores = self.target_sbert_norm @ user_sbert_profile
        lightgcn_scores = self.target_lightgcn_matrix @ user_gcn_profile
        
        # 3. 추천 타입 결정
        recommendation_type = 'combination' if available_time >= 240 else 'single'
        max_runtime = None if recommendation_type == 'combination' else available_time
        
        # 4. 필터링 (Track A용)
        filtered_ids, filtered_indices = self._apply_filters(
            self.common_movie_ids, preferred_genres, preferred_ott, max_runtime, allow_adult
        )
        
        if not filtered_ids:
            return recommendation_type, {
                'session_id': session_id,
                'recommendations': {'track_a': {'movies': []}, 'track_b': {'movies': []}},
                'total_results': 0
            }
        
        # 5. 점수 추출 및 정규화
        filtered_sbert_scores = sbert_scores[filtered_indices]
        filtered_lightgcn_scores = lightgcn_scores[filtered_indices]
        
        norm_sbert = self.scaler.fit_transform(filtered_sbert_scores.reshape(-1, 1)).squeeze()
        norm_lightgcn = self.scaler.fit_transform(filtered_lightgcn_scores.reshape(-1, 1)).squeeze()
        
        final_scores = self.sbert_weight * norm_sbert + self.lightgcn_weight * norm_lightgcn
        
        # 6. 본 영화 제외
        if exclude_seen:
            for i, mid in enumerate(filtered_ids):
                if mid in user_movie_ids:
                    final_scores[i] = -np.inf

        # 7. 결과 반환
        if recommendation_type == 'single':
            # Track A: 개인화 추천 (상위 3개)
            track_a = self._get_track_a_recommendations(
                filtered_ids, final_scores, top_k=3
            )
            
            # Track B: 인기 추천 (장르, OTT 필터 제외)
            track_a_ids = [movie['movie_id'] for movie in track_a]
            
            # Track B용 필터링 (장르, OTT 제외, 시간+성인물만)
            trackb_filtered_ids, trackb_filtered_indices = self._apply_filters(
                self.common_movie_ids,
                preferred_genres=None,  # 장르 필터 제거
                preferred_ott=None,     # OTT 필터 제거
                max_runtime=available_time,
                allow_adult=allow_adult
            )
            
            if trackb_filtered_ids:
                # Track B 전용 점수 계산
                trackb_sbert_scores = sbert_scores[trackb_filtered_indices]
                trackb_lightgcn_scores = lightgcn_scores[trackb_filtered_indices]
                
                trackb_norm_sbert = self.scaler.fit_transform(trackb_sbert_scores.reshape(-1, 1)).squeeze()
                trackb_norm_lightgcn = self.scaler.fit_transform(trackb_lightgcn_scores.reshape(-1, 1)).squeeze()
                
                trackb_final_scores = self.sbert_weight * trackb_norm_sbert + self.lightgcn_weight * trackb_norm_lightgcn
                
                # 본 영화 제외
                if exclude_seen:
                    for i, mid in enumerate(trackb_filtered_ids):
                        if mid in user_movie_ids:
                            trackb_final_scores[i] = -np.inf
                
                track_b = self._get_track_b_recommendations(
                    trackb_filtered_ids,
                    trackb_final_scores,
                    top_k=3,
                    exclude_ids=track_a_ids,
                    max_runtime=available_time,
                    allow_adult=allow_adult
                )
            else:
                track_b = []
            
            # 병합
            recommendations = self._merge_tracks(track_a, track_b)
            
            # API 응답 형식
            result = {
                'session_id': session_id,
                'query': {
                    'duration_minutes': available_time,
                    'genres': preferred_genres,
                    'include_adult': allow_adult
                },
                'recommendations': recommendations,
                'total_results': len(track_a) + len(track_b),
                'recommendation_reason': '회원님의 취향과 최근 인기 영화를 함께 추천드려요'
            }
            
            return recommendation_type, result
            
        else:
            # ===== 조합 추천 =====
            print("Finding movie combination for available time...")
            
            # 조합 찾기 (1개만)
            combination = self._find_movie_combinations(
                filtered_ids, final_scores, available_time, top_k=1
            )
            
            if not combination:
                # 조합을 찾지 못한 경우
                return recommendation_type, {
                    'session_id': session_id,
                    'query': {
                        'duration_minutes': available_time,
                        'genres': preferred_genres,
                        'include_adult': allow_adult
                    },
                    'recommendations': [],
                    'total_results': 0,
                    'recommendation_reason': '조건에 맞는 영화 조합을 찾지 못했습니다.'
                }
            
            # 조합 메타데이터 추가
            combo = combination[0]
            combo_movies = []
            for mid in combo['movies']:
                meta = self.metadata_map.get(mid, {})
                combo_movies.append({
                    'movie_id': mid,
                    'title_ko': meta.get('title_ko', 'Unknown'),
                    'genres': meta.get('genres', ''),
                    'runtime': meta.get('runtime', 0),
                    'release_date': meta.get('release_date', ''),
                    'popularity': meta.get('popularity', 0),
                    'adult': meta.get('adult', False),
                    'ott_providers': self.ott_map.get(mid, [])
                })
            
            recommendation_data = {
                'combination_score': combo['avg_score'],
                'total_runtime': combo['total_runtime'],
                'movies': combo_movies
            }
            
            result = {
                'session_id': session_id,
                'query': {
                    'duration_minutes': available_time,
                    'genres': preferred_genres,
                    'include_adult': allow_adult
                },
                'recommendations': [recommendation_data],
                'total_results': 1,
                'recommendation_reason': f'{available_time}분 동안 즐길 수 있는 영화 조합을 추천드려요'
            }
            
            return recommendation_type, result       
# -----------------------------------------------------------
# [사용자 입력 함수 복구]
# -----------------------------------------------------------
def get_user_input_for_filters(recommender: HybridRecommender):
    """사용자로부터 필터 입력 받기"""
    
    print("\n" + "="*80)
    print("FILTER SELECTION")
    print("="*80)
    
    # 0. 시간 입력
    print("\n[0] 이용 가능 시간 입력")
    print("-" * 80)
    print("영화를 볼 수 있는 시간을 분 단위로 입력하세요.")
    print("예) 120 (2시간), 240 (4시간), 480 (8시간)")
    print("※ 240분 이상 입력 시 영화 조합을 추천합니다.")
    
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
    
    # 1. 장르 선택
    print("\n[1] 선호 장르 선택 (중복 가능)")
    print("-" * 80)
    for i, genre in enumerate(recommender.all_genres, 1):
        print(f"{i:2d}. {genre}")
    
    genre_input = input("\n선택할 장르 번호들을 입력하세요 (쉼표로 구분, 엔터로 스킵): ").strip()
    
    selected_genres = []
    if genre_input:
        try:
            genre_indices = [int(x.strip()) for x in genre_input.split(',')]
            selected_genres = [recommender.all_genres[i-1] for i in genre_indices if 1 <= i <= len(recommender.all_genres)]
        except (ValueError, IndexError):
            print("잘못된 입력입니다. 장르 필터를 건너뜁니다.")
    
    # 2. OTT 선택
    print("\n[2] 선호 OTT 플랫폼 선택 (중복 가능)")
    print("-" * 80)
    for i, ott in enumerate(recommender.all_ott_providers, 1):
        print(f"{i:2d}. {ott}")
    
    ott_input = input("\n선택할 OTT 번호들을 입력하세요 (쉼표로 구분, 엔터로 스킵): ").strip()
    
    selected_ott = []
    if ott_input:
        try:
            ott_indices = [int(x.strip()) for x in ott_input.split(',')]
            selected_ott = [recommender.all_ott_providers[i-1] for i in ott_indices if 1 <= i <= len(recommender.all_ott_providers)]
        except (ValueError, IndexError):
            print("잘못된 입력입니다. OTT 필터를 건너뜁니다.")
    
    # 3. 성인물 허용
    print("\n[3] 성인물 허용 여부")
    print("-" * 80)
    adult_input = input("성인물을 포함하시겠습니까? (y/n, 기본: n): ").strip().lower()
    allow_adult = adult_input == 'y'
    
    print("\n" + "="*80)
    print("선택된 필터:")
    print(f"시간: {available_time}분 ({available_time//60}시간 {available_time%60}분)")
    print(f"장르: {selected_genres if selected_genres else '제한 없음'}")
    print(f"OTT: {selected_ott if selected_ott else '제한 없음'}")
    print(f"성인물 허용: {allow_adult}")
    print("="*80)
    
    return {
        'available_time': available_time,
        'preferred_genres': selected_genres if selected_genres else None,
        'preferred_ott': selected_ott if selected_ott else None,
        'allow_adult': allow_adult
    }


# 실행 예시
if __name__ == "__main__":
    # 경로 설정
    SBERT_EMBEDDINGS_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl"
    LIGHTGCN_MODEL_PATH = "/home/ubuntu/ai-model/models/light_gcn/checkpoints/best_model.pt"
    LIGHTGCN_DATA_PATH = "/home/ubuntu/ai-model/models/light_gcn/data"
    METADATA_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/pre_final_movies_processed.csv"
    OTT_PATH = "/home/ubuntu/ai-model/datas/data/tmdb_ott_raw.csv"
    
    # 추천 시스템 초기화 (한 번만)
    print("\n" + "="*80)
    print("INITIALIZING RECOMMENDATION SYSTEM...")
    print("="*80)
    
    recommender = HybridRecommender(
        sbert_embeddings_path=SBERT_EMBEDDINGS_PATH,
        lightgcn_model_path=LIGHTGCN_MODEL_PATH,
        lightgcn_data_path=LIGHTGCN_DATA_PATH,
        metadata_path=METADATA_PATH,
        ott_path=OTT_PATH,
        sbert_weight=0.7,
        lightgcn_weight=0.3
    )
    
    print("\n" + "="*80)
    print("INITIALIZATION COMPLETE!")
    print("="*80)
    
    # 더미 사용자 데이터
    user_preferred_movies = [1, 296, 356]
    user_id = 15  # 테스트용 사용자 ID
    
    # 무한 루프: 사용자가 종료할 때까지 반복
    while True:
        print("\n" + "="*120)
        print("USER PREFERRED MOVIES (INPUT)")
        print("="*120)
        print(f"{'ID':<6} | {'Title (KR)':<30} | {'Genres'}")
        print("-" * 120)
        
        for mid in user_preferred_movies:
            info = recommender.metadata_map.get(mid, {})
            title = str(info.get('title_ko', 'Unknown Title'))
            genres = str(info.get('genres', 'Unknown'))
            
            if len(title) > 28: title = title[:25] + "..."
            if len(genres) > 50: genres = genres[:47] + "..."
                
            print(f"{mid:<6} | {title:<30} | {genres}")
        
        # 사용자 입력으로 필터 받기
        filters = get_user_input_for_filters(recommender)
        
        # 추천 실행
        rec_type, result = recommender.recommend(
            user_movie_ids=user_preferred_movies,
            top_k=20,
            exclude_seen=True,
            user_id=user_id,
            **filters
        )
        
        # 결과 출력
        print("\n" + "="*160)
        print(f"Session ID: {result['session_id']}")
        print(f"RECOMMENDATION RESULTS ({'SINGLE MOVIE' if rec_type == 'single' else 'MOVIE COMBINATION'})")
        print("="*160)
        print(f"SBERT weight: {recommender.sbert_weight}, LightGCN weight: {recommender.lightgcn_weight}")
        print(f"Total Results: {result['total_results']}")
        print(f"Reason: {result.get('recommendation_reason', '')}")
        print("-" * 160)
        
        if rec_type == 'single':
            # Track A 출력
            track_a = result['recommendations']['track_a']
            print(f"\n[{track_a['label']}]")
            print(f"  {track_a['description']}")
            print("-" * 160)
            print(f"{'Rank':<4} | {'ID':<6} | {'Score':<6} | {'Title (KR)':<25} | {'Year':<4} | {'Runtime':<7} | {'Adult':<5} | {'OTT':<20} | {'Genres'}")
            print("-" * 160)
            
            for i, rec in enumerate(track_a['movies'], 1):
                title = str(rec['title_ko'])
                if len(title) > 23: title = title[:20] + "..."
                    
                genres = str(rec['genres'])
                if len(genres) > 25: genres = genres[:22] + "..."
                
                runtime = str(rec['runtime'])
                adult = str(rec['adult'])
                
                release_date = str(rec.get('release_date', ''))
                year = release_date[:4] if len(release_date) >= 4 else "Unk"
                
                ott_list = rec.get('ott_providers', [])
                ott_str = ', '.join(ott_list[:2])
                if len(ott_list) > 2:
                    ott_str += f" +{len(ott_list)-2}"
                if len(ott_str) > 18:
                    ott_str = ott_str[:15] + "..."

                score = rec.get('hybrid_score', 0)
                print(f"{i:<4} | {rec['movie_id']:<6} | {score:.4f} | {title:<25} | {year:<4} | {runtime:<7} | {adult:<5} | {ott_str:<20} | {genres}")
            
            # Track B 출력
            track_b = result['recommendations']['track_b']
            print(f"\n[{track_b['label']}]")
            print(f"  {track_b['description']}")
            print("-" * 160)
            print(f"{'Rank':<4} | {'ID':<6} | {'PopScore':<8} | {'Title (KR)':<25} | {'Year':<4} | {'Runtime':<7} | {'Pop':<8} | {'OTT':<20} | {'Genres'}")
            print("-" * 160)
            
            for i, rec in enumerate(track_b['movies'], 1):
                title = str(rec['title_ko'])
                if len(title) > 23: title = title[:20] + "..."
                    
                genres = str(rec['genres'])
                if len(genres) > 25: genres = genres[:22] + "..."
                
                runtime = str(rec['runtime'])
                popularity = rec.get('popularity', 0)
                
                release_date = str(rec.get('release_date', ''))
                year = release_date[:4] if len(release_date) >= 4 else "Unk"
                
                ott_list = rec.get('ott_providers', [])
                ott_str = ', '.join(ott_list[:2])
                if len(ott_list) > 2:
                    ott_str += f" +{len(ott_list)-2}"
                if len(ott_str) > 18:
                    ott_str = ott_str[:15] + "..."

                pop_score = rec.get('popularity_score', 0)
                print(f"{i:<4} | {rec['movie_id']:<6} | {pop_score:.4f} | {title:<25} | {year:<4} | {runtime:<7} | {popularity:<8.2f} | {ott_str:<20} | {genres}")
        
        else:
            # 영화 조합 출력
            recommendations = result['recommendations']
            
            if not recommendations:
                print("\n조건에 맞는 영화 조합을 찾지 못했습니다.")
            else:
                combo = recommendations[0]
                print(f"\n[Movie Combination] Total Runtime: {combo['total_runtime']}분 | Avg Score: {combo['combination_score']:.4f}")
                print("-" * 160)
                print(f"{'#':<2} | {'ID':<6} | {'Title (KR)':<35} | {'Year':<4} | {'Runtime':<7} | {'OTT':<30} | {'Genres'}")
                print("-" * 160)
                
                for j, movie in enumerate(combo['movies'], 1):
                    title = str(movie['title_ko'])
                    if len(title) > 33: title = title[:30] + "..."
                    
                    genres = str(movie['genres'])
                    if len(genres) > 30: genres = genres[:27] + "..."
                    
                    runtime = str(movie['runtime'])
                    release_date = str(movie.get('release_date', ''))
                    year = release_date[:4] if len(release_date) >= 4 else "Unk"
                    
                    ott_list = movie.get('ott_providers', [])
                    ott_str = ', '.join(ott_list[:3])
                    if len(ott_list) > 3:
                        ott_str += f" +{len(ott_list)-3}"
                    if len(ott_str) > 28:
                        ott_str = ott_str[:25] + "..."
                    
                    print(f"{j:<2} | {movie['movie_id']:<6} | {title:<35} | {year:<4} | {runtime:<7} | {ott_str:<30} | {genres}")
        
        # 계속할지 종료할지 선택
        print("\n" + "="*80)
        continue_input = input("다시 추천받으시겠습니까? (y/n, 기본: y): ").strip().lower()
        
        if continue_input == 'n':
            print("\n프로그램을 종료합니다.")
            break
        
        print("\n" + "="*80)
        print("새로운 추천을 시작합니다...")
        print("="*80)