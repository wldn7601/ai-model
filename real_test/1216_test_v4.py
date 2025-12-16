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
        
        # 1. 데이터 로드 순서 중요!
        self._load_sbert_data(sbert_embeddings_path)
        self._load_lightgcn_data(lightgcn_data_path)
        self._load_lightgcn_model(lightgcn_model_path)
        self._load_metadata(metadata_path)  # 먼저 로드
        self._load_ott_data(ott_path)  # metadata 필요
        
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
    
    def _load_ott_data(self, path: str):
        """OTT 데이터 로드 (메타데이터 PKL에서 추출)"""
        print(f"Loading OTT data from metadata PKL")
        
        try:
            all_providers = set()
            ott_map = {}
            
            for movie_id, meta in self.metadata_map.items():
                ott_data = meta.get('ott_providers', '')
                
                # 문자열 형태 JSON
                if isinstance(ott_data, str) and ott_data:
                    try:
                        import json
                        ott_list = json.loads(ott_data.replace("'", '"'))
                        provider_names = [item['provider_name'] for item in ott_list if 'provider_name' in item]
                        ott_map[movie_id] = provider_names
                        all_providers.update(provider_names)
                    except:
                        ott_map[movie_id] = []
                
                # 리스트 형태
                elif isinstance(ott_data, list):
                    provider_names = [item.get('provider_name', '') for item in ott_data if isinstance(item, dict)]
                    ott_map[movie_id] = provider_names
                    all_providers.update(provider_names)
                
                else:
                    ott_map[movie_id] = []
            
            self.ott_map = ott_map
            self.all_ott_providers = sorted(list(all_providers))
            
            print(f"Loaded OTT info for {len(self.ott_map)} movies")
            print(f"Total OTT providers found: {len(self.all_ott_providers)}")
            
        except Exception as e:
            print(f"Error loading OTT data: {e}")
            import traceback
            traceback.print_exc()
            self.ott_map = {}
            self.all_ott_providers = []

    def _load_metadata(self, path: str):
        print(f"Loading metadata from {path}")
        try:
            # CSV가 아닌 PKL 파일 로드
            df = pd.read_pickle(path)
            
            if 'movieId' in df.columns:
                df['movieId'] = df['movieId'].astype(int)
            self.metadata_map = df.set_index('movieId').to_dict('index')
            
            # genres 파싱 (PKL에서도 문자열 형태일 수 있음)
            all_genres = set()
            for movie_data in self.metadata_map.values():
                genres = movie_data.get('genres', '')
                
                if isinstance(genres, str) and genres:
                    # "공포, 액션, SF, TV 영화" 형식 처리
                    g_list = [g.strip() for g in genres.split(',')]
                    all_genres.update(g_list)
                elif isinstance(genres, list):
                    # 리스트 형태로 저장되어 있는 경우
                    all_genres.update(genres)
            
            self.all_genres = sorted(list(all_genres))
            print(f"Loaded metadata for {len(self.metadata_map)} movies")
            print(f"Total genres found: {len(self.all_genres)}")
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            import traceback
            traceback.print_exc()
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
        """필터링 적용"""
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
                
                # 문자열 형태
                if isinstance(genres, str) and genres:
                    g_list = [g.strip() for g in genres.split(',')]
                # 리스트 형태
                elif isinstance(genres, list):
                    g_list = genres
                else:
                    continue
                
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
        target_time: int,
        top_k: int = 20
    ) -> List[dict]:
        """
        [CPU 최적화] Knapsack 알고리즘으로 영화 조합 찾기
        기능: 중복 영화 제거, 동적 후보 수 조절
        """
        
        # 1. 후보 데이터 준비
        movie_data = []
        for i, mid in enumerate(movie_ids):
            runtime = self._get_movie_runtime(mid)
            if runtime > 0 and runtime <= target_time:
                movie_data.append((mid, runtime, scores[i]))
        
        # 2. 점수 기준 정렬
        movie_data.sort(key=lambda x: x[2], reverse=True)
        
        # 3. 동적 후보 수 결정 (연산량 제어)
        max_combinations_limit = 2_000_000 
        max_candidates = min(len(movie_data), 60)
        
        for n in range(20, min(len(movie_data), 100)):
            if comb(n, 5) > max_combinations_limit:
                max_candidates = n - 1
                break
            max_candidates = n
            
        movie_data = movie_data[:max_candidates]
        print(f"Finding combinations from {len(movie_data)} candidates...")
        
        valid_combinations = []
        
        # 4. 조합 탐색
        max_combo_size = min(6, len(movie_data) + 1)
        for combo_size in range(2, max_combo_size):
            for combo in combinations(movie_data, combo_size):
                total_runtime = sum(m[1] for m in combo)
                
                if target_time * 0.8 <= total_runtime <= target_time:
                    avg_score = np.mean([m[2] for m in combo])
                    movie_ids_in_combo = [m[0] for m in combo]
                    
                    valid_combinations.append({
                        'movies': movie_ids_in_combo,
                        'total_runtime': total_runtime,
                        'avg_score': avg_score,
                        'individual_scores': [m[2] for m in combo]
                    })
        
        # 5. 정렬
        valid_combinations.sort(key=lambda x: x['avg_score'], reverse=True)
        
        # 6. 중복 영화 제거 (Greedy Selection)
        final_combinations = []
        used_movies = set()
        
        for combo in valid_combinations:
            combo_movies = set(combo['movies'])
            if not (combo_movies & used_movies):
                final_combinations.append(combo)
                used_movies.update(combo_movies)
                if len(final_combinations) >= top_k:
                    break
                    
        return final_combinations

    def recommend(
        self,
        user_movie_ids: list,
        available_time: int,
        top_k: int = 20,
        exclude_seen: bool = True,
        preferred_genres: Optional[List[str]] = None,
        preferred_ott: Optional[List[str]] = None,
        allow_adult: bool = True
    ) -> Tuple[str, list]:
        
        print(f"\nStarting hybrid recommendation...")
        print(f"Available time: {available_time} min")
        
        # 1. 사용자 프로필 생성
        user_sbert_vecs = []
        for mid in user_movie_ids:
            if mid in self.sbert_movie_to_idx:
                user_sbert_vecs.append(self.sbert_embeddings[self.sbert_movie_to_idx[mid]])
        
        if not user_sbert_vecs: return 'single', []
        
        user_sbert_profile = np.mean(user_sbert_vecs, axis=0)
        user_sbert_profile = user_sbert_profile / (np.linalg.norm(user_sbert_profile) + 1e-10)
        
        user_gcn_vecs = []
        for mid in user_movie_ids:
            if mid in self.lightgcn_movie_to_idx:
                user_gcn_vecs.append(self.lightgcn_item_embeddings[self.lightgcn_movie_to_idx[mid]])
        
        if not user_gcn_vecs: return 'single', []
        user_gcn_profile = np.mean(user_gcn_vecs, axis=0)
        
        # 2. 전체 점수 계산 (Pre-aligned Matrix 사용으로 고속화)
        sbert_scores = self.target_sbert_norm @ user_sbert_profile
        lightgcn_scores = self.target_lightgcn_matrix @ user_gcn_profile
        
        # 3. 필터링
        recommendation_type = 'combination' if available_time >= 240 else 'single'
        max_runtime = None if recommendation_type == 'combination' else available_time
        
        filtered_ids, filtered_indices = self._apply_filters(
            self.common_movie_ids, preferred_genres, preferred_ott, max_runtime, allow_adult
        )
        
        if not filtered_ids:
            return recommendation_type, []
            
        # 4. 점수 추출 및 정규화
        filtered_sbert_scores = sbert_scores[filtered_indices]
        filtered_lightgcn_scores = lightgcn_scores[filtered_indices]
        
        norm_sbert = self.scaler.fit_transform(filtered_sbert_scores.reshape(-1, 1)).squeeze()
        norm_lightgcn = self.scaler.fit_transform(filtered_lightgcn_scores.reshape(-1, 1)).squeeze()
        
        final_scores = self.sbert_weight * norm_sbert + self.lightgcn_weight * norm_lightgcn
        
        # 5. 본 영화 제외
        if exclude_seen:
            for i, mid in enumerate(filtered_ids):
                if mid in user_movie_ids:
                    final_scores[i] = -np.inf

        # 6. 결과 반환
        if recommendation_type == 'single':
            top_indices = np.argsort(final_scores)[::-1][:top_k]
            recommendations = []
            for idx in top_indices:
                mid = filtered_ids[idx]
                meta = self.metadata_map.get(mid, {})
                recommendations.append({
                    'movie_id': mid,
                    'hybrid_score': final_scores[idx],
                    'title_ko': meta.get('title_ko', 'Unknown'),
                    'genres': meta.get('genres', ''),
                    'runtime': meta.get('runtime', 0),
                    'release_date': meta.get('release_date', ''),
                    'popularity': meta.get('popularity', 0),
                    'adult': meta.get('adult', False),
                    'ott_providers': self.ott_map.get(mid, [])
                })
        else:
            combinations = self._find_movie_combinations(filtered_ids, final_scores, available_time, top_k)
            recommendations = []
            for combo in combinations:
                combo_movies = []
                for mid in combo['movies']:
                    meta = self.metadata_map.get(mid, {})
                    combo_movies.append({
                        'movie_id': mid,
                        'title_ko': meta.get('title_ko', 'Unknown'),
                        'genres': meta.get('genres', ''),
                        'runtime': meta.get('runtime', 0),
                        'release_date': meta.get('release_date', ''),
                        'ott_providers': self.ott_map.get(mid, [])
                    })
                recommendations.append({
                    'combination_score': combo['avg_score'],
                    'total_runtime': combo['total_runtime'],
                    'movies': combo_movies
                })
                
        return recommendation_type, recommendations

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


if __name__ == "__main__":
    # 경로 설정
    SBERT_EMBEDDINGS_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl"
    LIGHTGCN_MODEL_PATH = "/home/ubuntu/ai-model/models/light_gcn/checkpoints/best_model.pt"
    LIGHTGCN_DATA_PATH = "/home/ubuntu/ai-model/models/light_gcn/data"
    METADATA_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/pre_final_movies_processed.pkl"  # ← PKL로 변경
    OTT_PATH = None  # 사용 안 함 (메타데이터에서 추출)
    
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
        rec_type, recommendations = recommender.recommend(
            user_movie_ids=user_preferred_movies,
            top_k=20,
            exclude_seen=True,
            **filters
        )
        
        # 결과 출력
        print("\n" + "="*160)
        print(f"RECOMMENDATION RESULTS ({'SINGLE MOVIE' if rec_type == 'single' else 'MOVIE COMBINATIONS'})")
        print("="*160)
        print(f"SBERT weight: {recommender.sbert_weight}, LightGCN weight: {recommender.lightgcn_weight}")
        print(f"\nTop {len(recommendations)} Recommendations:")
        print("-" * 160)
        
        if rec_type == 'single':
            # 단일 영화 출력
            print(f"{'Rank':<4} | {'ID':<6} | {'Score':<6} | {'Title (KR)':<25} | {'Year':<4} | {'Runtime':<7} | {'Adult':<5} | {'Pop':<8} | {'OTT':<20} | {'Genres'}")
            print("-" * 160)
            
            for i, rec in enumerate(recommendations, 1):
                title = str(rec['title_ko'])
                if len(title) > 23: title = title[:20] + "..."
                    
                genres = str(rec['genres'])
                if len(genres) > 25: genres = genres[:22] + "..."
                
                runtime = str(rec['runtime'])
                adult = str(rec['adult'])
                popularity = float(rec['popularity'])
                
                release_date = str(rec.get('release_date', ''))
                year = release_date[:4] if len(release_date) >= 4 else "Unk"
                
                ott_list = rec.get('ott_providers', [])
                ott_str = ', '.join(ott_list[:2])
                if len(ott_list) > 2:
                    ott_str += f" +{len(ott_list)-2}"
                if len(ott_str) > 18:
                    ott_str = ott_str[:15] + "..."

                print(f"{i:<4} | {rec['movie_id']:<6} | {rec['hybrid_score']:.4f} | {title:<25} | {year:<4} | {runtime:<7} | {adult:<5} | {popularity:<8.2f} | {ott_str:<20} | {genres}")
        
        else:
            # 영화 조합 출력
            for i, combo in enumerate(recommendations, 1):
                print(f"\n[Combination {i}] Total Runtime: {combo['total_runtime']}분 | Avg Score: {combo['combination_score']:.4f}")
                print("-" * 160)
                
                for j, movie in enumerate(combo['movies'], 1):
                    title = str(movie['title_ko'])
                    if len(title) > 35: title = title[:32] + "..."
                    
                    genres = str(movie['genres'])
                    if len(genres) > 30: genres = genres[:27] + "..."
                    
                    runtime = str(movie['runtime'])
                    release_date = str(movie.get('release_date', ''))
                    year = release_date[:4] if len(release_date) >= 4 else "Unk"
                    
                    ott_list = movie.get('ott_providers', [])
                    ott_str = ', '.join(ott_list[:3])
                    if len(ott_list) > 3:
                        ott_str += f" +{len(ott_list)-3}"
                    
                    print(f"  {j}. [{movie['movie_id']}] {title:<35} | {year} | {runtime}분 | {ott_str:<30} | {genres}")
        
        # 계속할지 종료할지 선택
        print("\n" + "="*80)
        continue_input = input("다시 추천받으시겠습니까? (y/n, 기본: y): ").strip().lower()
        
        if continue_input == 'n':
            print("\n프로그램을 종료합니다.")
            break
        
        print("\n" + "="*80)
        print("새로운 추천을 시작합니다...")
        print("="*80)