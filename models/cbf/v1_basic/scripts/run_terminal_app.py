import pickle
import json
import numpy as np
import os
import sys

# ==========================================
# 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
# 임베딩 데이터 (벡터 연산용)
EMBEDDING_PKL = os.path.join(BASE_DIR, 'outputs', 'movie_embeddings_centered.pkl')
# 서비스 자산 데이터 (UI 표시용)
ASSETS_JSON = os.path.join(BASE_DIR, 'data', 'service_assets.json')

class MovieTerminalApp:
    def __init__(self):
        print("\n🎬 영화 추천 서비스 초기화 중...")
        
        # 1. 자산 로드
        if not os.path.exists(ASSETS_JSON):
            print("❌ 에러: service_assets.json이 없습니다. build_service_assets.py를 먼저 실행하세요.")
            sys.exit(1)
            
        with open(ASSETS_JSON, 'r', encoding='utf-8') as f:
            self.assets = json.load(f)
            
        # 2. 임베딩 로드
        with open(EMBEDDING_PKL, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            self.movie_ids = data['movie_ids']
            
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        print("✅ 시스템 준비 완료!\n")

    def _select_multi_options(self, options, label):
        """
        [Helper] 목록을 보여주고 다중 선택을 받는 함수
        """
        print(f"\n📋 [{label} 목록]")
        
        # 보기 좋게 3열로 출력 (선택사항)
        for i, opt in enumerate(options, 1):
            print(f"{i:2}. {opt:<20}", end="")
            if i % 3 == 0: print() # 줄바꿈
        print() # 마지막 줄바꿈
        
        print(f"\n👉 선택할 {label} 번호를 쉼표(,)로 구분해서 입력하세요.")
        print("   (예: 1, 3, 5 / 선택하지 않으려면 그냥 엔터)")
        
        while True:
            user_input = input("입력 > ").strip()
            
            if not user_input:
                return [] # 선택 안 함
            
            try:
                # 쉼표로 분리하고 숫자로 변환
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                
                # 유효성 검사
                selected_items = []
                for idx in indices:
                    if 0 <= idx < len(options):
                        selected_items.append(options[idx])
                
                if selected_items:
                    print(f"   ✅ 선택됨: {', '.join(selected_items)}")
                    return selected_items
                else:
                    print("   ⚠️ 유효한 번호가 없습니다. 다시 입력해주세요.")
                    
            except ValueError:
                print("   ⚠️ 숫자와 쉼표만 입력해주세요.")

    def run_onboarding(self):
        """
        Step 1: 선호도 조사
        """
        print("="*60)
        print("🙋‍♂️ [Step 1] 당신의 취향을 알려주세요!")
        print("다양한 스타일의 영화 10개를 보여드립니다.")
        print("본 영화라면 평점(1~5)을, 안 봤다면 0을 입력하세요.")
        print("="*60)
        
        user_ratings = {}
        
        for i, movie in enumerate(self.assets['onboarding_movies'], 1):
            print(f"\n[{i}/10] {movie['title']}")
            print(f"   장르: {', '.join(movie['genres'])}")
            
            while True:
                try:
                    score_input = input("   👉 평점 (0=Pass, 1=최악 ~ 5=최고): ").strip()
                    if not score_input: score_input = "0"
                    score = int(score_input)
                    if 0 <= score <= 5:
                        if score > 0:
                            user_ratings[movie['movieId']] = float(score)
                        break
                    print("   ⚠️ 0에서 5 사이의 숫자를 입력해주세요.")
                except ValueError:
                    print("   ⚠️ 숫자만 입력해주세요.")
        
        return user_ratings

    def create_user_vector(self, ratings):
        """
        Step 2: 사용자 벡터 생성
        """
        if not ratings:
            print("\n⚠️ 평가된 영화가 없어 기본 추천(인기순)으로 대체합니다.")
            # 전체 평균 벡터 반환
            return np.mean(self.embeddings, axis=0)

        weighted_sum = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        
        for mid, score in ratings.items():
            if mid not in self.id_to_idx: continue
            
            idx = self.id_to_idx[mid]
            vec = self.embeddings[idx]
            
            # 1점(-1.0) ~ 5점(+1.0) 가중치
            weight = (score - 3.0) / 2.0
            
            weighted_sum += vec * weight
            total_weight += abs(weight)
            
        if total_weight == 0:
            return np.mean(self.embeddings, axis=0)
            
        user_vec = weighted_sum / total_weight
        # 정규화
        norm = np.linalg.norm(user_vec)
        if norm > 0: user_vec = user_vec / norm
            
        return user_vec

    def get_user_conditions(self):
        """
        Step 3: 필터링 조건 입력 (번호 선택 방식)
        """
        print("\n" + "="*60)
        print("✈️ [Step 2] 추천 조건을 입력해주세요")
        print("="*60)
        
        # 1. 장르 선택 (다중 선택)
        selected_genres = self._select_multi_options(
            self.assets['available_genres'], "장르"
        )
        
        # 2. OTT 선택 (다중 선택)
        selected_otts = self._select_multi_options(
            self.assets['available_providers'], "OTT"
        )
        
        # 3. 시간 선택
        while True:
            try:
                time_input = input("\n👉 최대 이동 시간(분) (예: 120, 없으면 엔터): ").strip()
                runtime = int(time_input) if time_input else 0
                break
            except:
                print("숫자만 입력하세요.")
                
        return {
            'genres': selected_genres, 
            'otts': selected_otts, 
            'runtime': runtime
        }

    def recommend_movies(self, user_vec, conditions):
        """
        Step 4: 최종 추천 (다중 조건 필터링)
        """
        print("\n" + "="*60)
        print("🍿 [Step 3] AI가 영화를 찾고 있습니다...")
        print("="*60)
        
        candidates = []
        
        # 필터 조건 가져오기
        target_genres = set(conditions['genres']) # 집합으로 변환 (검색 속도)
        target_otts = conditions['otts']
        max_time = conditions['runtime']
        
        for i, meta in enumerate(self.metadata):
            # 1. 장르 체크 (하나라도 겹치면 통과 - OR 조건)
            # 만약 "액션, 코미디"를 선택했다면 -> 액션 영화 OK, 코미디 영화 OK
            if target_genres:
                movie_genres = set(meta['genres'])
                # 교집합이 없으면(겹치는게 없으면) 탈락
                if not target_genres.intersection(movie_genres):
                    continue
                
            # 2. OTT 체크 (하나라도 겹치면 통과)
            if target_otts:
                movie_providers = meta.get('providers', [])
                # 영화의 OTT 중 하나라도 내가 선택한 OTT 목록에 있으면 OK
                # (부분 일치 처리를 위해 in 검사)
                is_available = False
                for my_ott in target_otts:
                    for mv_ott in movie_providers:
                        if my_ott.lower() in mv_ott.lower():
                            is_available = True
                            break
                    if is_available: break
                
                if not is_available:
                    continue
            
            # 3. 시간 체크
            if max_time > 0:
                # 런타임 정보가 없거나 0이면 999분으로 간주해버림 (보수적 접근)
                if meta.get('runtime', 999) > max_time:
                    continue
                    
            candidates.append(i)
            
        print(f"   ✓ 필터링 통과 후보: {len(candidates)}개")
        
        if not candidates:
            print("❌ 조건에 맞는 영화가 없습니다. 조건을 조금만 완화해주세요.")
            return

        # 벡터 유사도 계산
        candidate_vecs = self.embeddings[candidates]
        scores = np.dot(candidate_vecs, user_vec)
        
        # Top 3 추출
        top_k = min(3, len(candidates))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        print(f"\n🎉 [추천 결과 Top {top_k}]")
        print("-" * 60)
        for rank, local_idx in enumerate(top_indices, 1):
            real_idx = candidates[local_idx]
            meta = self.metadata[real_idx]
            score = scores[local_idx]
            
            print(f"{rank}위: {meta['title']} (유사도: {score:.4f})")
            print(f"   장르: {', '.join(meta['genres'])}")
            print(f"   시간: {meta['runtime']}분")
            providers = meta.get('providers', [])
            print(f"   OTT : {', '.join(providers) if providers else '정보 없음'}")
            print("-" * 60)

if __name__ == "__main__":
    app = MovieTerminalApp()
    
    # 1. 온보딩
    ratings = app.run_onboarding()
    
    # 2. 벡터 생성
    user_vec = app.create_user_vector(ratings)
    
    # 3. 조건 입력 (번호 선택)
    conditions = app.get_user_conditions()
    
    # 4. 추천
    app.recommend_movies(user_vec, conditions)