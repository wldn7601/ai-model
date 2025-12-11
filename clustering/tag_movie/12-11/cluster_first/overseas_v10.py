"""

조건: 클러스터링 먼저(전체 데이터) -> 그 후 필터링(평점 3.5이하), 인기순 고려, 2000-2019, 태그 관련도 0.5
-> 장르 불용어 제거

기존이랑 달라진 점 :
전체 영화 로드: 인기 있는 모든 영화를 가져옵니다.
스마트 태그 필터링: Action, Romance 같은 뻔한 장르 태그를 삭제합니다.
TF-IDF 적용: 너무 흔한 태그의 점수는 깎고, 특색 있는 태그(분위기)의 점수를 높입니다.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 🚫 1. 장르 불용어 (제거 대상)
GENRE_STOPWORDS = [
    'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary', 
    'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery', 
    'romance', 'science fiction', 'tv movie', 'thriller', 'war', 'western',
    'movie', 'film', 'cinema', 'story', 'classic'
]

# 🔄 2. 동의어 매핑 (통합 대상) - [추가됨]
SYNONYM_MAP = {
    # SF 관련
    'scifi': 'sci-fi', 'sci fi': 'sci-fi', 'future': 'sci-fi',
    # 10대 관련
    'teens': 'teen', 'teen movie': 'teen', 'high school': 'teen',
    # 원작 관련
    'adapted from:book': 'based on a book', 'books': 'based on a book', 
    'novel': 'based on a book', 'based on book': 'based on a book',
    # 기타
    'serial killer': 'slasher', 'splatter': 'slasher',
    'new york': 'new york city', 'nyc': 'new york city'
}

# ⭐ 3. 명작 키워드 (평점 기준 완화용) - [추가됨]
HQ_KEYWORDS = ['oscar', 'criterion', 'golden palm', 'top 250', 'masterpiece', 'award']

class ColdStartMovieSelector:
    def __init__(self, 
                 base_path='/home/ubuntu/ai-model/movielens_data',
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/12-11/results/cluster_first/v10',
                 # --- 설정 ---
                 clustering_min_vote=100, 
                 final_min_vote_count=500, 
                 
                 # [필터링 조건]
                 target_start_year=2000,
                 target_end_year=2019,
                 target_max_rating=3.5,     # 기본 평점 제한 (킬링타임)
                 hq_max_rating=4.3,         # [추가] 명작 그룹용 평점 제한
                 
                 num_clusters=12,
                 movies_per_cluster=10,
                 min_tag_relevance=0.5
                 ): 
        
        self.base_path = base_path
        self.output_path = output_path
        
        self.clustering_min_vote = clustering_min_vote
        self.final_min_vote_count = final_min_vote_count
        
        self.target_start_year = target_start_year
        self.target_end_year = target_end_year
        self.target_max_rating = target_max_rating
        self.hq_max_rating = hq_max_rating # 저장
        
        self.num_clusters = num_clusters
        self.movies_per_cluster = movies_per_cluster
        self.min_tag_relevance = min_tag_relevance
        
        os.makedirs(self.output_path, exist_ok=True)
        
        print(f"{'='*60}")
        print("🚀 Cold Start 영화 선정 (동의어 통합 + 명작 예외 처리)")
        print(f"{'='*60}")
        print(f"1. 전처리 : 불용어 제거 & 동의어 통합 (scifi -> sci-fi)")
        print(f"2. 필터링 : 기본 {target_max_rating}점 이하 / 명작 {hq_max_rating}점 이하")
        print(f"{'='*60}\n")
        
    def load_data_and_define_universe(self):
        print("📂 [1단계] 전체 영화 데이터 로드...")
        
        meta = pd.read_csv(f'{self.base_path}/movies_metadata_with_details.csv')
        meta['release_date'] = meta['release_date'].astype(str).str.replace('.', '-', regex=False)
        meta['release_date'] = pd.to_datetime(meta['release_date'], errors='coerce')
        meta['year'] = meta['release_date'].dt.year
        
        ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']}).reset_index()
        stats.columns = ['movieId', 'vote_count', 'avg_rating']
        
        meta['movieId'] = meta['movieId'].astype(str)
        stats['movieId'] = stats['movieId'].astype(str)
        
        merged = meta.merge(stats, on='movieId', how='inner')
        self.universe = merged[merged['vote_count'] >= self.clustering_min_vote].copy()
        
        print(f"   ✓ 클러스터링 대상 전체 영화: {len(self.universe):,}편")
        
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        self.tagdl.rename(columns={'item_id': 'movieId'}, inplace=True)
        self.tagdl['movieId'] = self.tagdl['movieId'].astype(str)

    def build_tag_matrix(self):
        print("\n🧬 [2단계] 태그 매트릭스 생성 (전처리 적용)...")
        
        target_ids = self.universe['movieId'].unique()
        tagdl_filtered = self.tagdl[self.tagdl['movieId'].isin(target_ids)].copy()
        
        print(f"   - (전) 전체 태그 데이터: {len(tagdl_filtered):,}개")
        
        # 1. 태그 점수 필터링
        tagdl_filtered = tagdl_filtered[abs(tagdl_filtered['score']) >= self.min_tag_relevance]
        
        # 2. 불용어 제거
        mask = ~tagdl_filtered['tag'].str.lower().isin(GENRE_STOPWORDS)
        tagdl_filtered = tagdl_filtered[mask].copy()
        
        # 3. [추가] 동의어 통합
        print("   🔨 동의어 매핑 적용 중...")
        tagdl_filtered['tag'] = tagdl_filtered['tag'].str.lower().replace(SYNONYM_MAP)
        
        print(f"   - (후) 전처리 완료 데이터: {len(tagdl_filtered):,}개")

        # 4. 피벗 테이블 (중복 태그는 max 점수로 통합)
        self.tag_matrix = tagdl_filtered.pivot_table(
            index='movieId', 
            columns='tag', 
            values='score', 
            aggfunc='max',  # scifi(0.8)와 sci-fi(0.9)가 만나면 0.9 선택
            fill_value=0
        )
        
        print(f"   ✓ 매트릭스 크기: {self.tag_matrix.shape}")
        
        final_ids = self.tag_matrix.index
        self.universe = self.universe[self.universe['movieId'].isin(final_ids)]

    def perform_clustering(self):
        print(f"\n🎯 [3단계] 전체 영화 클러스터링 (K={self.num_clusters})...")
        
        scaler = StandardScaler()
        tag_matrix_scaled = scaler.fit_transform(self.tag_matrix)
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tag_matrix_scaled)
        
        self.cluster_centers = kmeans.cluster_centers_
        self.tag_names = self.tag_matrix.columns.tolist()
        
        self.cluster_results = pd.DataFrame({
            'movieId': self.tag_matrix.index, 
            'cluster': clusters
        })

    def filter_and_select_representatives(self):
        print(f"\n🏆 [4단계] 후행 필터링 및 대표 선정 (동적 기준 적용)...")
        
        full_data = self.cluster_results.merge(self.universe, on='movieId', how='left')
        
        representatives_list = []
        cluster_tags_dict = {}
        cluster_desc_dict = {}
        
        for cid in range(self.num_clusters):
            # 1. 태그 설명 생성
            center = self.cluster_centers[cid]
            top_indices = np.argsort(center)[-5:][::-1]
            if len(top_indices) > 0:
                top_tags = [self.tag_names[i] for i in top_indices]
                top_scores = [center[i] for i in top_indices]
                tags_str = ', '.join(top_tags[:3])
                desc_str = ' | '.join([f"{t}({s:.1f})" for t, s in zip(top_tags, top_scores)])
            else:
                tags_str, desc_str = "No tags", ""
                
            cluster_tags_dict[cid] = tags_str
            cluster_desc_dict[cid] = desc_str
            
            # 2. [추가] 동적 평점 기준 결정
            # 태그에 'oscar', 'masterpiece' 등이 있으면 평점 기준을 높여줌
            is_hq_cluster = any(keyword in tags_str for keyword in HQ_KEYWORDS)
            
            if is_hq_cluster:
                current_limit = self.hq_max_rating  # 4.3 (명작은 살려줌)
                limit_msg = f"⭐ 명작 그룹 (limit: {current_limit})"
            else:
                current_limit = self.target_max_rating # 3.5 (일반 그룹)
                limit_msg = f"🍿 일반 그룹 (limit: {current_limit})"

            # 3. 필터링 수행
            group_movies = full_data[full_data['cluster'] == cid]
            filtered_group = group_movies[
                (group_movies['year'] >= self.target_start_year) & 
                (group_movies['year'] <= self.target_end_year) &
                (group_movies['avg_rating'] <= current_limit) & # 동적 기준 적용
                (group_movies['vote_count'] >= self.final_min_vote_count)
            ].copy()
            
            if len(filtered_group) == 0:
                print(f"   ⚠️ [Cluster {cid:02d}] '{tags_str}' -> 0건 ({limit_msg})")
                continue
            
            print(f"   ✅ [Cluster {cid:02d}] '{tags_str}' -> {len(filtered_group)}건 후보 ({limit_msg})")
            
            # 4. 인기순 정렬
            top_n = filtered_group.sort_values('vote_count', ascending=False).head(self.movies_per_cluster)
            representatives_list.append(top_n)
            
        if not representatives_list:
            raise ValueError("모든 그룹이 필터링되었습니다.")
            
        final_reps = pd.concat(representatives_list)
        final_reps['cluster_tags'] = final_reps['cluster'].map(cluster_tags_dict)
        final_reps['cluster_desc'] = final_reps['cluster'].map(cluster_desc_dict)
        
        return final_reps.sort_values(['cluster', 'vote_count'], ascending=[True, False])

    def save_results(self, representatives):
        print(f"\n💾 결과 저장: {self.output_path}")
        
        output_data = {}
        for cid in sorted(representatives['cluster'].unique()):
            group = representatives[representatives['cluster'] == cid]
            output_data[f"cluster_{cid}"] = {
                "tags": group.iloc[0]['cluster_tags'],
                "description": group.iloc[0]['cluster_desc'],
                "movies": group[['movieId', 'title_ko', 'vote_count', 'avg_rating', 'year']].to_dict('records')
            }
            
        with open(os.path.join(self.output_path, 'filtered_survey_movies.json'), 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        representatives.to_csv(os.path.join(self.output_path, 'filtered_survey_movies.csv'), index=False, encoding='utf-8-sig')
        print(f"   ✓ 저장 완료")

    def run(self):
        try:
            self.load_data_and_define_universe()
            self.build_tag_matrix()
            self.perform_clustering()
            reps = self.filter_and_select_representatives()
            self.save_results(reps)
            
            print("\n" + "="*80)
            for cid in sorted(reps['cluster'].unique()):
                group = reps[reps['cluster'] == cid]
                print(f"[{cid:02d}] {group.iloc[0]['cluster_tags']}")
                for _, row in group.head(10).iterrows():
                    print(f"   - {row['title_ko']} ({int(row['year'])}) | 평점: {row['avg_rating']:.2f}, 인기: {row['vote_count']}")
                    
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    selector = ColdStartMovieSelector()
    selector.run()