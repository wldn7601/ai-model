# 1. 구글 드라이브 마운트
# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import json
import os
import warnings

warnings.filterwarnings('ignore')

# 🚫 장르 불용어
GENRE_STOPWORDS = [
    'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary', 
    'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery', 
    'romance', 'science fiction', 'tv movie', 'thriller', 'war', 'western',
    'movie', 'film', 'cinema', 'story'
]

# 🔄 동의어 매핑 사전
SYNONYM_MAP = {
    'scifi': 'sci-fi', 'sci fi': 'sci-fi', 'future': 'sci-fi',
    'teens': 'teen', 'teen movie': 'teen', 'high school': 'teen',
    'adapted from:book': 'based on a book', 'books': 'based on a book', 
    'novel': 'based on a book', 'based on book': 'based on a book',
    'serial killer': 'slasher', 'splatter': 'slasher',
    'new york': 'new york city'
}

class SmartColdStartSelectorV3_Print:
    def __init__(self, 
                 base_path='/home/ubuntu/ai-model/movielens_data',
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/12-11/results/cluster_first/v12',
                 clustering_min_vote=100,  
                 final_min_vote_count=500, 
                 target_start_year=2010,
                 target_end_year=2019,
                 target_max_rating=3.8,     # 일반 평점 제한
                 hq_max_rating=4.3,         # 명작 평점 제한
                 movies_per_cluster=10,
                 min_tag_relevance=0.8,
                 num_clusters=12
                 ): 
        
        self.base_path = base_path
        self.output_path = output_path
        self.clustering_min_vote = clustering_min_vote
        self.final_min_vote_count = final_min_vote_count
        self.target_start_year = target_start_year
        self.target_end_year = target_end_year
        self.target_max_rating = target_max_rating
        self.hq_max_rating = hq_max_rating
        self.num_clusters = num_clusters
        self.movies_per_cluster = movies_per_cluster
        self.min_tag_relevance = min_tag_relevance
        
        os.makedirs(self.output_path, exist_ok=True)
        print(f"{'='*60}")
        print("🚀 스마트 Cold Start V3 (결과 출력 버전)")
        print(f"   - 일반 평점: ~{self.target_max_rating} / 명작: ~{self.hq_max_rating}")
        print(f"   - 기간: {self.target_start_year} ~ {self.target_end_year}")
        print(f"{'='*60}")

    def load_data(self):
        print("📂 [1단계] 데이터 로드...")
        path = f'{self.base_path}/movies_metadata_with_details.csv'
        if not os.path.exists(path): path = f'{self.base_path}/movies_metadata_restored.csv'
        meta = pd.read_csv(path)
        
        meta['release_date'] = pd.to_datetime(meta['release_date'].astype(str).str.replace('.', '-'), errors='coerce')
        meta['year'] = meta['release_date'].dt.year
        meta['movieId'] = pd.to_numeric(meta['movieId'], errors='coerce')
        meta.dropna(subset=['movieId'], inplace=True)
        meta['movieId'] = meta['movieId'].astype(int)

        ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']}).reset_index()
        stats.columns = ['movieId', 'vote_count', 'avg_rating']
        stats['movieId'] = pd.to_numeric(stats['movieId'], errors='coerce')
        
        merged = meta.merge(stats, on='movieId', how='inner')
        self.universe = merged[merged['vote_count'] >= self.clustering_min_vote].copy()
        print(f"   ✓ 대상 영화: {len(self.universe):,}편")
        
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        self.tagdl.rename(columns={'item_id': 'movieId'}, inplace=True)
        self.tagdl['movieId'] = pd.to_numeric(self.tagdl['movieId'], errors='coerce')

    def build_and_transform_matrix(self):
        print("\n🧬 [2단계] 태그 전처리 및 매트릭스 생성...")
        target_ids = self.universe['movieId'].unique()
        tags = self.tagdl[self.tagdl['movieId'].isin(target_ids)].copy()
        
        tags = tags[abs(tags['score']) >= self.min_tag_relevance]
        
        mask = ~tags['tag'].str.lower().isin(GENRE_STOPWORDS)
        tags = tags[mask].copy()

        print("   🔨 동의어 통합 중...")
        tags['tag'] = tags['tag'].str.lower().replace(SYNONYM_MAP)
        
        raw_matrix = tags.pivot_table(
            index='movieId', columns='tag', values='score', aggfunc='max', fill_value=0
        )
        
        tfidf = TfidfTransformer()
        matrix_weighted = tfidf.fit_transform(raw_matrix)
        
        self.matrix_weighted_df = pd.DataFrame(
            matrix_weighted.toarray(), 
            index=raw_matrix.index, 
            columns=raw_matrix.columns
        )
        print(f"   ✓ 최종 매트릭스: {self.matrix_weighted_df.shape}")
        
        final_ids = self.matrix_weighted_df.index
        self.universe = self.universe[self.universe['movieId'].isin(final_ids)]

    def get_smart_centroids(self):
        print(f"\n🧠 [3단계] 스마트 초기값(Seed) 선정 중...")
        variances = self.matrix_weighted_df.var().sort_values(ascending=False)
        top_tags = variances.head(self.num_clusters).index.tolist()
        
        init_centroids = []
        for tag in top_tags:
            best_movie_id = self.matrix_weighted_df[tag].idxmax()
            seed_vector = self.matrix_weighted_df.loc[best_movie_id].values
            init_centroids.append(seed_vector)
        return np.array(init_centroids)

    def perform_clustering(self):
        print(f"\n🎯 [4단계] 클러스터링 수행...")
        smart_seeds = self.get_smart_centroids()
        
        kmeans = KMeans(
            n_clusters=self.num_clusters, 
            init=smart_seeds,
            n_init=1,
            random_state=42
        )
        
        clusters = kmeans.fit_predict(self.matrix_weighted_df)
        self.cluster_centers = kmeans.cluster_centers_
        self.tag_names = self.matrix_weighted_df.columns.tolist()
        
        self.cluster_results = pd.DataFrame({
            'movieId': self.matrix_weighted_df.index, 
            'cluster': clusters
        })

    def filter_and_save(self):
        print(f"\n🏆 [5단계] 필터링 및 결과 출력...")
        full_data = self.cluster_results.merge(self.universe, on='movieId', how='left')
        
        final_list = []
        output_json = {}
        
        HQ_KEYWORDS = ['oscar', 'criterion', 'golden palm', 'top 250', 'masterpiece', 'classic']

        print(f"{'='*80}")
        
        for cid in range(self.num_clusters):
            center = self.cluster_centers[cid]
            top_idx = np.argsort(center)[-5:][::-1]
            top_tags = [self.tag_names[i] for i in top_idx] if len(top_idx) > 0 else []
            tags_str = ', '.join(top_tags[:3])
            
            is_hq_cluster = any(k in tags_str for k in HQ_KEYWORDS)
            
            if is_hq_cluster:
                current_limit = self.hq_max_rating
                mode_str = "⭐ 명작 모드"
            else:
                current_limit = self.target_max_rating
                mode_str = "🍿 일반 모드"

            group = full_data[full_data['cluster'] == cid]
            filtered = group[
                (group['year'] >= self.target_start_year) & 
                (group['year'] <= self.target_end_year) & 
                (group['avg_rating'] <= current_limit) &  
                (group['vote_count'] >= self.final_min_vote_count)
            ].copy()
            
            # 결과 출력용 헤더
            print(f"\n📂 [Cluster {cid:02d}] {mode_str}")
            print(f"   🏷️  Tags: {tags_str}")
            
            if len(filtered) == 0:
                print(f"   ⚠️ 조건 만족 영화 없음 (Limit: {current_limit})")
                continue
            
            top_n = filtered.sort_values('vote_count', ascending=False).head(self.movies_per_cluster)
            top_n['cluster_tags'] = tags_str
            final_list.append(top_n)
            
            # 🎥 영화 목록 출력 (여기가 추가된 부분입니다)
            for i, (_, row) in enumerate(top_n.iterrows(), 1):
                title = row['title_ko'] if pd.notnull(row['title_ko']) else row['title']
                print(f"   {i:>2}. {title} ({int(row['year'])}) - ⭐ {row['avg_rating']:.1f}")

            # JSON 저장용
            movies_json = []
            for _, row in top_n.iterrows():
                movies_json.append({
                    'movieId': int(row['movieId']),
                    'title_ko': row['title_ko'],
                    'year': int(row['year']),
                    'rating': float(row['avg_rating']),
                    'votes': int(row['vote_count'])
                })
            
            output_json[f"cluster_{cid}"] = {
                "tags": tags_str,
                "type": "Masterpiece" if is_hq_cluster else "Casual",
                "movies": movies_json
            }

        print(f"\n{'='*80}")

        if final_list:
            final_df = pd.concat(final_list)
            json_path = os.path.join(self.output_path, 'smart_filtered_movies_v3.json')
            csv_path = os.path.join(self.output_path, 'smart_filtered_movies_v3.csv')
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"💾 결과 파일 저장 완료: {csv_path}")
        else:
            print("❌ 조건에 맞는 영화가 하나도 없습니다.")

    def run(self):
        try:
            self.load_data()
            self.build_and_transform_matrix()
            self.perform_clustering() 
            self.filter_and_save()
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    selector = SmartColdStartSelectorV3_Print()
    selector.run()