import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ColdStartMovieSelector:
    """
    TagDL 기반 대표 영화 선정기 (수정됨: 20개 클러스터)
    """
    
    def __init__(self, 
                 base_path='/home/ubuntu/ai-model/movielens_data',
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/data',
                 min_vote_count=3000,       
                 target_count=20,           # [수정] 클러스터 수 20개로 변경
                 min_tag_relevance=0.8,     
                 movies_per_cluster=5):     
        
        self.base_path = base_path
        self.output_path = output_path
        self.min_vote_count = min_vote_count
        self.target_count = target_count
        self.min_tag_relevance = min_tag_relevance
        self.movies_per_cluster = movies_per_cluster
        
        os.makedirs(self.output_path, exist_ok=True)
        
        print("🚀 Cold Start 대표 영화 선정 시작 (20 Clusters Mode)")
        print(f"   - 입력 데이터 경로: {self.base_path}")
        print(f"   - 결과 저장 경로: {self.output_path}")
        print(f"   - 최소 평점 수: {min_vote_count:,}")
        print(f"   - 목표 클러스터(테마) 수: {target_count}")
        print(f"   - 클러스터 당 선정 영화 수: {movies_per_cluster}")
        print(f"   - 최소 태그 관련도: {min_tag_relevance}")
        
    def load_data(self):
        """데이터 로드"""
        print("\n📂 데이터 로딩 중...")
        self.movies = pd.read_csv(f'{self.base_path}/movies_metadata_restored.csv')
        self.ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        print(f"   ✓ 데이터 로드 완료 (영화: {len(self.movies):,}, 태그: {len(self.tagdl):,})")
        
    def calculate_popularity(self):
        """인기도 계산 및 필터링"""
        print("\n📊 인기도 계산 및 데이터 정제 중...")
        
        # 1. 인기도 계산
        popularity = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        popularity.columns = ['movieId', 'vote_count', 'avg_rating']
        
        # 2. 태그 데이터가 있는 영화만 필터링 (관련도 0.8 이상)
        high_score_tags = self.tagdl[abs(self.tagdl['score']) >= self.min_tag_relevance]
        movies_with_high_tags = set(high_score_tags['item_id'].unique())
        
        print(f"   ✓ 관련도 {self.min_tag_relevance} 이상 태그 보유 영화: {len(movies_with_high_tags):,}편")
        
        popularity_filtered = popularity[
            popularity['movieId'].isin(movies_with_high_tags)
        ]
        
        # 3. 평점 수 기준 필터링
        self.popular_movies = popularity_filtered[
            popularity_filtered['vote_count'] >= self.min_vote_count
        ].copy()
        
        print(f"   ✓ 인기 영화 ({self.min_vote_count}+ 평점 & High Tag): {len(self.popular_movies):,}편")
        
        # 메타데이터 병합
        self.popular_movies = self.popular_movies.merge(
            self.movies[['movieId', 'title_ko', 'poster_path', 'overview']],
            on='movieId',
            how='left'
        )
        
        # 데이터 부족 시 자동 완화 로직 (목표: 클러스터 수 * 클러스터당 영화 수)
        min_required = self.target_count * self.movies_per_cluster
        if len(self.popular_movies) < min_required:
            print(f"\n   ⚠️  영화 수가 부족합니다 ({len(self.popular_movies)} < {min_required})")
            
            for reduced_count in [2000, 1000, 500, 100]:
                self.popular_movies = popularity_filtered[
                    popularity_filtered['vote_count'] >= reduced_count
                ].copy()
                
                if len(self.popular_movies) >= min_required:
                    print(f"   ✓ 기준 완화: {reduced_count}+ 평점으로 조정")
                    self.min_vote_count = reduced_count
                    self.popular_movies = self.popular_movies.merge(
                        self.movies[['movieId', 'title_ko', 'poster_path', 'overview']],
                        on='movieId',
                        how='left'
                    )
                    break
            
            print(f"   ✓ 최종 후보 영화: {len(self.popular_movies):,}편")
    
    def build_tag_matrix(self):
        """태그 매트릭스 생성"""
        print("\n🧬 태그 매트릭스 생성 중...")
        
        popular_ids = self.popular_movies['movieId'].unique()
        
        tagdl_filtered = self.tagdl[
            self.tagdl['item_id'].isin(popular_ids)
        ].copy()
        
        tagdl_filtered = tagdl_filtered[
            abs(tagdl_filtered['score']) >= self.min_tag_relevance
        ]
        
        self.tag_matrix = tagdl_filtered.pivot_table(
            index='item_id',
            columns='tag',
            values='score',
            fill_value=0
        )
        
        print(f"   ✓ 태그 매트릭스 shape: {self.tag_matrix.shape}")
        
        if self.tag_matrix.shape[0] < self.target_count:
            old_target = self.target_count
            self.target_count = max(2, self.tag_matrix.shape[0] // self.movies_per_cluster)
            print(f"   ⚠️  데이터 부족으로 클러스터 수 조정: {old_target} → {self.target_count}")
        
    def perform_clustering(self):
        """K-Means 클러스터링 수행"""
        print(f"\n🎯 K-Means 클러스터링 수행 중 (K={self.target_count})...")
        
        scaler = StandardScaler()
        tag_matrix_scaled = scaler.fit_transform(self.tag_matrix)
        
        kmeans = KMeans(
            n_clusters=self.target_count,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        clusters = kmeans.fit_predict(tag_matrix_scaled)
        
        cluster_df = pd.DataFrame({
            'movieId': self.tag_matrix.index,
            'cluster': clusters
        })
        
        self.cluster_centers = kmeans.cluster_centers_
        self.tag_names = self.tag_matrix.columns.tolist()
        
        return cluster_df
        
    def select_representatives(self, cluster_df):
        """각 클러스터별 상위 N개 영화 선정"""
        print(f"\n🏆 클러스터별 상위 {self.movies_per_cluster}개 영화 선정 중...")
        
        result = cluster_df.merge(
            self.popular_movies,
            on='movieId',
            how='left'
        )
        
        # 1. 클러스터별, 평점순 정렬 후 상위 N개 추출
        representatives = result.sort_values(
            ['cluster', 'vote_count'], 
            ascending=[True, False]
        ).groupby('cluster').head(self.movies_per_cluster).copy()
        
        # 2. 클러스터 태그 정보 매핑
        cluster_tags_list = []
        for cluster_id in range(self.target_count):
            if cluster_id < len(self.cluster_centers):
                center = self.cluster_centers[cluster_id]
                top_indices = np.argsort(center)[-3:][::-1]
                top_tags = [self.tag_names[i] for i in top_indices]
                cluster_tags_list.append(', '.join(top_tags))
            else:
                cluster_tags_list.append('Unknown')
        
        representatives['cluster_tags'] = representatives['cluster'].map(
            dict(enumerate(cluster_tags_list))
        )
        
        representatives = representatives.sort_values(['cluster', 'vote_count'], ascending=[True, False])
        
        print(f"   ✓ 총 {len(representatives)}편 선정 완료 (클러스터당 최대 {self.movies_per_cluster}편)")
        
        return representatives.reset_index(drop=True)
    
    def run(self):
        try:
            self.load_data()
            self.calculate_popularity()
            self.build_tag_matrix()
            
            cluster_df = self.perform_clustering()
            representatives = self.select_representatives(cluster_df)
            
            return representatives
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            raise

    def save_results(self, representatives):
        print(f"\n💾 결과 저장 중... (경로: {self.output_path})")
        
        # CSV 저장
        csv_path = os.path.join(self.output_path, 'cold_start_survey_movies.csv')
        representatives.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # JSON 저장
        json_path = os.path.join(self.output_path, 'cold_start_survey_movies.json')
        survey_json = representatives[[
            'movieId', 'title_ko', 'poster_path', 'cluster_tags', 
            'vote_count', 'avg_rating', 'cluster'
        ]].to_dict('records')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(survey_json, f, ensure_ascii=False, indent=2)
            
        print(f"   ✓ 파일 저장 완료: {csv_path}")
        self._print_results(representatives)
        return {'csv': csv_path, 'json': json_path}

    def _print_results(self, representatives):
        print("\n" + "="*100)
        print(f"📋 선정된 영화 목록 (Top {self.movies_per_cluster} per Cluster)")
        print("="*100)
        
        for cluster_id, group in representatives.groupby('cluster'):
            tags = group.iloc[0]['cluster_tags']
            print(f"\n🏷️  CLUSTER {cluster_id:02d} [{tags}] - 총 {len(group)}편")
            print("-" * 80)
            
            for idx, row in group.iterrows():
                print(f"   • {row['title_ko']} (★{row['avg_rating']:.1f}, {row['vote_count']:,} votes)")
            print("-" * 80)

def main():
    INPUT_PATH = '/home/ubuntu/ai-model/movielens_data'
    OUTPUT_PATH = '/home/ubuntu/ai-model/clustering/tag_movie/data'
    
    # [수정] 클러스터 20개로 변경
    configs = [
        {
            'min_vote_count': 3000, 
            'target_count': 20,         # <-- 20개로 변경됨
            'min_tag_relevance': 0.8,  
            'movies_per_cluster': 5    
        },
        # 예비 설정도 20개로 맞춤 (조건 완화 버전)
        {
            'min_vote_count': 1000, 
            'target_count': 20, 
            'min_tag_relevance': 0.6, 
            'movies_per_cluster': 5
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*100}")
        print(f"실행 시도 #{i+1}: {config}")
        print(f"{'='*100}\n")
        
        try:
            selector = ColdStartMovieSelector(
                base_path=INPUT_PATH,
                output_path=OUTPUT_PATH,
                **config
            )
            
            representatives = selector.run()
            
            # 클러스터 20개 * 2개(최소) = 40개 이상은 되어야 정상
            if len(representatives) < 40: 
                raise ValueError(f"선정된 영화가 너무 적습니다 ({len(representatives)}편).")
                
            selector.save_results(representatives)
            print(f"\n✅ 실행 #{i+1} 성공!")
            break
            
        except Exception as e:
            print(f"\n⚠️  실행 #{i+1} 실패 또는 조건 미달: {e}")
            if i < len(configs) - 1:
                print(f"   조건을 완화하여 다시 시도합니다...\n")

if __name__ == "__main__":
    main()