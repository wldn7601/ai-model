"""
조건: 클러스터링 먼저(전체 데이터) -> 그 후 필터링(평점 3.7이하), 인기순 고려, 2010-2019, 태그 관련도 0.8
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ColdStartMovieSelector:
    def __init__(self, 
                 base_path='/home/ubuntu/ai-model/movielens_data',
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/12-11/results/cluster_first/v4',
                 # --- 설정 ---
                 clustering_min_vote=100, # 클러스터링을 위한 최소 인기도 (전체 지도용)
                 final_min_vote_count=500, # 결과용 최소 인기도
                 
                 # [필터링 조건] - 나중에 적용됨
                 target_start_year=2010,
                 target_end_year=2019,
                 target_max_rating=3.7,
                 
                 num_clusters=12,
                 movies_per_cluster=10,
                 min_tag_relevance=0.8
                 ): 
        
        self.base_path = base_path
        self.output_path = output_path
        
        self.clustering_min_vote = clustering_min_vote
        self.final_min_vote_count = final_min_vote_count
        
        self.target_start_year = target_start_year
        self.target_end_year = target_end_year
        self.target_max_rating = target_max_rating
        
        self.num_clusters = num_clusters
        self.movies_per_cluster = movies_per_cluster
        self.min_tag_relevance = min_tag_relevance
        
        os.makedirs(self.output_path, exist_ok=True)
        
        print(f"{'='*60}")
        print("🚀 Cold Start 영화 선정 (Real Cluster First)")
        print(f"{'='*60}")
        print(f"1. 전체 지도 생성 : 모든 연도, 모든 평점의 영화로 클러스터링")
        print(f"2. 후행 필터링   : 2010-2019년 & 평점 3.7이하만 추출")
        print(f"3. 정렬 기준     : 인기순 (Vote Count)")
        print(f"{'='*60}\n")
        
    def load_data_and_define_universe(self):
        print("📂 [1단계] 전체 영화 데이터 로드 (모든 연도)...")
        
        # 1. 메타데이터 로드
        meta = pd.read_csv(f'{self.base_path}/movies_metadata_with_details.csv')
        meta['release_date'] = meta['release_date'].astype(str).str.replace('.', '-', regex=False)
        meta['release_date'] = pd.to_datetime(meta['release_date'], errors='coerce')
        meta['year'] = meta['release_date'].dt.year
        
        # 2. Ratings 통계 계산
        ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']}).reset_index()
        stats.columns = ['movieId', 'vote_count', 'avg_rating']
        
        # 3. 병합 (전체 영화)
        meta['movieId'] = meta['movieId'].astype(str)
        stats['movieId'] = stats['movieId'].astype(str)
        
        merged = meta.merge(stats, on='movieId', how='inner')
        
        # [중요] 여기서는 연도나 평점으로 거르지 않습니다! (인기도만 살짝 봄)
        self.universe = merged[merged['vote_count'] >= self.clustering_min_vote].copy()
        
        print(f"   ✓ 클러스터링 대상 전체 영화: {len(self.universe):,}편")
        
        # 태그 로드
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        self.tagdl.rename(columns={'item_id': 'movieId'}, inplace=True)
        self.tagdl['movieId'] = self.tagdl['movieId'].astype(str)

    def build_tag_matrix(self):
        print("\n🧬 [2단계] 태그 매트릭스 생성...")
        
        target_ids = self.universe['movieId'].unique()
        tagdl_filtered = self.tagdl[self.tagdl['movieId'].isin(target_ids)].copy()
        
        # 태그 엄격 필터링
        tagdl_filtered = tagdl_filtered[abs(tagdl_filtered['score']) >= self.min_tag_relevance]
        
        if len(tagdl_filtered) == 0:
            raise ValueError("❌ 기준이 너무 높아 태그 데이터가 없습니다.")

        self.tag_matrix = tagdl_filtered.pivot_table(
            index='movieId', columns='tag', values='score', fill_value=0
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
        print(f"\n🏆 [4단계] 후행 필터링 및 대표 선정...")
        print(f"   조건: {self.target_start_year}~{self.target_end_year}년 & 평점 {self.target_max_rating}이하")
        
        full_data = self.cluster_results.merge(self.universe, on='movieId', how='left')
        
        representatives_list = []
        cluster_tags_dict = {}
        cluster_desc_dict = {}
        
        for cid in range(self.num_clusters):
            group_movies = full_data[full_data['cluster'] == cid]
            
            # [여기서 모든 필터링을 수행합니다]
            filtered_group = group_movies[
                (group_movies['year'] >= self.target_start_year) & 
                (group_movies['year'] <= self.target_end_year) &
                (group_movies['avg_rating'] <= self.target_max_rating) &
                (group_movies['vote_count'] >= self.final_min_vote_count)
            ].copy()
            
            # 태그 설명 (그룹 전체 기준)
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
            
            if len(filtered_group) == 0:
                print(f"   ⚠️ [Cluster {cid:02d}] '{tags_str}' 그룹은 조건 만족 영화 없음 (전멸)")
                continue
            
            # 인기순 정렬 & 상위 N개
            top_n = filtered_group.sort_values('vote_count', ascending=False).head(self.movies_per_cluster)
            representatives_list.append(top_n)
            
        if not representatives_list:
            raise ValueError("모든 그룹이 필터링되었습니다. 조건을 완화하세요.")
            
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
                    
        except ValueError as e:
            print(f"\n❌ 실행 실패: {e}")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    selector = ColdStartMovieSelector()
    selector.run()