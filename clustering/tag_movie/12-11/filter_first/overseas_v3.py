"""

조건 3
평점 3.5이하, 인기순은 고려

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
                 # [수정] 경로 v3
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/12-11/results/filter_first/v3',
                 # --- 사용자 요청 설정 ---
                 min_vote_count=500,     # 최소한의 데이터 신뢰도를 위한 하한선
                 max_rating=3.5,         # [수정] 3.5점 이하 (더 엄격해짐)
                 num_clusters=12,        # 클러스터 12개
                 movies_per_cluster=10,  # 각 대표영화 10개
                 min_tag_relevance=0.8   # 태그 관련도 0.8 (매우 엄격)
                 # ---------------------
                 ): 
        
        self.base_path = base_path
        self.output_path = output_path
        self.min_vote_count = min_vote_count
        self.max_rating = max_rating
        self.num_clusters = num_clusters
        self.movies_per_cluster = movies_per_cluster
        self.total_target = num_clusters * movies_per_cluster
        self.min_tag_relevance = min_tag_relevance
        
        os.makedirs(self.output_path, exist_ok=True)
        
        print(f"{'='*60}")
        print("🚀 Cold Start 영화 선정 (평점 3.5↓ & 인기순 정렬 버전)")
        print(f"{'='*60}")
        print(f"1. 평점 조건 : {max_rating}점 이하만 선택")
        print(f"2. 정렬 기준 : [인기도 고려] -> 평점 개수(vote_count) 많은 순서")
        print(f"3. 태그 기준 : 관련도 {min_tag_relevance} 이상 (매우 엄격)")
        print(f"4. 저장 경로 : {self.output_path}")
        print(f"{'='*60}\n")
        
    def load_and_process_data(self):
        print("📂 데이터 로딩 및 집계 중...")
        
        ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'vote_count', 'avg_rating']
        
        print(f"   ✓ 전체 영화 수: {len(movie_stats):,}편")
        
        # 1차 필터링
        self.candidates = movie_stats[
            (movie_stats['vote_count'] >= self.min_vote_count) & 
            (movie_stats['avg_rating'] <= self.max_rating)
        ].copy()
        
        print(f"   ✓ 1차 필터링 완료: {len(self.candidates):,}편")
        
        meta = pd.read_csv(f'{self.base_path}/movies_metadata_restored.csv')
        self.candidates['movieId'] = self.candidates['movieId'].astype(str)
        meta['movieId'] = meta['movieId'].astype(str)
        
        self.candidates = self.candidates.merge(
            meta[['movieId', 'title_ko', 'poster_path']], 
            on='movieId', 
            how='inner'
        )
        
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        self.tagdl.rename(columns={'item_id': 'movieId'}, inplace=True)
        self.tagdl['movieId'] = self.tagdl['movieId'].astype(str)

    def build_tag_matrix(self):
        print("\n🧬 태그 매트릭스 생성...")
        
        target_ids = self.candidates['movieId'].unique()
        tagdl_filtered = self.tagdl[self.tagdl['movieId'].isin(target_ids)].copy()
        
        # 태그 관련도 필터링
        tagdl_filtered = tagdl_filtered[abs(tagdl_filtered['score']) >= self.min_tag_relevance]
        
        if len(tagdl_filtered) == 0:
            raise ValueError(f"❌ 기준이 너무 높아 남은 태그가 없습니다.")

        self.tag_matrix = tagdl_filtered.pivot_table(
            index='movieId', columns='tag', values='score', fill_value=0
        )
        
        print(f"   ✓ 최종 분석 대상 영화: {self.tag_matrix.shape[0]:,}편")
        
        if self.tag_matrix.shape[0] < self.num_clusters:
             raise ValueError(f"❌ 남은 영화가 클러스터 수보다 적습니다.")

        final_ids = self.tag_matrix.index
        self.candidates = self.candidates[self.candidates['movieId'].isin(final_ids)]

    def perform_clustering(self):
        print(f"\n🎯 K-Means 클러스터링 (K={self.num_clusters})...")
        
        scaler = StandardScaler()
        tag_matrix_scaled = scaler.fit_transform(self.tag_matrix)
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tag_matrix_scaled)
        
        self.cluster_centers = kmeans.cluster_centers_
        self.tag_names = self.tag_matrix.columns.tolist()
        
        return pd.DataFrame({
            'movieId': self.tag_matrix.index, 
            'cluster': clusters
        })

    def select_representatives(self, cluster_df):
        print(f"\n🏆 대표 영화 선정 (인기순)...")
        
        result = cluster_df.merge(self.candidates, on='movieId', how='left')
        
        # [핵심 변경]
        # 인기도(vote_count) 내림차순(False) 정렬로 복귀
        representatives = result.sort_values(
            ['cluster', 'vote_count'], ascending=[True, False]
        ).groupby('cluster').head(self.movies_per_cluster).copy()
        
        cluster_tags_dict = {}
        cluster_desc_dict = {}
        
        for cid in range(self.num_clusters):
            center = self.cluster_centers[cid]
            top_indices = np.argsort(center)[-5:][::-1]
            
            if len(top_indices) > 0:
                top_tags = [self.tag_names[i] for i in top_indices]
                top_scores = [center[i] for i in top_indices]
                cluster_tags_dict[cid] = ', '.join(top_tags[:3])
                cluster_desc_dict[cid] = ' | '.join([f"{t}({s:.1f})" for t, s in zip(top_tags, top_scores)])
            else:
                cluster_tags_dict[cid] = "No tags"
                cluster_desc_dict[cid] = ""
            
        representatives['cluster_tags'] = representatives['cluster'].map(cluster_tags_dict)
        representatives['cluster_desc'] = representatives['cluster'].map(cluster_desc_dict)
        
        return representatives

    def save_results(self, representatives):
        print(f"\n💾 결과 저장: {self.output_path}")
        
        output_data = {}
        for cid in range(self.num_clusters):
            group = representatives[representatives['cluster'] == cid]
            if group.empty: continue
            
            output_data[f"cluster_{cid}"] = {
                "tags": group.iloc[0]['cluster_tags'],
                "description": group.iloc[0]['cluster_desc'],
                "movies": group[['movieId', 'title_ko', 'vote_count', 'avg_rating']].to_dict('records')
            }
            
        json_path = os.path.join(self.output_path, 'filtered_survey_movies.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        representatives.to_csv(os.path.join(self.output_path, 'filtered_survey_movies.csv'), index=False, encoding='utf-8-sig')
        print(f"   ✓ 저장 완료")

    def run(self):
        try:
            self.load_and_process_data()
            self.build_tag_matrix()
            cluster_df = self.perform_clustering()
            reps = self.select_representatives(cluster_df)
            self.save_results(reps)
            
            # 결과 미리보기 (10개 다 출력)
            print("\n" + "="*80)
            for cid in range(self.num_clusters):
                group = reps[reps['cluster'] == cid]
                if group.empty: continue
                print(f"[{cid:02d}] {group.iloc[0]['cluster_tags']}")
                # 인기도 순 정렬 확인을 위해 vote_count 출력
                for _, row in group.head(10).iterrows():
                    print(f"   - {row['title_ko']} (인기: {row['vote_count']}, 평점: {row['avg_rating']:.2f})")
                    
        except ValueError as e:
            print(f"\n❌ 실행 실패: {e}")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    selector = ColdStartMovieSelector()
    selector.run()