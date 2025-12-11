"""
조건: 클러스터링 먼저(전체 데이터) -> 그 후 필터링(평점 3.8이하), 인기순 고려, 태그 관련도 0.8
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
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/12-11/results/cluster_first/v1',
                 # --- 설정 ---
                 clustering_vote_threshold=100, # 클러스터링을 위한 최소 인기도 (노이즈 제거용)
                 final_min_vote_count=500,      # 최종 결과 필터링용 인기도
                 max_rating=3.8,                # 최종 결과 필터링용 평점 상한
                 num_clusters=12,
                 movies_per_cluster=10,
                 min_tag_relevance=0.8
                 ): 
        
        self.base_path = base_path
        self.output_path = output_path
        
        # 클러스터링용 기준 (넓게 잡음)
        self.clustering_vote_threshold = clustering_vote_threshold
        
        # 최종 필터링용 기준 (엄격하게 잡음)
        self.final_min_vote_count = final_min_vote_count
        self.max_rating = max_rating
        
        self.num_clusters = num_clusters
        self.movies_per_cluster = movies_per_cluster
        self.min_tag_relevance = min_tag_relevance
        
        os.makedirs(self.output_path, exist_ok=True)
        
        print(f"{'='*60}")
        print("🚀 Cold Start 영화 선정 (Cluster First -> Filter Last)")
        print(f"{'='*60}")
        print(f"1. 전체 지도 생성 : 평점 수 {clustering_vote_threshold}개 이상인 모든 영화로 클러스터링")
        print(f"2. 최종 걸러내기 : 평점 {max_rating}이하 & 평점 수 {final_min_vote_count}개 이상")
        print(f"3. 정렬 기준     : 인기순 (Vote Count)")
        print(f"4. 태그 기준     : 관련도 {min_tag_relevance} 이상")
        print(f"{'='*60}\n")
        
    def load_data_and_cluster_universe(self):
        print("📂 [1단계] 전체 영화 데이터 로드 (클러스터링용)...")
        
        ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'vote_count', 'avg_rating']
        
        # [중요] 여기서는 평점 3.5 이하 필터링을 하지 않습니다!
        # 전체 지도를 그리기 위해 최소한의 인지도만 있으면 다 포함시킵니다.
        self.universe = movie_stats[movie_stats['vote_count'] >= self.clustering_vote_threshold].copy()
        
        print(f"   ✓ 클러스터링 대상 전체 영화: {len(self.universe):,}편")
        
        # 메타데이터 병합
        meta = pd.read_csv(f'{self.base_path}/movies_metadata_restored.csv')
        self.universe['movieId'] = self.universe['movieId'].astype(str)
        meta['movieId'] = meta['movieId'].astype(str)
        
        self.universe = self.universe.merge(
            meta[['movieId', 'title_ko', 'poster_path']], 
            on='movieId', 
            how='inner'
        )
        
        # 태그 데이터 로드
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        self.tagdl.rename(columns={'item_id': 'movieId'}, inplace=True)
        self.tagdl['movieId'] = self.tagdl['movieId'].astype(str)

    def build_tag_matrix(self):
        print("\n🧬 [2단계] 태그 매트릭스 생성...")
        
        target_ids = self.universe['movieId'].unique()
        tagdl_filtered = self.tagdl[self.tagdl['movieId'].isin(target_ids)].copy()
        
        tagdl_filtered = tagdl_filtered[abs(tagdl_filtered['score']) >= self.min_tag_relevance]
        
        if len(tagdl_filtered) == 0:
            raise ValueError(f"❌ 기준이 너무 높아 태그가 없습니다.")

        self.tag_matrix = tagdl_filtered.pivot_table(
            index='movieId', columns='tag', values='score', fill_value=0
        )
        
        print(f"   ✓ 매트릭스 크기: {self.tag_matrix.shape}")
        
        # 매트릭스에 있는 영화만 universe로 확정
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
        
        # 전체 영화에 클러스터 라벨 부여
        self.cluster_results = pd.DataFrame({
            'movieId': self.tag_matrix.index, 
            'cluster': clusters
        })
        
        # 클러스터별 전체 영화 수 확인 (필터링 전)
        print("   📊 필터링 전 그룹별 크기:")
        print(self.cluster_results['cluster'].value_counts().sort_index())

    def filter_and_select_representatives(self):
        print(f"\n🏆 [4단계] 조건 필터링 및 대표 선정...")
        print(f"   조건: 평점 {self.max_rating} 이하 AND 인기 {self.final_min_vote_count} 이상")
        
        # 전체 정보 병합
        full_data = self.cluster_results.merge(self.universe, on='movieId', how='left')
        
        representatives_list = []
        cluster_tags_dict = {}
        cluster_desc_dict = {}
        
        for cid in range(self.num_clusters):
            # 1. 해당 클러스터의 영화 가져오기
            group_movies = full_data[full_data['cluster'] == cid]
            
            # 2. [여기서 필터링 수행]
            filtered_group = group_movies[
                (group_movies['avg_rating'] <= self.max_rating) &
                (group_movies['vote_count'] >= self.final_min_vote_count)
            ].copy()
            
            # 3. 태그 정보 생성 (필터링 전 전체 데이터를 기준으로 중심점 설명)
            center = self.cluster_centers[cid]
            top_indices = np.argsort(center)[-5:][::-1]
            if len(top_indices) > 0:
                top_tags = [self.tag_names[i] for i in top_indices]
                top_scores = [center[i] for i in top_indices]
                tags_str = ', '.join(top_tags[:3])
                desc_str = ' | '.join([f"{t}({s:.1f})" for t, s in zip(top_tags, top_scores)])
            else:
                tags_str = "No tags"
                desc_str = ""
            
            cluster_tags_dict[cid] = tags_str
            cluster_desc_dict[cid] = desc_str
            
            # 4. 결과 확인 (텅 빈 그룹이 있는지 체크)
            if len(filtered_group) == 0:
                print(f"   ⚠️ [Cluster {cid:02d}] '{tags_str}' 그룹은 조건에 맞는 영화가 0개입니다! (전멸)")
                continue
            
            # 5. 인기순 정렬 및 상위 N개 추출
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
        for cid in range(self.num_clusters):
            group = representatives[representatives['cluster'] == cid]
            if group.empty: continue
            
            output_data[f"cluster_{cid}"] = {
                "tags": group.iloc[0]['cluster_tags'],
                "description": group.iloc[0]['cluster_desc'],
                "movies": group[['movieId', 'title_ko', 'vote_count', 'avg_rating']].to_dict('records')
            }
            
        with open(os.path.join(self.output_path, 'filtered_survey_movies.json'), 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        representatives.to_csv(os.path.join(self.output_path, 'filtered_survey_movies.csv'), index=False, encoding='utf-8-sig')
        print(f"   ✓ 저장 완료")

    def run(self):
        try:
            self.load_data_and_cluster_universe()
            self.build_tag_matrix()
            self.perform_clustering()
            reps = self.filter_and_select_representatives()
            self.save_results(reps)
            
            print("\n" + "="*80)
            seen_clusters = reps['cluster'].unique()
            print(f"✅ 총 {len(seen_clusters)}개 그룹이 살아남았습니다. (목표: {self.num_clusters})")
            print("="*80)
            
            for cid in sorted(seen_clusters):
                group = reps[reps['cluster'] == cid]
                print(f"[{cid:02d}] {group.iloc[0]['cluster_tags']}")
                for _, row in group.head(10).iterrows():
                    print(f"   - {row['title_ko']} (평점: {row['avg_rating']:.2f}, 인기: {row['vote_count']})")
                    
        except ValueError as e:
            print(f"\n❌ 실행 실패: {e}")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    selector = ColdStartMovieSelector()
    selector.run()