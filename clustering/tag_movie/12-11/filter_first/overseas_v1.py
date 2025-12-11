"""

조건 1
평점 3.8이하, 인기순은 고려

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
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/12-11/results/filter_first/v1',
                 # --- 사용자 요청 설정 ---
                 min_vote_count=500,     # 인기도 고려 (최소 500명 이상 평가)
                 max_rating=3.8,         # [핵심] 4.0 초과는 제외 (즉, 4.0 이하만 포함)
                 num_clusters=12,        # 클러스터 12개
                 movies_per_cluster=10,  # 각 대표영화 10개
                 min_tag_relevance=0.8   # [핵심] 태그 관련도 0.8 (매우 엄격)
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
        print("🚀 Cold Start 영화 선정 (평점 상한 & 엄격한 태그 버전)")
        print(f"{'='*60}")
        print(f"1. 평점 조건 : {max_rating}점 이하만 선택 (초고득점 명작 제외)")
        print(f"2. 인기도    : 평점 개수 {min_vote_count}개 이상 (인기순 정렬)")
        print(f"3. 태그 기준 : 관련도 {min_tag_relevance} 이상 (매우 엄격)")
        print(f"4. 목표 개수 : 12개 그룹 x 10개 = 총 {self.total_target}편")
        print(f"{'='*60}\n")
        
    def load_and_process_data(self):
        print("📂 데이터 로딩 및 집계 중...")
        
        # 1. Ratings 로드 및 집계
        ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        
        # 영화별 통계 계산
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'vote_count', 'avg_rating']
        
        print(f"   ✓ 전체 영화 수: {len(movie_stats):,}편")
        
        # ---------------------------------------------------------
        # [핵심 필터링 로직]
        # ---------------------------------------------------------
        # 1. 인기도 필터 (사람들이 많이 본 것)
        # 2. 평점 필터 (4.0점 이하인 것만)
        self.candidates = movie_stats[
            (movie_stats['vote_count'] >= self.min_vote_count) & 
            (movie_stats['avg_rating'] <= self.max_rating)
        ].copy()
        
        print(f"   ✓ 1차 필터링 완료 (인기 {self.min_vote_count}↑ & 평점 {self.max_rating}↓): {len(self.candidates):,}편")
        
        # 3. 메타데이터 병합 (제목 확보)
        meta = pd.read_csv(f'{self.base_path}/movies_metadata_restored.csv')
        
        # ID 타입 통일
        self.candidates['movieId'] = self.candidates['movieId'].astype(str)
        meta['movieId'] = meta['movieId'].astype(str)
        
        # 병합
        self.candidates = self.candidates.merge(
            meta[['movieId', 'title_ko', 'poster_path']], 
            on='movieId', 
            how='inner'
        )
        print(f"   ✓ 메타데이터 병합 후 후보: {len(self.candidates):,}편")

        # 4. TagDL 로드
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        self.tagdl.rename(columns={'item_id': 'movieId'}, inplace=True)
        self.tagdl['movieId'] = self.tagdl['movieId'].astype(str)

    def build_tag_matrix(self):
        print("\n🧬 태그 매트릭스 생성 (엄격 모드)...")
        
        target_ids = self.candidates['movieId'].unique()
        
        # 후보 영화의 태그만 가져오기
        tagdl_filtered = self.tagdl[self.tagdl['movieId'].isin(target_ids)].copy()
        print(f"   - 필터링 전 태그 데이터 수: {len(tagdl_filtered):,}개")
        
        # [핵심] 태그 관련도 0.8 이상 필터링
        tagdl_filtered = tagdl_filtered[abs(tagdl_filtered['score']) >= self.min_tag_relevance]
        print(f"   - 필터링 후 태그 데이터 수(score >= {self.min_tag_relevance}): {len(tagdl_filtered):,}개")
        
        if len(tagdl_filtered) == 0:
            raise ValueError(f"❌ 태그 관련도 기준({self.min_tag_relevance})이 너무 높아 남은 데이터가 없습니다! 기준을 낮춰주세요.")

        # 매트릭스 생성
        self.tag_matrix = tagdl_filtered.pivot_table(
            index='movieId', columns='tag', values='score', fill_value=0
        )
        
        # 태그가 너무 엄격해서 다 탈락하고 남은 영화 수 확인
        print(f"   ✓ 최종 분석 대상 영화: {self.tag_matrix.shape[0]:,}편 (매트릭스 크기: {self.tag_matrix.shape})")
        
        if self.tag_matrix.shape[0] < self.num_clusters:
             raise ValueError(f"❌ 남은 영화({self.tag_matrix.shape[0]}편)가 클러스터 수({self.num_clusters}개)보다 적습니다.")

        # 후보군 업데이트 (매트릭스에 살아남은 영화만)
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
        
        return pd.DataFrame({'movieId': self.tag_matrix.index, 'cluster': clusters})

    def select_representatives(self, cluster_df):
        print(f"\n🏆 대표 영화 선정...")
        
        result = cluster_df.merge(self.candidates, on='movieId', how='left')
        
        # 정렬: 클러스터 -> 인기도(vote_count) 내림차순
        # (평점은 이미 4.0 이하로 걸러졌으므로, 그 중에서 인기 많은 순으로 뽑음)
        representatives = result.sort_values(
            ['cluster', 'vote_count'], ascending=[True, False]
        ).groupby('cluster').head(self.movies_per_cluster).copy()
        
        cluster_tags_dict = {}
        cluster_desc_dict = {}
        
        for cid in range(self.num_clusters):
            center = self.cluster_centers[cid]
            top_indices = np.argsort(center)[-5:][::-1]
            
            # 태그가 없을 수도 있으므로 예외처리
            if len(top_indices) > 0:
                top_tags = [self.tag_names[i] for i in top_indices]
                top_scores = [center[i] for i in top_indices]
                cluster_tags_dict[cid] = ', '.join(top_tags[:3])
                cluster_desc_dict[cid] = ' | '.join([f"{t}({s:.1f})" for t, s in zip(top_tags, top_scores)])
            else:
                cluster_tags_dict[cid] = "No dominant tags"
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
            
            # 결과 미리보기
            print("\n" + "="*80)
            for cid in range(self.num_clusters):
                group = reps[reps['cluster'] == cid]
                if group.empty: continue
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