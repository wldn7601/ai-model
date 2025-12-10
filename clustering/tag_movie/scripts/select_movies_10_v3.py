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
    """
    
    def __init__(self, 
                 base_path='/home/ubuntu/ai-model/movielens_data',
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/data',
                 min_vote_count=5000,  # 100개를 뽑아야 하므로 기준 완화
                 num_clusters=15,  # 클러스터 개수
                 movies_per_cluster=1,  # 클러스터당 영화 수
                 min_tag_relevance=0.8):  # 태그 관련도 기준 상향
        
        self.base_path = base_path
        self.output_path = output_path
        self.min_vote_count = min_vote_count
        self.num_clusters = num_clusters
        self.movies_per_cluster = movies_per_cluster
        self.total_target = num_clusters * movies_per_cluster
        self.min_tag_relevance = min_tag_relevance
        
        os.makedirs(self.output_path, exist_ok=True)
        
        print("🚀 Cold Start 대표 영화 선정 시작 (개선 버전)")
        print(f"   - 입력 데이터 경로: {self.base_path}")
        print(f"   - 결과 저장 경로: {self.output_path}")
        print(f"   - 최소 평점 수: {min_vote_count:,}")
        print(f"   - 클러스터 수: {num_clusters}개")
        print(f"   - 클러스터당 영화 수: {movies_per_cluster}개")
        print(f"   - 총 목표 영화 수: {self.total_target}개")
        print(f"   - 최소 태그 관련도: {min_tag_relevance}")
        
    def load_data(self):
        """데이터 로드"""
        print("\n📂 데이터 로딩 중...")
        
        self.movies = pd.read_csv(f'{self.base_path}/movies_metadata_restored.csv')
        print(f"   ✓ 영화 메타데이터: {len(self.movies):,}편")
        
        self.ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        print(f"   ✓ 평점 데이터: {len(self.ratings):,}개")
        
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        print(f"   ✓ TagDL 데이터: {len(self.tagdl):,}개")
        
    def calculate_popularity(self):
        """인기도 계산 및 필터링 (더 많은 영화 확보)"""
        print("\n📊 인기도 계산 및 데이터 정제 중...")
        
        popularity = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        popularity.columns = ['movieId', 'vote_count', 'avg_rating']
        
        movies_with_tags = set(self.tagdl['item_id'].unique())
        print(f"   ✓ 태그 데이터 보유 영화: {len(movies_with_tags):,}편")
        
        popularity_with_tags = popularity[
            popularity['movieId'].isin(movies_with_tags)
        ]
        
        # 필요한 영화 수를 확보하기 위해 여유있게 필터링
        # 목표의 3배 정도 확보 (300개)
        target_candidate_count = self.total_target * 3
        
        self.popular_movies = popularity_with_tags[
            popularity_with_tags['vote_count'] >= self.min_vote_count
        ].copy()
        
        print(f"   ✓ 인기 영화 ({self.min_vote_count}+ 평점): {len(self.popular_movies):,}편")
        
        # 기준 완화 로직
        if len(self.popular_movies) < target_candidate_count:
            print(f"\n   ⚠️  후보 영화 수가 부족합니다 (목표: {target_candidate_count}편)")
            
            for reduced_count in [2000, 1500, 1000, 500]:
                self.popular_movies = popularity_with_tags[
                    popularity_with_tags['vote_count'] >= reduced_count
                ].copy()
                
                if len(self.popular_movies) >= target_candidate_count:
                    print(f"   ✓ 기준 완화: {reduced_count}+ 평점으로 조정")
                    self.min_vote_count = reduced_count
                    break
            
            print(f"   ✓ 최종 후보 영화: {len(self.popular_movies):,}편")
        
        # 메타데이터 병합
        self.popular_movies = self.popular_movies.merge(
            self.movies[['movieId', 'title_ko', 'poster_path', 'overview']],
            on='movieId',
            how='left'
        )
        
        # 충분한 영화 확보 확인
        if len(self.popular_movies) < self.total_target:
            raise ValueError(
                f"데이터 부족: {self.total_target}개 필요, {len(self.popular_movies)}개만 존재"
            )
    
    def build_tag_matrix(self):
        """태그 매트릭스 생성 (개선)"""
        print("\n🧬 태그 매트릭스 생성 중...")
        
        popular_ids = self.popular_movies['movieId'].unique()
        tagdl_filtered = self.tagdl[
            self.tagdl['item_id'].isin(popular_ids)
        ].copy()
        
        print(f"   - 필터링 전 태그 데이터: {len(tagdl_filtered):,}개")
        
        # 태그 관련도 필터링 (더 엄격하게)
        tagdl_filtered = tagdl_filtered[
            abs(tagdl_filtered['score']) >= self.min_tag_relevance
        ]
        
        print(f"   - 필터링 후 태그 데이터: {len(tagdl_filtered):,}개")
        
        # 피벗 테이블 생성
        self.tag_matrix = tagdl_filtered.pivot_table(
            index='item_id',
            columns='tag',
            values='score',
            fill_value=0
        )
        
        print(f"   ✓ 태그 매트릭스 shape: {self.tag_matrix.shape}")
        print(f"     - 영화: {self.tag_matrix.shape[0]:,}편")
        print(f"     - 태그 차원: {self.tag_matrix.shape[1]:,}개")
        
        # 영화가 충분한지 확인
        if self.tag_matrix.shape[0] < self.total_target:
            print(f"   ⚠️  태그 관련도 기준을 낮춥니다...")
            
            # 기준을 낮춰서 재시도
            for lower_threshold in [0.1, 0.05, 0.01]:
                tagdl_filtered = self.tagdl[
                    self.tagdl['item_id'].isin(popular_ids)
                ]
                tagdl_filtered = tagdl_filtered[
                    abs(tagdl_filtered['score']) >= lower_threshold
                ]
                
                self.tag_matrix = tagdl_filtered.pivot_table(
                    index='item_id',
                    columns='tag',
                    values='score',
                    fill_value=0
                )
                
                if self.tag_matrix.shape[0] >= self.total_target:
                    self.min_tag_relevance = lower_threshold
                    print(f"   ✓ 기준 완화: {lower_threshold}로 조정")
                    print(f"   ✓ 최종 태그 매트릭스: {self.tag_matrix.shape}")
                    break
        
    def perform_clustering(self):
        """K-Means 클러스터링 수행"""
        print(f"\n🎯 K-Means 클러스터링 수행 중 (K={self.num_clusters})...")
        
        # 표준화
        scaler = StandardScaler()
        tag_matrix_scaled = scaler.fit_transform(self.tag_matrix)
        
        # K-Means 클러스터링
        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        
        clusters = kmeans.fit_predict(tag_matrix_scaled)
        
        print(f"   ✓ 클러스터링 완료: {self.num_clusters}개 그룹")
        print(f"   ✓ Inertia: {kmeans.inertia_:.2f}")
        
        # 클러스터 정보 저장
        cluster_df = pd.DataFrame({
            'movieId': self.tag_matrix.index,
            'cluster': clusters
        })
        
        # 클러스터별 영화 수 확인
        cluster_counts = cluster_df['cluster'].value_counts().sort_index()
        print(f"\n   📊 클러스터별 영화 분포:")
        for cluster_id in range(self.num_clusters):
            count = cluster_counts.get(cluster_id, 0)
            print(f"      클러스터 {cluster_id:02d}: {count:3d}편")
        
        # 저장
        self.cluster_centers = kmeans.cluster_centers_
        self.scaler = scaler
        self.tag_names = self.tag_matrix.columns.tolist()
        
        return cluster_df
        
    def select_representatives(self, cluster_df):
        """각 클러스터별 대표 영화 선정 (클러스터당 10개)"""
        print(f"\n🏆 클러스터별 대표 영화 선정 중 (각 {self.movies_per_cluster}개)...")
        
        # 클러스터 + 인기도 정보 병합
        result = cluster_df.merge(
            self.popular_movies,
            on='movieId',
            how='left'
        )
        
        # 각 클러스터에서 인기도 순으로 N개씩 선택
        representatives = result.sort_values(
            ['cluster', 'vote_count'], 
            ascending=[True, False]
        ).groupby('cluster').head(self.movies_per_cluster).copy()
        
        # 클러스터별 주요 태그 분석
        cluster_tags_dict = {}
        cluster_descriptions = {}
        
        for cluster_id in range(self.num_clusters):
            center = self.cluster_centers[cluster_id]
            
            # 상위 5개 태그 추출 (더 많은 정보)
            top_indices = np.argsort(center)[-5:][::-1]
            top_tags = [self.tag_names[i] for i in top_indices]
            top_scores = [center[i] for i in top_indices]
            
            # 태그 설명 생성
            cluster_tags_dict[cluster_id] = ', '.join(top_tags[:3])  # 상위 3개만 표시
            
            # 상세 설명
            tag_details = [f"{tag}({score:.2f})" for tag, score in zip(top_tags, top_scores)]
            cluster_descriptions[cluster_id] = ' | '.join(tag_details)
        
        representatives['cluster_tags'] = representatives['cluster'].map(cluster_tags_dict)
        representatives['cluster_description'] = representatives['cluster'].map(cluster_descriptions)
        
        # 클러스터 내 순위 추가
        representatives['rank_in_cluster'] = representatives.groupby('cluster').cumcount() + 1
        
        print(f"   ✓ 총 {len(representatives)}편 선정 완료")
        
        # 클러스터별 선정 결과 요약
        print(f"\n   📊 클러스터별 선정 결과:")
        for cluster_id in range(self.num_clusters):
            cluster_movies = representatives[representatives['cluster'] == cluster_id]
            print(f"      클러스터 {cluster_id:02d}: {len(cluster_movies)}편 - [{cluster_tags_dict[cluster_id]}]")
        
        return representatives.sort_values(['cluster', 'rank_in_cluster'])
    
    def run(self):
        """전체 실행"""
        try:
            self.load_data()
            self.calculate_popularity()
            self.build_tag_matrix()
            
            cluster_df = self.perform_clustering()
            representatives = self.select_representatives(cluster_df)
            
            return representatives
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_results(self, representatives):
        """결과 저장"""
        print(f"\n💾 결과 저장 중... (경로: {self.output_path})")
        
        # 1. CSV 저장 (전체 정보)
        csv_path = os.path.join(self.output_path, 'cold_start_survey_movies.csv')
        representatives.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✓ CSV 저장: {csv_path}")
        
        # 2. JSON 저장 (프론트엔드용 - 클러스터별 그룹화)
        json_path = os.path.join(self.output_path, 'cold_start_survey_movies.json')
        
        # 클러스터별로 그룹화
        clusters_json = {}
        for cluster_id in range(self.num_clusters):
            cluster_movies = representatives[representatives['cluster'] == cluster_id]
            
            clusters_json[f"cluster_{cluster_id}"] = {
                'cluster_id': int(cluster_id),
                'cluster_tags': cluster_movies.iloc[0]['cluster_tags'],
                'cluster_description': cluster_movies.iloc[0]['cluster_description'],
                'movie_count': len(cluster_movies),
                'movies': cluster_movies[[
                    'movieId', 'title_ko', 'poster_path', 
                    'vote_count', 'avg_rating', 'rank_in_cluster'
                ]].to_dict('records')
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_json, f, ensure_ascii=False, indent=2)
        print(f"   ✓ JSON 저장 (클러스터별): {json_path}")
        
        # 3. 평탄화된 JSON (단순 리스트)
        flat_json_path = os.path.join(self.output_path, 'cold_start_survey_movies_flat.json')
        survey_json = representatives[[
            'movieId', 'title_ko', 'poster_path', 'cluster', 'cluster_tags',
            'vote_count', 'avg_rating', 'rank_in_cluster'
        ]].to_dict('records')
        
        with open(flat_json_path, 'w', encoding='utf-8') as f:
            json.dump(survey_json, f, ensure_ascii=False, indent=2)
        print(f"   ✓ JSON 저장 (평탄화): {flat_json_path}")
        
        # 4. 클러스터별 영화 ID 리스트
        cluster_ids_path = os.path.join(self.output_path, 'cluster_movie_ids.json')
        cluster_ids_dict = {}
        for cluster_id in range(self.num_clusters):
            cluster_movies = representatives[representatives['cluster'] == cluster_id]
            cluster_ids_dict[f"cluster_{cluster_id}"] = cluster_movies['movieId'].tolist()
        
        with open(cluster_ids_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_ids_dict, f, indent=2)
        print(f"   ✓ 클러스터별 ID 저장: {cluster_ids_path}")
        
        # 5. Python 설정 파일
        config_path = os.path.join(self.output_path, 'cold_start_config.py')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("# Cold Start Survey Movies Configuration\n")
            f.write("# Auto-generated file - Do not edit manually\n\n")
            f.write(f"NUM_CLUSTERS = {self.num_clusters}\n")
            f.write(f"MOVIES_PER_CLUSTER = {self.movies_per_cluster}\n")
            f.write(f"TOTAL_SURVEY_MOVIES = {len(representatives)}\n\n")
            
            # 클러스터별 영화 ID
            f.write("CLUSTER_MOVIE_IDS = {\n")
            for cluster_id in range(self.num_clusters):
                cluster_movies = representatives[representatives['cluster'] == cluster_id]
                f.write(f"    {cluster_id}: {cluster_movies['movieId'].tolist()},\n")
            f.write("}\n\n")
            
            # 전체 영화 ID 리스트
            f.write(f"ALL_SURVEY_MOVIE_IDS = {representatives['movieId'].tolist()}\n\n")
            
            # 클러스터 설명
            f.write("CLUSTER_DESCRIPTIONS = {\n")
            for cluster_id in range(self.num_clusters):
                cluster_movies = representatives[representatives['cluster'] == cluster_id]
                tags = cluster_movies.iloc[0]['cluster_tags']
                f.write(f"    {cluster_id}: '{tags}',\n")
            f.write("}\n")
        
        print(f"   ✓ Python 설정 저장: {config_path}")
        
        # 6. 결과 출력
        self._print_results(representatives)
        
        return {
            'csv_path': csv_path,
            'json_path': json_path,
            'flat_json_path': flat_json_path,
            'cluster_ids_path': cluster_ids_path,
            'config_path': config_path
        }
    
    def _print_results(self, representatives):
        """결과 출력"""
        print("\n" + "="*100)
        print(f"📋 선정된 대표 영화 목록 (총 {len(representatives)}편)")
        print("="*100)
        
        for cluster_id in range(self.num_clusters):
            cluster_movies = representatives[representatives['cluster'] == cluster_id]
            
            if len(cluster_movies) == 0:
                continue
            
            print(f"\n{'='*100}")
            print(f"🎬 클러스터 {cluster_id:02d} [{cluster_movies.iloc[0]['cluster_tags']}]")
            print(f"   설명: {cluster_movies.iloc[0]['cluster_description']}")
            print(f"   영화 수: {len(cluster_movies)}편")
            print(f"{'='*100}")
            
            for idx, row in cluster_movies.iterrows():
                print(f"  #{row['rank_in_cluster']:2d}. {row['title_ko']}")
                print(f"       영화ID: {row['movieId']}")
                print(f"       평점: {row['avg_rating']:.2f}/5.0 ({row['vote_count']:,}개)")
                print(f"       포스터: https://image.tmdb.org/t/p/w500{row['poster_path']}")
                print()


def main():
    """
    메인 실행 함수 (단일 설정 모드)
    - Fallback(재시도) 로직을 제거하고, 지정된 설정으로 딱 한 번만 실행합니다.
    """
    
    # 1. 경로 설정
    INPUT_PATH = '/home/ubuntu/ai-model/movielens_data'
    OUTPUT_PATH = '/home/ubuntu/ai-model/clustering/tag_movie/data'
    
    # 2. 사용자가 원하는 단 하나의 엄격한 설정
    strict_config = {
        'min_vote_count': 5000,       # 평점 5000개 이상 (엄격)
        'num_clusters': 15,           # 15개 그룹
        'movies_per_cluster': 1,     # 그룹당 1개 (총 15개)
        'min_tag_relevance': 0.8     # 태그 관련도 0.8 이상 (엄격)
    }
    
    print(f"\n{'='*100}")
    print(f"🚀 단일 설정으로 실행 시작: {strict_config}")
    print(f"{'='*100}\n")
    
    try:
        # 객체 생성 (strict_config의 값들이 __init__의 기본값을 덮어씁니다)
        selector = ColdStartMovieSelector(
            base_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            **strict_config
        )
        
        # 실행
        representatives = selector.run()
        
        # 결과 저장
        saved_files = selector.save_results(representatives)
        
        print(f"\n✅ 실행 성공! (기준 완화 없이 완료)")
        print(f"\n📦 생성된 파일:")
        for key, path in saved_files.items():
            print(f"   - {key}: {path}")
            
    except ValueError as e:
        print(f"\n❌ 실패: 조건에 맞는 영화가 부족합니다.")
        print(f"   원인: {e}")
        print("   -> min_vote_count나 min_tag_relevance를 조금 낮춰서 다시 시도해보세요.")
    
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()