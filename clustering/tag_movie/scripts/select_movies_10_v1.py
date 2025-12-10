import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os  # 폴더 생성을 위해 추가
import warnings
warnings.filterwarnings('ignore')

"""
10개의 클러스터 군집화에서 1개의 대표 영화만 찾기
-> 클러스터링이 잘 안 됐음
"""


class ColdStartMovieSelector:
    """
    TagDL 기반 대표 영화 선정기
    """
    
    def __init__(self, 
                 base_path='/home/ubuntu/ai-model/movielens_data',
                 output_path='/home/ubuntu/ai-model/clustering/tag_movie/data', # 저장 경로 파라미터 추가
                 min_vote_count=5000,
                 target_count=10,
                 min_tag_relevance=0.1):
        
        self.base_path = base_path
        self.output_path = output_path # 저장 경로 설정
        self.min_vote_count = min_vote_count
        self.target_count = target_count
        self.min_tag_relevance = min_tag_relevance
        
        # 저장 경로가 없으면 자동으로 생성 (mkdir -p)
        os.makedirs(self.output_path, exist_ok=True)
        
        print("🚀 Cold Start 대표 영화 선정 시작")
        print(f"   - 입력 데이터 경로: {self.base_path}")
        print(f"   - 결과 저장 경로: {self.output_path}")
        print(f"   - 최소 평점 수: {min_vote_count:,}")
        print(f"   - 목표 영화 수: {target_count}")
        print(f"   - 최소 태그 관련도: {min_tag_relevance}")
        
    def load_data(self):
        """데이터 로드 (입력 경로 사용)"""
        print("\n📂 데이터 로딩 중...")
        
        self.movies = pd.read_csv(f'{self.base_path}/movies_metadata_restored.csv')
        print(f"   ✓ 영화 메타데이터: {len(self.movies):,}편")
        
        self.ratings = pd.read_csv(f'{self.base_path}/ratings.csv')
        print(f"   ✓ 평점 데이터: {len(self.ratings):,}개")
        
        self.tagdl = pd.read_csv(f'{self.base_path}/tagdl.csv')
        print(f"   ✓ TagDL 데이터: {len(self.tagdl):,}개")
        print(f"     - 고유 태그: {self.tagdl['tag'].nunique()}")
        print(f"     - 고유 영화: {self.tagdl['item_id'].nunique()}")
        
    def calculate_popularity(self):
        """인기도 계산 및 필터링"""
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
        
        self.popular_movies = popularity_with_tags[
            popularity_with_tags['vote_count'] >= self.min_vote_count
        ].copy()
        
        print(f"   ✓ 인기 영화 ({self.min_vote_count}+ 평점): {len(self.popular_movies):,}편")
        
        # 메타데이터 병합
        self.popular_movies = self.popular_movies.merge(
            self.movies[['movieId', 'title_ko', 'poster_path', 'overview']],
            on='movieId',
            how='left'
        )
        
        # 기준 완화 로직
        if len(self.popular_movies) < self.target_count:
            print(f"\n   ⚠️  영화 수가 부족합니다 ({len(self.popular_movies)}편)")
            
            for reduced_count in [3000, 2000, 1000, 500]:
                self.popular_movies = popularity_with_tags[
                    popularity_with_tags['vote_count'] >= reduced_count
                ].copy()
                
                if len(self.popular_movies) >= self.target_count:
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
            self.target_count = self.tag_matrix.shape[0]
            print(f"   ⚠️  클러스터 수 조정: {old_target} → {self.target_count}")
        
    def perform_clustering(self):
        """K-Means 클러스터링 수행"""
        print(f"\n🎯 K-Means 클러스터링 수행 중 (K={self.target_count})...")
        
        scaler = StandardScaler()
        tag_matrix_scaled = scaler.fit_transform(self.tag_matrix)
        
        kmeans = KMeans(
            n_clusters=self.target_count,
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        
        clusters = kmeans.fit_predict(tag_matrix_scaled)
        
        print(f"   ✓ 클러스터링 완료: {self.target_count}개 그룹")
        
        cluster_df = pd.DataFrame({
            'movieId': self.tag_matrix.index,
            'cluster': clusters
        })
        
        self.cluster_centers = kmeans.cluster_centers_
        self.scaler = scaler
        self.tag_names = self.tag_matrix.columns.tolist()
        
        return cluster_df
        
    def select_representatives(self, cluster_df):
        """각 클러스터별 대표 영화 선정"""
        print("\n🏆 클러스터별 대표 영화 선정 중...")
        
        result = cluster_df.merge(
            self.popular_movies,
            on='movieId',
            how='left'
        )
        
        representatives = result.sort_values(
            ['cluster', 'vote_count'], 
            ascending=[True, False]
        ).groupby('cluster').head(1).copy()
        
        # 클러스터별 주요 태그 분석
        cluster_tags_list = []
        
        for cluster_id in range(self.target_count):
            center = self.cluster_centers[cluster_id]
            top_indices = np.argsort(center)[-3:][::-1]
            top_tags = [self.tag_names[i] for i in top_indices]
            cluster_tags_list.append(', '.join(top_tags))
        
        representatives['cluster_tags'] = representatives['cluster'].map(
            dict(enumerate(cluster_tags_list))
        )
        
        print(f"   ✓ {len(representatives)}편 대표 영화 선정 완료")
        
        representatives = representatives.sort_values('vote_count', ascending=False)
        
        return representatives.reset_index(drop=True)
    
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
            raise
    
    def save_results(self, representatives):
        """결과 저장 - 저장 경로(output_path) 사용"""
        print(f"\n💾 결과 저장 중... (경로: {self.output_path})")
        
        # 1. CSV 저장 (전체 정보) - base_path 대신 output_path 사용
        csv_path = os.path.join(self.output_path, 'cold_start_survey_movies.csv')
        representatives.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✓ CSV 저장: {csv_path}")
        
        # 2. JSON 저장 (프론트엔드용)
        json_path = os.path.join(self.output_path, 'cold_start_survey_movies.json')
        survey_json = representatives[[
            'movieId', 'title_ko', 'poster_path', 'cluster_tags', 
            'vote_count', 'avg_rating', 'cluster'
        ]].to_dict('records')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(survey_json, f, ensure_ascii=False, indent=2)
        print(f"   ✓ JSON 저장: {json_path}")
        
        # 3. 영화 ID 리스트만 별도 저장
        id_list_path = os.path.join(self.output_path, 'cold_start_movie_ids.txt')
        with open(id_list_path, 'w') as f:
            for movie_id in representatives['movieId']:
                f.write(f"{movie_id}\n")
        print(f"   ✓ ID 리스트 저장: {id_list_path}")
        
        # 4. 파이썬 모듈용 설정 파일 저장
        config_path = os.path.join(self.output_path, 'cold_start_config.py')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("# Cold Start Survey Movies Configuration\n")
            f.write("# Auto-generated file - Do not edit manually\n\n")
            f.write(f"SURVEY_MOVIE_IDS = {representatives['movieId'].tolist()}\n\n")
            f.write(f"SURVEY_MOVIE_COUNT = {len(representatives)}\n\n")
            f.write("SURVEY_MOVIES_DETAILS = [\n")
            for idx, row in representatives.iterrows():
                f.write("    {\n")
                f.write(f"        'movieId': {row['movieId']},\n")
                f.write(f"        'title': '{row['title_ko']}',\n")
                f.write(f"        'cluster': {row['cluster']},\n")
                f.write(f"        'cluster_tags': '{row['cluster_tags']}',\n")
                f.write(f"        'vote_count': {row['vote_count']},\n")
                f.write(f"        'avg_rating': {row['avg_rating']:.2f}\n")
                f.write("    },\n")
            f.write("]\n")
        print(f"   ✓ Python 설정 저장: {config_path}")
        
        # 5. 결과 출력
        self._print_results(representatives)
        
        return {
            'csv_path': csv_path,
            'json_path': json_path,
            'id_list_path': id_list_path,
            'config_path': config_path
        }
    
    def _print_results(self, representatives):
        """결과 출력"""
        print("\n" + "="*100)
        print(f"📋 선정된 대표 영화 목록 (총 {len(representatives)}편)")
        print("="*100)
        
        for idx, row in representatives.iterrows():
            print(f"\n🎬 #{idx+1} - 클러스터 {row['cluster']:02d} [{row['cluster_tags']}]")
            print(f"   제목: {row['title_ko']}")
            print(f"   영화ID: {row['movieId']}")
            print(f"   평점 수: {row['vote_count']:,}개")
            print(f"   평균 평점: {row['avg_rating']:.2f}/5.0")
            if pd.notna(row['poster_path']):
                print(f"   포스터: https://image.tmdb.org/t/p/w500{row['poster_path']}")
            print("-"*100)


def main():
    """메인 실행 함수"""
    
    # 설정: 저장 경로 지정
    INPUT_PATH = '/home/ubuntu/ai-model/movielens_data'
    OUTPUT_PATH = '/home/ubuntu/ai-model/clustering/tag_movie/data'
    
    configs = [
        {'min_vote_count': 5000, 'target_count': 10, 'min_tag_relevance': 0.1},
        {'min_vote_count': 3000, 'target_count': 10, 'min_tag_relevance': 0.1},
        {'min_vote_count': 2000, 'target_count': 10, 'min_tag_relevance': 0.05},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*100}")
        print(f"실행 #{i+1}: {config}")
        print(f"{'='*100}\n")
        
        try:
            selector = ColdStartMovieSelector(
                base_path=INPUT_PATH,      # 입력 데이터 위치
                output_path=OUTPUT_PATH,   # 출력 데이터 위치 (여기서 설정)
                **config
            )
            
            representatives = selector.run()
            saved_files = selector.save_results(representatives)
            
            print(f"\n✅ 실행 #{i+1} 성공!")
            print(f"\n📦 생성된 파일:")
            for key, path in saved_files.items():
                print(f"   - {key}: {path}")
            
            break
            
        except ValueError as e:
            print(f"\n⚠️  실행 #{i+1} 실패: {e}")
            if i < len(configs) - 1:
                print(f"   다음 설정으로 재시도합니다...\n")
        
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류: {e}")
            raise


if __name__ == "__main__":
    main()