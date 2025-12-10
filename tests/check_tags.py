import pandas as pd
import os

# --- 경로 설정 (tests 폴더에서 실행 기준) ---
DATA_DIR = "../datas"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")

PATH_2019 = os.path.join(DATA_DIR, "2019_data.json")
PATH_TAGS = os.path.join(ARCHIVE_DIR, "tags.csv")
PATH_GENOME_SCORES = os.path.join(ARCHIVE_DIR, "genome-scores.csv")
PATH_GENOME_TAGS = os.path.join(ARCHIVE_DIR, "genome-tags.csv")

def check_data():
    print(f"📂 데이터 경로 확인: {os.path.abspath(DATA_DIR)}")
    
    # 1. 2019 데이터 로드
    if not os.path.exists(PATH_2019):
        print(f"❌ 파일을 찾을 수 없습니다: {PATH_2019}")
        return
        
    print("⏳ 2019 데이터 로딩 중...")
    df_2019 = pd.read_json(PATH_2019)
    total_movies = len(df_2019)
    print(f"✅ 2019 영화 데이터 수: {total_movies:,}개")
    
    # movieId가 키값입니다.
    movie_ids_2019 = set(df_2019['movieId'].unique())

    # 2. tags.csv (사용자 생성 태그) 확인
    if os.path.exists(PATH_TAGS):
        print("\n⏳ tags.csv 분석 중...")
        df_tags = pd.read_csv(PATH_TAGS)
        
        # 교집합 (2019 데이터에 있는 영화 중 태그가 있는 것)
        matched_tags = df_tags[df_tags['movieId'].isin(movie_ids_2019)]
        matched_count = matched_tags['movieId'].nunique()
        coverage = (matched_count / total_movies) * 100
        
        print(f"   - 태그가 존재하는 영화 수: {matched_count:,}개 ({coverage:.1f}%)")
        
        # 샘플 출력
        if not matched_tags.empty:
            sample_id = matched_tags.iloc[0]['movieId']
            sample_title = df_2019[df_2019['movieId'] == sample_id]['title'].values[0]
            tags = matched_tags[matched_tags['movieId'] == sample_id]['tag'].tolist()
            print(f"   - 예시 [ID: {sample_id}, 제목: {sample_title}]: {tags[:5]} ...")
    else:
        print("❌ tags.csv 파일이 없습니다.")

    # 3. genome-scores.csv (고품질 태그 점수) 확인
    # 파일이 크므로 청크 단위로 읽거나, 필요한 컬럼만 읽어서 메모리 절약
    if os.path.exists(PATH_GENOME_SCORES) and os.path.exists(PATH_GENOME_TAGS):
        print("\n⏳ genome-scores.csv (데이터 큼) 분석 중...")
        
        # 메모리 절약을 위해 movieId만 먼저 로드해서 매칭 여부 확인
        # 실제 사용 시에는 점수 높은 것만 필터링해야 함
        try:
            # unique한 movieId만 빠르게 파악
            df_genome_ids = pd.read_csv(PATH_GENOME_SCORES, usecols=['movieId']).drop_duplicates()
            
            matched_genome_count = df_genome_ids[df_genome_ids['movieId'].isin(movie_ids_2019)].shape[0]
            coverage = (matched_genome_count / total_movies) * 100
            
            print(f"   - 게놈 데이터가 존재하는 영화 수: {matched_genome_count:,}개 ({coverage:.1f}%)")
            
            if coverage > 50:
                print("   👉 [추천] 게놈 데이터 커버리지가 높습니다! 이 데이터를 임베딩에 쓰면 검색 품질이 매우 좋아집니다.")
            else:
                print("   👉 게놈 데이터 커버리지가 낮습니다. tags.csv 위주로 사용하세요.")
                
        except Exception as e:
            print(f"   ⚠️ 게놈 파일 읽기 실패 (메모리 부족 등): {e}")
    else:
        print("❌ genome-scores.csv 또는 genome-tags.csv 파일이 없습니다.")

if __name__ == "__main__":
    check_data()