import pandas as pd
import json
import os

# ==========================================
# 1. 파일 경로 설정 (사용자 환경 기준)
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/datas'

FILE_PATHS = {
    'ALS_TRAIN': os.path.join(BASE_DIR, 'train_ratings.csv'),
    'LINKS': os.path.join(BASE_DIR, 'archive', 'links.csv'),
    'CBF_JSON': os.path.join(BASE_DIR, 'trend_data_with_ai_tags.json'),
    'TAG_DL': os.path.join(BASE_DIR, 'scores', 'tagdl.csv') # 누님이 준 딥러닝 스코어
}

def load_and_inspect():
    print("="*60)
    print("DATA RELATIONSHIP INSPECTION REPORT")
    print("="*60)

    # ------------------------------------------------
    # 1. 데이터 로드 및 기본 구조 확인
    # ------------------------------------------------
    
    # (1) LINKS (연결 고리)
    try:
        df_links = pd.read_csv(FILE_PATHS['LINKS'])
        # NaN 제거 및 Int 변환 (tmdbId가 소수점으로 읽히는 것 방지)
        df_links = df_links.dropna(subset=['tmdbId', 'movieId'])
        df_links['tmdbId'] = df_links['tmdbId'].astype(int)
        df_links['movieId'] = df_links['movieId'].astype(int)
        
        print(f"\n[1] Links Data (Key Map)")
        print(f" - Count: {len(df_links):,}")
        print(f" - Columns: {list(df_links.columns)}")
        print(f" - Sample: \n{df_links.head(3).to_string(index=False)}")
    except Exception as e:
        print(f"[Error] Links 로드 실패: {e}")
        return

    # (2) ALS Data (Train Ratings)
    try:
        # movie_idx가 실제로 movieId인지 확인하기 위함
        df_als = pd.read_csv(FILE_PATHS['ALS_TRAIN'], nrows=5) # 헤더 확인용
        print(f"\n[2] ALS Training Data (Behavior)")
        print(f" - Columns: {list(df_als.columns)}")
        # 전체 로드 (ID 확인용)
        df_als_full = pd.read_csv(FILE_PATHS['ALS_TRAIN'], usecols=['movie_idx'])
        unique_als_movies = set(df_als_full['movie_idx'].unique())
        print(f" - Unique Movies in ALS: {len(unique_als_movies):,}")
    except Exception as e:
        print(f"[Error] ALS Data 로드 실패: {e}")
        return

    # (3) CBF Data (JSON Metadata)
    try:
        with open(FILE_PATHS['CBF_JSON'], 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        df_meta = pd.DataFrame(meta_data)
        # 보통 JSON의 'id'는 TMDB ID임
        if 'id' in df_meta.columns:
            df_meta.rename(columns={'id': 'tmdbId'}, inplace=True)
        
        print(f"\n[3] CBF Meta Data (Content)")
        print(f" - Count: {len(df_meta):,}")
        print(f" - Columns (Top 5): {list(df_meta.columns)[:5]}")
    except Exception as e:
        print(f"[Error] CBF JSON 로드 실패: {e}")
        return

    # (4) Tag Genome (Deep Learning Scores)
    try:
        # 파일이 큰 경우 일부만 읽어 컬럼 확인
        df_tag = pd.read_csv(FILE_PATHS['TAG_DL'], nrows=5)
        print(f"\n[4] Tag Genome Data (Deep Learning)")
        print(f" - Columns: {list(df_tag.columns)}")
        
        # 실제 ID 확인을 위해 movie_id 컬럼 로드 (컬럼명이 movieId인지 movie_id인지 체크 필요)
        col_name = 'movieId' if 'movieId' in df_tag.columns else 'movie_id' # 보통 둘 중 하나
        df_tag_ids = pd.read_csv(FILE_PATHS['TAG_DL'], usecols=[col_name])
        unique_tag_movies = set(df_tag_ids[col_name].unique())
        print(f" - Unique Movies with Tags: {len(unique_tag_movies):,}")
    except Exception as e:
        print(f"[Warning] Tag Data 로드 실패 (파일명/경로 확인 필요): {e}")
        unique_tag_movies = set()

    # ------------------------------------------------
    # 2. 관계성(Connectivity) 검증
    # ------------------------------------------------
    print("\n" + "="*60)
    print("RELATIONSHIP ANALYSIS (Join Coverage)")
    print("="*60)

    # A. ALS(MovieLens ID) <-> Links Check
    # train_ratings의 movie_idx가 links의 movieId에 포함되는가?
    als_in_links = len(unique_als_movies.intersection(set(df_links['movieId'])))
    als_coverage = (als_in_links / len(unique_als_movies)) * 100
    
    print(f"\n[Check A] ALS(train_ratings) <-> Links(movieId)")
    print(f" - Matched Movies: {als_in_links:,}")
    print(f" - Coverage: {als_coverage:.2f}%")
    if als_coverage < 5:
        print("  ⚠️ [CRITICAL] ALS 데이터의 ID가 MovieLens ID가 아닌 것 같습니다. (Re-indexing 의심)")
    else:
        print("  ✅ ALS 데이터는 MovieLens ID를 사용 중입니다.")

    # B. CBF(JSON) <-> Links Check
    # JSON의 tmdbId가 links의 tmdbId에 포함되는가?
    unique_meta_ids = set(df_meta['tmdbId'])
    meta_in_links = len(unique_meta_ids.intersection(set(df_links['tmdbId'])))
    meta_coverage = (meta_in_links / len(unique_meta_ids)) * 100

    print(f"\n[Check B] CBF(JSON) <-> Links(tmdbId)")
    print(f" - Matched Movies: {meta_in_links:,}")
    print(f" - Coverage: {meta_coverage:.2f}%")
    if meta_coverage < 50:
        print("  ⚠️ JSON 데이터의 많은 영화가 Links 파일에 없습니다. (최신 영화 누락 가능성)")
    else:
        print("  ✅ JSON 데이터와 ID 매핑이 원활합니다.")

    # C. Tag Genome <-> ALS Check
    # ALS 학습에 쓰는 영화 중 몇 개나 Tag 정보를 가지고 있는가?
    # (Tag Genome은 보통 movieId를 사용한다고 가정)
    als_with_tags = len(unique_als_movies.intersection(unique_tag_movies))
    tag_coverage = (als_with_tags / len(unique_als_movies)) * 100
    
    print(f"\n[Check C] ALS Movies <-> Tag Genome Data")
    print(f" - Movies with Tag Scores: {als_with_tags:,}")
    print(f" - Coverage: {tag_coverage:.2f}%")
    
    print("\n" + "="*60)
    print("FINAL DIAGNOSIS")
    print("="*60)
    
    if als_coverage > 80 and meta_coverage > 50 and tag_coverage > 50:
        print("🚀 [READY] 모든 데이터가 유기적으로 연결되어 있습니다.")
        print("   Recommended Logic: ALS(train_ratings) + CBF(tagdl.csv + trend_data.json)")
    else:
        print("🔧 [FIX NEEDED] 일부 데이터의 연결 고리가 끊어져 있습니다. 위의 Warning을 확인하세요.")

if __name__ == "__main__":
    load_and_inspect()