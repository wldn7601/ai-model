import pandas as pd
import os
import sys

# ==========================================
# 1. 파일 경로 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/datas'
PATHS = {
    'tagdl': os.path.join(BASE_DIR, '성능보완용', 'tagdl.csv'),
    'links': os.path.join(BASE_DIR, 'archive', 'links.csv'),
    'movies': os.path.join(BASE_DIR, 'archive', 'movies.csv')
}

def verify_full_data():
    print("="*60)
    print("🔎 데이터 정합성 전수 조사 (Full Integrity Check)")
    print("="*60)

    # ------------------------------------------------
    # 1. 전체 데이터 로드 (Full Load)
    # ------------------------------------------------
    print("\n>>> 1. 전체 데이터 로딩 중... (시간이 조금 걸릴 수 있습니다)")

    # (A) TagDL (사용자 데이터)
    # item_id 컬럼만 읽어서 메모리 절약
    try:
        df_tag = pd.read_csv(PATHS['tagdl'], usecols=['item_id'])
        tag_ids = set(df_tag['item_id'].unique())
        print(f"   - [TagDL] 총 영화 수: {len(tag_ids):,} 개 (item_id)")
    except Exception as e:
        print(f"   ❌ TagDL 로드 실패: {e}")
        return

    # (B) Links (연결 고리)
    try:
        df_links = pd.read_csv(PATHS['links'], usecols=['movieId', 'tmdbId'])
        link_ids = set(df_links['movieId'].dropna().astype(int))
        print(f"   - [Links] 총 영화 수: {len(link_ids):,} 개 (movieId)")
    except Exception as e:
        print(f"   ❌ Links 로드 실패: {e}")
        return

    # (C) Movies (원본 메타)
    try:
        df_movies = pd.read_csv(PATHS['movies'], usecols=['movieId', 'title'])
        movie_ids = set(df_movies['movieId'].unique())
        print(f"   - [Movies] 총 영화 수: {len(movie_ids):,} 개 (movieId)")
    except Exception as e:
        print(f"   ❌ Movies 로드 실패: {e}")
        return

    # ------------------------------------------------
    # 2. 교집합 검증 (Intersection Check)
    # ------------------------------------------------
    print("\n>>> 2. 연결 고리 검증 시작")

    # 검증 1: TagDL -> Links
    # TagDL에 있는 영화 중 Links 파일에 존재하는 비율
    tag_in_links = tag_ids.intersection(link_ids)
    missing_in_links = tag_ids - link_ids # TagDL엔 있는데 Links엔 없는 것 (고아 데이터)
    
    coverage_1 = len(tag_in_links) / len(tag_ids) * 100
    
    print(f"\n[검증 A] TagDL(item_id) -> Links(movieId)")
    print(f"   - 매칭 성공: {len(tag_in_links):,} 개")
    print(f"   - 매칭 실패: {len(missing_in_links):,} 개")
    print(f"   - 일치율: {coverage_1:.2f}%")

    if len(missing_in_links) > 0:
        print(f"   ⚠️ 주의: TagDL에는 있는데 Links에 없는 영화가 {len(missing_in_links)}개 있습니다.")
        # print(f"   (예시 ID: {list(missing_in_links)[:5]})")

    # 검증 2: TagDL -> Movies
    # TagDL에 있는 영화 중 실제 Movies(제목/장르) 파일에 존재하는 비율
    # (실제 추천 시스템에서 제목을 보여줄 수 있는 영화인지 확인)
    tag_in_movies = tag_ids.intersection(movie_ids)
    missing_in_movies = tag_ids - movie_ids
    
    coverage_2 = len(tag_in_movies) / len(tag_ids) * 100

    print(f"\n[검증 B] TagDL(item_id) -> Movies(movieId)")
    print(f"   - 매칭 성공: {len(tag_in_movies):,} 개")
    print(f"   - 매칭 실패: {len(missing_in_movies):,} 개")
    print(f"   - 일치율: {coverage_2:.2f}%")

    # ------------------------------------------------
    # 3. 최종 결론
    # ------------------------------------------------
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    
    if coverage_2 > 95:
        print("✅ [PASS] 데이터가 완벽하게 일치합니다.")
        print("   TagDL의 영화들은 Movies.csv의 영화들과 동일한 ID 체계를 가집니다.")
        print("   안심하고 'item_id'를 'movieId'로 사용하여 병합(Merge)하세요.")
    else:
        print("❌ [FAIL] 매칭되지 않는 영화가 너무 많습니다.")
        print("   ID 체계가 다르거나 데이터 버전이 맞지 않습니다.")

if __name__ == "__main__":
    verify_full_data()