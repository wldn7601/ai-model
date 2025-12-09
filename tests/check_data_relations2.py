import pandas as pd
import os

# ==========================================
# 1. 파일 경로 설정 (사용자 환경 반영)
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/datas'

FILE_PATHS = {
    # [Archive 폴더] 원본 데이터 그룹
    'LINKS': os.path.join(BASE_DIR, 'archive', 'links.csv'),
    'RATINGS_ORIGIN': os.path.join(BASE_DIR, 'archive', 'ratings.csv'),
    
    # [성능보완용 폴더] 딥러닝 태그 점수
    'TAG_DL': os.path.join(BASE_DIR, '성능보완용', 'tagdl.csv')
}

def inspect_folder_relations():
    print("="*60)
    print("DATA RELATIONSHIP DIAGNOSIS V2")
    print("="*60)

    # ------------------------------------------------
    # 1. [Archive 폴더] 내부 결속력 확인
    # ------------------------------------------------
    print("\n>>> [1] Archive 폴더 내부 관계 점검 (Links <-> Ratings)")
    
    try:
        # Links 로드 (ID 매핑의 기준)
        links = pd.read_csv(FILE_PATHS['LINKS'])
        links = links.dropna(subset=['movieId'])
        links['movieId'] = links['movieId'].astype(int)
        
        # 원본 Ratings 로드 (데이터가 크므로 일부만 로드하여 ID 확인)
        ratings_origin = pd.read_csv(FILE_PATHS['RATINGS_ORIGIN'], usecols=['movieId'], nrows=100000)
        
        # 교집합 확인
        common_ids = set(links['movieId']).intersection(set(ratings_origin['movieId']))
        match_rate = len(common_ids) / len(set(ratings_origin['movieId'])) * 100
        
        print(f" - Archive Ratings Sample ID Match: {match_rate:.2f}%")
        if match_rate > 99:
            print("  ✅ [확인] Archive 폴더 내 데이터들은 'movieId'를 공유합니다.")
        else:
            print("  ⚠️ [경고] Archive 폴더 내에서도 ID가 다를 수 있습니다.")
            
    except Exception as e:
        print(f"  [Error] Archive 데이터 로드 실패: {e}")
        return

    # ------------------------------------------------
    # 2. [성능보완용 폴더]의 소속 확인
    # ------------------------------------------------
    print("\n>>> [2] '성능보완용' 데이터가 'Archive' ID를 사용하는지 점검")
    
    try:
        # TagDL 로드
        if not os.path.exists(FILE_PATHS['TAG_DL']):
            print(f"  [Error] 파일을 찾을 수 없습니다: {FILE_PATHS['TAG_DL']}")
            return

        tag_dl = pd.read_csv(FILE_PATHS['TAG_DL'])
        
        # 컬럼명 확인 ('movieId' vs 'movie_id')
        id_col = 'movieId' if 'movieId' in tag_dl.columns else 'movie_id'
        print(f"  - TagDL ID Column Name: '{id_col}'")
        
        tag_ids = set(tag_dl[id_col].unique())
        print(f"  - TagDL Unique Movies: {len(tag_ids):,}")

        # Archive의 Links와 매칭 시도
        matched_tags = len(tag_ids.intersection(set(links['movieId'])))
        coverage = matched_tags / len(tag_ids) * 100
        
        print(f"  - Archive(Links)와의 일치율: {coverage:.2f}% ({matched_tags:,}개 영화)")
        
        if coverage > 80:
            print("\n🎉 [결론] '성능보완용/tagdl.csv'는 'Archive' 폴더와 같은 ID 체계입니다!")
            print("   -> 즉, 'archive/ratings.csv'와 '성능보완용/tagdl.csv'를 바로 합쳐서 사용할 수 있습니다.")
        else:
            print("\n❌ [결론] '성능보완용/tagdl.csv'는 독자적인 ID를 씁니다.")
            print("   -> 추가적인 매핑 파일이 필요하거나, 데이터 생성 과정을 다시 확인해야 합니다.")

    except Exception as e:
        print(f"  [Error] TagDL 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    inspect_folder_relations()