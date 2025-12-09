import pandas as pd
import os

# 1. 경로 설정
BASE_DIR = '/home/ubuntu/ai-model/datas'
TAG_DL_PATH = os.path.join(BASE_DIR, '성능보완용', 'tagdl.csv')
LINKS_PATH = os.path.join(BASE_DIR, 'archive', 'links.csv')

def check_and_merge():
    print(">>> TagDL 데이터 컬럼 및 병합 테스트")
    
    # 1. TagDL 파일 로드 (첫 5줄만)
    try:
        tag_dl = pd.read_csv(TAG_DL_PATH, nrows=5)
        print(f"1. TagDL 컬럼 목록: {list(tag_dl.columns)}")
        
        # 'item_id'가 있는지 확인
        if 'item_id' in tag_dl.columns:
            join_col = 'item_id'
            print(f"   ✅ 'item_id' 컬럼을 발견했습니다. (MovieLens ID와 매칭 예정)")
        elif 'movieId' in tag_dl.columns:
            join_col = 'movieId'
            print(f"   ✅ 'movieId' 컬럼을 발견했습니다.")
        else:
            print("   ❌ ID로 추정되는 컬럼(item_id, movieId)이 없습니다.")
            return
            
    except Exception as e:
        print(f"   ❌ TagDL 로드 실패: {e}")
        return

    # 2. 실제 데이터 병합 테스트 (Links와 결합)
    print("\n>>> ID 매핑 검증 (Links <-> TagDL)")
    
    # 전체 로드
    tag_dl_full = pd.read_csv(TAG_DL_PATH, usecols=[join_col])
    links = pd.read_csv(LINKS_PATH)
    
    # Links의 movieId와 TagDL의 join_col 비교
    links_ids = set(links['movieId'].dropna().astype(int))
    tag_ids = set(tag_dl_full[join_col].dropna().astype(int))
    
    matched = len(links_ids.intersection(tag_ids))
    coverage = matched / len(tag_ids) * 100
    
    print(f" - TagDL 고유 영화 수: {len(tag_ids):,}")
    print(f" - MovieLens와 매칭된 영화 수: {matched:,}")
    print(f" - 일치율(Coverage): {coverage:.2f}%")
    
    if coverage > 90:
        print("\n🚀 [확정] 'tagdl.csv'의 'item_id'는 MovieLens 'movieId'와 동일합니다.")
        print("   이제 하이브리드 모델 학습을 바로 시작할 수 있습니다.")
    else:
        print("\n⚠️ [주의] 일치율이 낮습니다. 데이터 확인이 필요합니다.")

if __name__ == "__main__":
    check_and_merge()