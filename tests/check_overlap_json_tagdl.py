import pandas as pd
import json
import os

# ==========================================
# 1. 파일 경로 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/datas'
# 경로 주의: tagdl.csv는 '성능보완용' 폴더 안에 있습니다.
TAG_DL_PATH = os.path.join(BASE_DIR, '성능보완용', 'tagdl.csv')
JSON_2019_PATH = os.path.join(BASE_DIR, '2019_data.json')

def check_overlap():
    print("="*60)
    print("🔎 데이터 교집합(Overlap) 분석 리포트")
    print("   [Target 1]: tagdl.csv (AI 엔진)")
    print("   [Target 2]: 2019_data.json (메타데이터)")
    print("="*60)

    # ------------------------------------------------
    # 1. TagDL 로드 (AI 엔진 데이터)
    # ------------------------------------------------
    print("\n>>> 1. TagDL 데이터 로딩 중...")
    try:
        df_tag = pd.read_csv(TAG_DL_PATH)
        # item_id를 movieId로 통일
        if 'item_id' in df_tag.columns:
            df_tag.rename(columns={'item_id': 'movieId'}, inplace=True)
            
        tag_ids = set(df_tag['movieId'].unique())
        print(f"   - TagDL 총 영화 수: {len(tag_ids):,}개 (분석 가능)")
    except Exception as e:
        print(f"   ❌ TagDL 로드 실패: {e}")
        return

    # ------------------------------------------------
    # 2. 2019_data.json 로드 (표출용 데이터)
    # ------------------------------------------------
    print("\n>>> 2. JSON 데이터 로딩 중...")
    try:
        with open(JSON_2019_PATH, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        df_json = pd.DataFrame(data_json)
        
        # movieId 컬럼 확인
        if 'movieId' not in df_json.columns:
            print("   ❌ JSON 파일에 'movieId' 컬럼이 없습니다. (연결 불가)")
            return
            
        json_ids = set(df_json['movieId'].unique())
        print(f"   - JSON 총 영화 수: {len(json_ids):,}개 (정보 표출 가능)")
    except Exception as e:
        print(f"   ❌ JSON 로드 실패: {e}")
        return

    # ------------------------------------------------
    # 3. 교집합 및 차집합 계산
    # ------------------------------------------------
    print("\n>>> 3. 비교 분석 결과")
    
    # 교집합 (둘 다 있는 것 -> 실제 사용 가능)
    intersection = tag_ids.intersection(json_ids)
    
    # 차집합 A (TagDL엔 있는데 JSON엔 없는 것 -> 제목을 몰라서 못 쓰는 영화)
    only_in_tag = tag_ids - json_ids
    
    # 차집합 B (JSON엔 있는데 TagDL엔 없는 것 -> 태그 정보가 없어서 추천 못 해주는 영화)
    only_in_json = json_ids - tag_ids
    
    print(f"\n✅ [교집합] 함께 사용할 수 있는 영화: {len(intersection):,}개")
    print(f"   (비율: TagDL 기준 {len(intersection)/len(tag_ids)*100:.1f}%)")

    print("-" * 40)
    print(f"⚠️ [누락 1] TagDL에는 있지만 JSON에 없는 영화: {len(only_in_tag):,}개")
    print(f"   -> 이 영화들은 '제목'을 찾을 수 없어 버려집니다.")
    
    print("-" * 40)
    print(f"⚠️ [누락 2] JSON에는 있지만 TagDL에 없는 영화: {len(only_in_json):,}개")
    print(f"   -> 이 영화들은 '태그 점수'가 없어 CBF 추천 후보에서 제외됩니다.")

    # ------------------------------------------------
    # 4. 결론
    # ------------------------------------------------
    print("\n" + "="*60)
    print("FINAL CONCLUSION")
    print("="*60)
    
    if len(intersection) > 5000:
        print("🚀 [성공] 충분한 양의 데이터가 확보되었습니다.")
        print(f"   총 {len(intersection):,}개의 영화로 모델을 학습시킬 수 있습니다.")
        print("   -> archive 폴더 없이 이 두 파일만으로 진행 가능합니다!")
    else:
        print("⚠️ [주의] 교집합 개수가 너무 적습니다. 데이터 확인이 필요합니다.")

if __name__ == "__main__":
    check_overlap()