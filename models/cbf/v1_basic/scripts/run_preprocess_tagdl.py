import pandas as pd
import json
import os
import time

# ==========================================
# 1. 경로 설정
# ==========================================
SOURCE_ROOT = '/home/ubuntu/ai-model/datas'
# 입력 파일
PATH_TAGDL = os.path.join(SOURCE_ROOT, '성능보완용', 'tagdl.csv')
PATH_JSON = os.path.join(SOURCE_ROOT, '2019_data.json')

# 출력 파일 (v1_basic/data 폴더)
TARGET_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic/data'
OUTPUT_JSON = os.path.join(TARGET_DIR, '2019_data_with_tags.json')

def inject_tags():
    print("🚀 [JSON Tag Injection] 데이터 병합 시작...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    start_time = time.time()

    # ------------------------------------------------
    # 1. TagDL 데이터 로드 및 최적화
    # ------------------------------------------------
    print(">>> 1. TagDL(AI 점수) 로드 및 그룹화 중...")
    
    df_tag = pd.read_csv(PATH_TAGDL)
    if 'item_id' in df_tag.columns:
        df_tag.rename(columns={'item_id': 'movieId'}, inplace=True)

    # 노이즈 제거 (점수가 너무 낮은 태그는 용량 낭비이므로 제외)
    # 0.05 미만은 의미 없는 수준
    df_tag = df_tag[df_tag['score'] >= 0.05]

    # 영화별로 태그 묶기 (GroupBy) -> Dictionary 변환
    # 구조: { movieId: [{'tag': 'funny', 'score': 0.9}, ...] }
    # 이 과정이 속도를 크게 좌우함
    
    print("   - 태그 그룹화(Grouping) 수행 중...")
    
    # 1) 각 행을 dict로 변환
    # (apply 대신 list comprehension 사용이 더 빠를 수 있음)
    tag_dict = {}
    
    # 데이터프레임을 순회하며 딕셔너리에 추가 (속도 최적화)
    # movieId 기준으로 정렬되어 있다고 가정하면 더 빠르지만, 안전하게 전체 순회
    # pandas groupby는 대용량에서 느릴 수 있어 for loop + dict 사용
    
    # 영화별 그룹핑을 위한 임시 딕셔너리
    grouped = df_tag.groupby('movieId')
    
    for mid, group in grouped:
        # 상위 50개 태그만 저장 (용량 최적화)
        top_tags = group.nlargest(50, 'score')[['tag', 'score']]
        tag_list = top_tags.to_dict('records') # [{'tag':.., 'score':..}, ...]
        tag_dict[mid] = tag_list

    print(f"   - 태그 딕셔너리 생성 완료 ({len(tag_dict):,}개 영화)")

    # ------------------------------------------------
    # 2. JSON 데이터 로드 및 주입
    # ------------------------------------------------
    print(">>> 2. 2019_data.json 로드 및 태그 주입...")
    
    with open(PATH_JSON, 'r', encoding='utf-8') as f:
        movie_list = json.load(f)
    
    print(f"   - 원본 영화 수: {len(movie_list):,}개")

    final_list = []
    matched_count = 0

    for movie in movie_list:
        # movieId 확인
        mid = movie.get('movieId')
        if mid is None:
            continue
            
        # 태그가 있는지 확인
        if mid in tag_dict:
            # 태그 정보 주입!
            movie['predicted_tags'] = tag_dict[mid]
            final_list.append(movie)
            matched_count += 1
    
    # ------------------------------------------------
    # 3. 저장
    # ------------------------------------------------
    print(f">>> 3. 결과 저장 중... ({len(final_list):,}개 영화)")
    print(f"   - 위치: {OUTPUT_JSON}")
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"\n🎉 완료! (소요 시간: {end_time - start_time:.2f}초)")
    print(f"   - 장르, OTT 정보 등이 그대로 유지되었습니다.")
    print(f"   - 이제 'predicted_tags' 필드로 CBF 계산이 가능합니다.")

if __name__ == "__main__":
    inject_tags()