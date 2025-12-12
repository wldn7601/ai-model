import pandas as pd
import os
import sys

"""

/home/ubuntu/ai-model/datas/data 폴더의
final_movies_processed.pkl 파일의 movieId와
tagdl.csv 파일의 item_id가 매핑이 되는지 확인

"""

# 1. 파일 경로 설정
pkl_path = '/home/ubuntu/ai-model/datas/data/final_movies_processed.pkl'
csv_path = '/home/ubuntu/ai-model/datas/data/tagdl.csv'

print(f"Checking Mapping between:\n1. {pkl_path}\n2. {csv_path}\n")

# 2. 파일 존재 확인
if not os.path.exists(pkl_path) or not os.path.exists(csv_path):
    print("Error: 파일 경로를 확인해주세요.")
    sys.exit(1)

try:
    # 3. 데이터 로드
    print(">>> 데이터 로딩 중...")
    df_movie = pd.read_pickle(pkl_path)
    df_tags = pd.read_csv(csv_path)

    # 4. CSV 컬럼 확인 (태그 컬럼명 파악용)
    print("\n" + "="*50)
    print("1. Tag 파일(CSV) 컬럼 정보")
    print("="*50)
    print(df_tags.columns.tolist())
    print(df_tags.head(3))

    # 5. ID 컬럼 데이터 타입 비교 (매핑 실패 주원인 점검)
    print("\n" + "="*50)
    print("2. ID 데이터 타입 점검")
    print("="*50)
    id_col_pkl = 'movieId'
    id_col_csv = 'item_id'

    print(f"PKL ({id_col_pkl}): {df_movie[id_col_pkl].dtype}")
    print(f"CSV ({id_col_csv}): {df_tags[id_col_csv].dtype}")

    # 타입 통일 (둘 다 int로 변환 시도)
    # 매핑을 위해 타입을 강제로 맞춥니다.
    try:
        df_movie[id_col_pkl] = df_movie[id_col_pkl].astype(int)
        df_tags[id_col_csv] = df_tags[id_col_csv].astype(int)
        print(">>> 데이터 타입 일치 확인 (Integer)")
    except Exception as e:
        print(f">>> [Warning] 타입 변환 중 에러 발생: {e}")

    # 6. 매핑 분석 (Intersection)
    print("\n" + "="*50)
    print("3. 매핑 통계 (Intersection Analysis)")
    print("="*50)
    
    unique_movies = set(df_movie[id_col_pkl])
    unique_tags_movies = set(df_tags[id_col_csv])
    
    # 교집합 (매칭되는 영화 ID 개수)
    matched_ids = unique_movies.intersection(unique_tags_movies)
    
    print(f"전체 영화 수 (PKL)      : {len(unique_movies):,} 개")
    print(f"태그가 있는 영화 수 (CSV): {len(unique_tags_movies):,} 개 (Unique ID 기준)")
    print(f"매칭된 영화 수          : {len(matched_ids):,} 개")
    print(f"매칭 비율 (Coverage)    : {len(matched_ids) / len(unique_movies) * 100:.2f}%")

    # 7. 실제 매칭 데이터 샘플 확인
    print("\n" + "="*50)
    print("4. 매칭 결과 샘플 (Merge Result)")
    print("="*50)
    
    # Inner Join으로 매칭된 데이터만 병합
    merged_df = pd.merge(df_movie, df_tags, left_on=id_col_pkl, right_on=id_col_csv, how='inner')
    
    if merged_df.empty:
        print(">>> 매칭된 데이터가 없습니다. ID 값들이 전혀 다르거나 타입이 맞지 않습니다.")
    else:
        # 주요 컬럼만 출력 (제목, ID, 태그 관련 컬럼)
        # CSV의 태그 컬럼명을 모르므로 'tag' 혹은 'score' 등이 포함된 컬럼을 추정하여 출력
        display_cols = ['title_ko', id_col_pkl, id_col_csv]
        
        # CSV의 나머지 컬럼 중 2개 정도만 추가해서 보여줌
        tag_cols = [c for c in df_tags.columns if c != id_col_csv][:2] 
        display_cols.extend(tag_cols)
        
        print(merged_df[display_cols].head(5))

    # 8. 매칭되지 않은 영화 샘플
    print("\n" + "="*50)
    print("5. 태그가 없는 영화 샘플 (Unmatched)")
    print("="*50)
    unmatched_ids = unique_movies - unique_tags_movies
    if unmatched_ids:
        unmatched_sample = df_movie[df_movie[id_col_pkl].isin(list(unmatched_ids)[:3])]
        print(unmatched_sample[['movieId', 'title_ko']])
    else:
        print("모든 영화에 태그가 존재합니다.")

except Exception as e:
    print(f"\nError 발생: {e}")