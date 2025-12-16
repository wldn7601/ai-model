import pandas as pd
import json
import os

"""
final_movies_processed.pkl 파일에 overview 추가 전처리:
1. insert_movies.csv에서 overview 로드
2. overview가 없는(빈 문자열/null) 행만 제거
3. final_movies_processed.pkl과 병합
4. 기존 컬럼 유지하면서 overview만 업데이트
"""

# 파일 경로
insert_csv = '/home/ubuntu/ai-model/datas/data/insert_movies.csv'
input_pkl = '/home/ubuntu/ai-model/datas/data/final_movies_processed.pkl'
output_base = '/home/ubuntu/ai-model/models/cbf/v2/data/pre_final_movies_processed'

print("="*60)
print("1. insert_movies.csv 로드 및 overview 확인")
print("="*60)

# CSV 로드
df_insert = pd.read_csv(insert_csv)
print(f"전체 영화 수: {len(df_insert):,}")

# overview 컬럼 존재 확인
if 'overview' not in df_insert.columns:
    raise ValueError("❌ overview 컬럼이 없습니다.")

# overview 상태 확인
print("\noverview 상태:")
print(f"  - 전체: {len(df_insert):,}")
print(f"  - null: {df_insert['overview'].isnull().sum():,}")
print(f"  - 빈 문자열: {(df_insert['overview'] == '').sum():,}")

# overview가 유효한 행만 선택 (null이 아니고 빈 문자열이 아닌 것)
df_insert_valid = df_insert[
    df_insert['overview'].notna() & 
    (df_insert['overview'].str.strip() != '')
].copy()

print(f"\noverview 유효한 영화: {len(df_insert_valid):,}")
print(f"제거된 영화: {len(df_insert) - len(df_insert_valid):,}")

# 병합을 위한 키 확인
print("\n병합 키 확인:")
if 'movieId' in df_insert_valid.columns:
    merge_key = 'movieId'
elif 'tmdb_id' in df_insert_valid.columns:
    merge_key = 'tmdb_id'
else:
    raise ValueError("❌ movieId 또는 tmdb_id 컬럼이 없습니다.")
print(f"병합 키: {merge_key}")

# 필요한 컬럼만 선택
df_overview = df_insert_valid[[merge_key, 'overview']].copy()

print("="*60)
print("2. final_movies_processed.pkl 로드")
print("="*60)

df = pd.read_pickle(input_pkl)
print(f"전체 행 수: {len(df):,}")
print(f"컬럼: {list(df.columns)}")

# 기존 overview 상태 확인
if 'overview' in df.columns:
    print(f"\n기존 overview 있음: {df['overview'].notna().sum():,}")
    print(f"기존 overview 없음: {df['overview'].isna().sum():,}")

print("="*60)
print("3. overview 병합")
print("="*60)

# 기존 overview 컬럼 제거 (있다면)
if 'overview' in df.columns:
    df = df.drop(columns=['overview'])
    print("기존 overview 컬럼 제거")

# overview 병합
df_merged = df.merge(df_overview, on=merge_key, how='inner')

print(f"\n병합 후 영화 수: {len(df_merged):,}")
print(f"제거된 영화 수: {len(df) - len(df_merged):,}")

# overview 검증
print("\noverview 검증:")
print(f"  - null 개수: {df_merged['overview'].isnull().sum()}")
print(f"  - 평균 길이: {df_merged['overview'].str.len().mean():.1f}자")
print(f"  - 최소 길이: {df_merged['overview'].str.len().min()}자")
print(f"  - 최대 길이: {df_merged['overview'].str.len().max()}자")

print("="*60)
print("4. 결측치 제거 및 정리")
print("="*60)

# 결측치 확인
print("결측치 현황:")
print(df_merged.isnull().sum())

# 결측치 제거
df_cleaned = df_merged.dropna()
print(f"\n결측치 제거 후: {len(df_cleaned):,} 행")

# text_input, embedding 컬럼 삭제 (있다면)
columns_to_drop = [col for col in ['text_input', 'embedding'] if col in df_cleaned.columns]
if columns_to_drop:
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    print(f"컬럼 삭제: {columns_to_drop}")

print(f"\n최종 컬럼: {list(df_cleaned.columns)}")
print(f"컬럼 수: {len(df_cleaned.columns)}")

print("="*60)
print("5. 샘플 데이터 확인")
print("="*60)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

print(df_cleaned[['movieId', 'title_ko', 'overview']].head(3))

print("="*60)
print("6. 파일 저장")
print("="*60)

# PKL 저장
pkl_path = f"{output_base}.pkl"
df_cleaned.to_pickle(pkl_path)
print(f"✅ PKL 저장: {pkl_path}")

# CSV 저장 (확인용)
csv_path = f"{output_base}.csv"
df_for_csv = df_cleaned.copy()

# dict/list 컬럼을 JSON 문자열로 변환
for col in ['tag_genome', 'ott_providers', 'genres']:
    if col in df_for_csv.columns:
        df_for_csv[col] = df_for_csv[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
        )

df_for_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ CSV 저장: {csv_path}")

print(f"\n최종 데이터: {len(df_cleaned):,} 행 x {len(df_cleaned.columns)} 컬럼")

# 파일 크기
pkl_size = os.path.getsize(pkl_path) / (1024**2)
csv_size = os.path.getsize(csv_path) / (1024**2)

print(f"\nPKL 파일 크기: {pkl_size:.2f} MB")
print(f"CSV 파일 크기: {csv_size:.2f} MB")

print("="*60)
print("✅ 전처리 완료")
print("="*60)