import pandas as pd
import json
import os

"""
final_movies_processed.pkl 파일 정리:
1. 결측치가 있는 행 제거
2. text_input 컬럼 삭제
3. embedding 컬럼 삭제
"""

# 파일 경로
input_path = '/home/ubuntu/ai-model/datas/data/final_movies_processed.pkl'
output_base = '/home/ubuntu/ai-model/models/cbf/v2/data/pre_final_movies_processed'

print(f"원본 파일: {input_path}\n")

# 데이터 로드
df = pd.read_pickle(input_path)

print("="*60)
print("1. 원본 데이터 상태")
print("="*60)
print(f"전체 행 수: {len(df):,}")
print(f"전체 컬럼 수: {len(df.columns)}")
print("\n결측치 현황:")
print(df.isnull().sum())

# 결측치 제거
df_cleaned = df.dropna()

print("\n" + "="*60)
print("2. 결측치 제거 후")
print("="*60)
print(f"남은 행 수: {len(df_cleaned):,}")
print(f"제거된 행 수: {len(df) - len(df_cleaned):,}")

# text_input, embedding 컬럼 삭제
columns_to_drop = ['text_input', 'embedding']
df_cleaned = df_cleaned.drop(columns=columns_to_drop)

print("\n" + "="*60)
print("3. 컬럼 삭제 후 최종 상태")
print("="*60)
print(f"남은 컬럼: {list(df_cleaned.columns)}")
print(f"컬럼 수: {len(df_cleaned.columns)}")

# 최종 데이터 타입 확인
print("\n" + "="*60)
print("4. 최종 데이터 타입")
print("="*60)
print(df_cleaned.dtypes)

# 샘플 데이터 확인
print("\n" + "="*60)
print("5. 정리된 데이터 샘플 (상위 3개)")
print("="*60)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df_cleaned.head(3))

# PKL 저장 (프로그램용)
pkl_path = f"{output_base}.pkl"
df_cleaned.to_pickle(pkl_path)

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

print("\n" + "="*60)
print("저장 완료")
print("="*60)
print(f"PKL 파일: {pkl_path}")
print(f"CSV 파일: {csv_path}")
print(f"최종 데이터: {len(df_cleaned):,} 행 x {len(df_cleaned.columns)} 컬럼")

# 파일 크기 비교
original_size = os.path.getsize(input_path) / (1024**2)
pkl_size = os.path.getsize(pkl_path) / (1024**2)
csv_size = os.path.getsize(csv_path) / (1024**2)

print(f"\n원본 파일 크기: {original_size:.2f} MB")
print(f"PKL 파일 크기: {pkl_size:.2f} MB")
print(f"CSV 파일 크기: {csv_size:.2f} MB")
print(f"PKL 감소량: {original_size - pkl_size:.2f} MB ({(1 - pkl_size/original_size)*100:.1f}%)")