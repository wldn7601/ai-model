import pandas as pd
import os

"""
데이터 전처리 스크립트
대상: extracted_movies.csv

작업 내용:
1. 데이터 로드
2. Overview가 없는(Null, 빈 문자열) 영화 제거
3. 컬럼명 변경 (tmdb_id -> tmdbId)
4. 중복 tmdbId 제거 (첫 번째만 유지)
5. 결과 저장 (CSV, PKL)
"""

# ============================================================
# 설정
# ============================================================
input_csv = '/home/ubuntu/ai-model/datas/data/extracted_movies.csv'
output_dir = '/home/ubuntu/ai-model/models/cbf/v2/data'
output_base = os.path.join(output_dir, 'pre_final_movies_processed')

# 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("1. 데이터 로드")
print("="*60)

if not os.path.exists(input_csv):
    print(f"❌ 오류: 입력 파일이 없습니다.\n경로: {input_csv}")
    exit()

# CSV 로드
df = pd.read_csv(input_csv)
print(f"전체 원본 영화 수: {len(df):,}")
print(f"원본 컬럼 목록: {list(df.columns)}")

# ============================================================
# 2. Overview 전처리
# ============================================================
print("\n" + "="*60)
print("2. Overview 필터링")
print("="*60)

# 유효한 데이터만 남기기 (Overview가 Null이 아니고 빈 문자열도 아닌 경우)
df_clean = df[
    df['overview'].notna() & 
    (df['overview'].astype(str).str.strip() != '')
].copy()

removed_count = len(df) - len(df_clean)

print(f"✅ 유효한 영화 수: {len(df_clean):,}")
print(f"❌ 제거된 영화 수 (Overview 없음): {removed_count:,}")

# ============================================================
# 3. 데이터 정리 및 컬럼명 변경
# ============================================================
print("\n" + "="*60)
print("3. 데이터 정리 (tmdbId 변환 + 중복 제거)")
print("="*60)

# 인덱스 재정렬
df_clean = df_clean.reset_index(drop=True)

# 컬럼명 변경: tmdb_id -> tmdbId
if 'tmdb_id' in df_clean.columns:
    df_clean = df_clean.rename(columns={'tmdb_id': 'tmdbId'})
    print("✅ 컬럼명 변경 완료: tmdb_id -> tmdbId")
elif 'tmdbId' not in df_clean.columns:
    print("❌ 오류: tmdb_id 또는 tmdbId 컬럼을 찾을 수 없습니다.")
    exit()
else:
    print("ℹ️ 이미 tmdbId 컬럼이 존재합니다.")

# 중복 제거
before_dedup = len(df_clean)
duplicates = df_clean[df_clean.duplicated(subset=['tmdbId'], keep=False)]

if len(duplicates) > 0:
    print(f"\n⚠️ 중복 tmdbId 발견: {len(duplicates)}개 행")
    print("중복 샘플 (최대 10개):")
    sample_dups = duplicates[['tmdbId', 'title']].drop_duplicates().head(10)
    for _, row in sample_dups.iterrows():
        print(f"   - tmdbId: {row['tmdbId']}, title: {row['title']}")

df_clean = df_clean.drop_duplicates(subset=['tmdbId'], keep='first')
after_dedup = len(df_clean)

print(f"\n✅ 중복 제거 완료: {before_dedup - after_dedup}개 제거")
print(f"✅ 최종 영화 수: {after_dedup:,}")
print(f"최종 컬럼 목록: {list(df_clean.columns)}")

# Overview 길이 통계
ov_lens = df_clean['overview'].astype(str).str.len()
print(f"평균 Overview 길이: {ov_lens.mean():.1f}자")

# ============================================================
# 4. 저장
# ============================================================
print("\n" + "="*60)
print("4. 파일 저장")
print("="*60)

# 1) PKL 저장
pkl_path = f"{output_base}.pkl"
df_clean.to_pickle(pkl_path)
print(f"✅ PKL 저장 완료: {pkl_path}")

# 2) CSV 저장
csv_path = f"{output_base}.csv"
df_clean.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ CSV 저장 완료: {csv_path}")

# ============================================================
# 5. 최종 요약
# ============================================================
print("\n" + "="*60)
print("완료 요약")
print("="*60)
print(f"원본 영화 수:        {len(df):,}")
print(f"Overview 제거:       -{removed_count:,}")
print(f"중복 제거:           -{before_dedup - after_dedup:,}")
print(f"최종 영화 수:        {after_dedup:,}")
print(f"\n출력 파일:")
print(f"  - {pkl_path}")
print(f"  - {csv_path}")