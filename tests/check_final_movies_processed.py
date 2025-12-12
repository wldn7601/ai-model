import pandas as pd
import os
import sys

"""

SBERT에서 쓸 데이터 final_movies_processed.pkl
파일의 데이터 형식 확인

"""

# 1. 경로 설정 (제공된 절대 경로 사용)
file_path = '/home/ubuntu/ai-model/datas/data/final_movies_processed.pkl'

print(f"Checking file at: {file_path}")

# 2. 파일 존재 여부 확인
if not os.path.exists(file_path):
    print("Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit(1)

try:
    # 3. 데이터 로드
    df = pd.read_pickle(file_path)

    # 4. 기본 정보 출력
    print("\n" + "="*40)
    print("1. 데이터 기본 정보 (Info)")
    print("="*40)
    # 데이터 타입, Non-Null Count 확인
    print(df.info()) 

    print("\n" + "="*40)
    print("2. 결측치 확인 (Null Count)")
    print("="*40)
    print(df.isnull().sum())

    # 5. 데이터 샘플 확인 (텍스트 내용 확인용)
    print("\n" + "="*40)
    print("3. 상위 3개 행 샘플 (Text Columns 확인)")
    print("="*40)
    
    # 텍스트가 길어 생략되는 것을 방지
    pd.set_option('display.max_colwidth', None) 
    pd.set_option('display.max_columns', None)
    
    print(df.head(3))

    # 6. 컬럼별 데이터 타입 정밀 검사 (SBERT 입력 전처리용)
    print("\n" + "="*40)
    print("4. 컬럼별 실제 데이터 타입 (첫번째 행 기준)")
    print("="*40)
    for col in df.columns:
        first_val = df[col].iloc[0] if len(df) > 0 else None
        print(f"Column: {col:20} | Type: {type(first_val)} | Example: {str(first_val)[:50]}...")

    # =========================================================
    # [추가된 로직] 상위 3개 데이터의 모든 값을 상세 출력
    # =========================================================
    print("\n" + "="*40)
    print("5. 상위 3개 행 모든 값 상세 출력 (All Values Detail)")
    print("="*40)

    # 상위 3개 행만 반복
    for i in range(min(3, len(df))):
        print(f"\n>>> [Row Index: {i}] 데이터 상세 시작 " + "-"*20)
        row = df.iloc[i]
        
        # 해당 행의 모든 컬럼을 순회하며 출력 (Key: Value 형태)
        for col_name, value in row.items():
            print(f"[{col_name}]")
            print(f"{value}")
            print("-" * 10) # 가독성을 위한 구분선
        
        print(f">>> [Row Index: {i}] 데이터 끝\n" + "="*40)

except Exception as e:
    print(f"\nError 발생: {e}")