"""
전처리 실행 스크립트
전체 데이터 또는 샘플 데이터 전처리
"""

import sys
sys.path.append('..')  # src import용

import pandas as pd
import config
from src.preprocess import filter_data, split_data, save_data, save_statistics


def main():
    print("\n" + "="*60)
    print("Item-CF v1_basic - Data Preprocessing")
    print("="*60)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 데이터 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[1/4] Loading data...")
    print(f"Path: {config.RAW_DATA_PATH}")
    
    if config.USE_SAMPLE:
        print(f"Sample mode: Loading first {config.SAMPLE_SIZE:,} rows")
        ratings = pd.read_csv(config.RAW_DATA_PATH, nrows=config.SAMPLE_SIZE)
    else:
        print("Full mode: Loading all data")
        ratings = pd.read_csv(config.RAW_DATA_PATH)
    
    original_size = len(ratings)
    print(f"✓ Loaded {original_size:,} ratings")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 필터링
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[2/4] Filtering...")
    filtered = filter_data(
        ratings,
        config.MIN_USER_RATINGS,
        config.MIN_MOVIE_RATINGS,
        config.MAX_ITERATIONS
    )
    
    retention = len(filtered) / original_size * 100
    print(f"Retention: {retention:.1f}%")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Train/Test 분할
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[3/4] Splitting...")
    train, test = split_data(
        filtered,
        config.TEST_SIZE,
        config.RANDOM_STATE
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[4/4] Saving...")
    save_data(train, test, config.OUTPUT_DIR)
    
    save_statistics(
        original_size,
        filtered,
        train,
        test,
        config.MIN_USER_RATINGS,
        config.MIN_MOVIE_RATINGS,
        config.OUTPUT_DIR
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 완료
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n" + "="*60)
    print("✓ Preprocessing Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {config.OUTPUT_DIR}/train.csv")
    print(f"  - {config.OUTPUT_DIR}/test.csv")
    print(f"  - {config.OUTPUT_DIR}/stats.json")
    print("\n")


if __name__ == "__main__":
    main()