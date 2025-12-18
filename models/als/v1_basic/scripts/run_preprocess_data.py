# scripts/run_preprocess.py
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import time

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import preprocess_pipeline
import config


def split_data():
    """
    데이터 분할 (80:20)
    """
    print("\n" + "="*80)
    print("Step 1: Data Splitting (80:20 Train/Test)")
    print("="*80 + "\n")
    
    # 원본 데이터 로드
    print(f"Loading data from: {config.SOURCE_DATA_PATH}")
    
    if not config.SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {config.SOURCE_DATA_PATH}")
    
    df = pd.read_csv(config.SOURCE_DATA_PATH)
    
    print(f"\nOriginal data:")
    print(f"  Total ratings: {len(df):,}")
    print(f"  Unique users: {df['userId'].nunique():,}")
    print(f"  Unique movies (TMDB IDs): {df['tmdbId'].nunique():,}")
    
    # 평점 분포
    print(f"\nRating distribution:")
    rating_dist = df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {rating}: {count:,} ({percentage:.1f}%)")
    
    # Threshold별 데이터 비율 미리보기
    print(f"\nData retention by threshold:")
    for threshold in [0, 2.0, 3.0, 3.5, 4.0]:
        kept = (df['rating'] >= threshold).sum()
        percentage = (kept / len(df)) * 100
        print(f"  >= {threshold}: {kept:,} ({percentage:.1f}%)")
    
    # Stratify 여부 결정
    user_counts = df['userId'].value_counts()
    can_stratify = user_counts.min() >= config.SPLIT_CONFIG['min_ratings_per_user']
    
    if can_stratify:
        print(f"\n✓ Using stratified split")
        stratify_column = df['userId']
    else:
        print(f"\n✗ Using random split")
        stratify_column = None
    
    # Train/Test 분할
    print(f"\nSplitting (test_size={config.SPLIT_CONFIG['test_size']})...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=config.SPLIT_CONFIG['test_size'],
        random_state=config.SPLIT_CONFIG['random_state'],
        stratify=stratify_column
    )
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_df):,} ratings ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} ratings ({len(test_df)/len(df)*100:.1f}%)")
    
    # 저장
    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)
    
    print(f"\n✓ Saved:")
    print(f"  - {config.TRAIN_CSV}")
    print(f"  - {config.TEST_CSV}")
    
    return len(train_df), len(test_df)


def run_preprocessing():
    """
    전처리 실행
    """
    print("\n" + "="*80)
    print("Step 2: Preprocessing (Threshold >= 3.0)")
    print("="*80 + "\n")
    
    print(f"Configuration:")
    print(f"  Alpha: {config.IMPLICIT_CONFIG['alpha']}")
    print(f"  Min rating threshold: {config.IMPLICIT_CONFIG['min_rating_threshold']}")
    print(f"  (Only ratings >= {config.IMPLICIT_CONFIG['min_rating_threshold']} will be used)")
    
    stats = preprocess_pipeline(
        train_csv_path=config.TRAIN_CSV,
        test_csv_path=config.TEST_CSV,
        user_mapping_path=config.USER_MAPPING_PATH,
        movie_mapping_path=config.MOVIE_MAPPING_PATH,
        train_matrix_path=config.TRAIN_MATRIX_PATH,
        test_matrix_path=config.TEST_MATRIX_PATH,
        alpha=config.IMPLICIT_CONFIG['alpha'],
        min_rating=config.IMPLICIT_CONFIG['min_rating_threshold']
    )
    
    return stats


def main():
    print("\n" + "="*80)
    print("ALS Data Preparation Pipeline")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: 데이터 분할
    train_size, test_size = split_data()
    
    # Step 2: 전처리
    stats = run_preprocessing()
    
    elapsed_time = time.time() - start_time
    
    # 최종 요약
    print("\n" + "="*80)
    print("Pipeline Summary")
    print("="*80)
    
    print(f"\n[Data Splitting]")
    print(f"  Train: {train_size:,} ratings")
    print(f"  Test: {test_size:,} ratings")
    
    print(f"\n[After Filtering (>= 3.0)]")
    print(f"  Users: {stats['n_users']:,}")
    print(f"  Movies (TMDB IDs): {stats['n_movies']:,}")
    print(f"  Train ratings: {stats['train_ratings']:,}")
    print(f"  Test ratings: {stats['test_ratings']:,}")
    
    print(f"\n[Matrix Shapes]")
    print(f"  Train: {stats['train_shape']}")
    print(f"  Test: {stats['test_shape']}")
    
    print(f"\n[Files Generated]")
    print(f"  ✓ {config.TRAIN_CSV}")
    print(f"  ✓ {config.TEST_CSV}")
    print(f"  ✓ {config.USER_MAPPING_PATH}")
    print(f"  ✓ {config.MOVIE_MAPPING_PATH}")
    print(f"  ✓ {config.TRAIN_MATRIX_PATH}")
    print(f"  ✓ {config.TEST_MATRIX_PATH}")
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    print("\n" + "="*80)
    print("✓ Ready for model training!")
    print("="*80 + "\n")
    
    print("Next step:")
    print("  python scripts/run_train.py")


if __name__ == "__main__":
    main()