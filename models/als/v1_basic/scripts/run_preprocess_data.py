# scripts/02_preprocess_data.py
import sys
from pathlib import Path
import time

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import preprocess_pipeline
import config


def main():
    print("\n" + "="*80)
    print("ALS Data Preprocessing")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # 전처리 실행
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
    
    elapsed_time = time.time() - start_time
    
    # 결과 출력
    print("\n" + "="*80)
    print("Preprocessing Summary")
    print("="*80)
    print(f"Number of users:        {stats['n_users']:,}")
    print(f"Number of movies:       {stats['n_movies']:,}")
    print(f"Train ratings:          {stats['train_ratings']:,}")
    print(f"Test ratings:           {stats['test_ratings']:,}")
    print(f"Train matrix shape:     {stats['train_shape']}")
    print(f"Test matrix shape:      {stats['test_shape']}")
    print(f"\nTotal processing time:  {elapsed_time:.2f} seconds")
    print("="*80 + "\n")
    
    print("Generated files:")
    print(f"  - {config.USER_MAPPING_PATH}")
    print(f"  - {config.MOVIE_MAPPING_PATH}")
    print(f"  - {config.TRAIN_MATRIX_PATH}")
    print(f"  - {config.TEST_MATRIX_PATH}")
    print("\nReady for model training!")


if __name__ == "__main__":
    main()