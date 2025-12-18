# scripts/run_train.py
import sys
from pathlib import Path
import time
import numpy as np

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import ALSRecommender, load_train_matrix
import config


def main():
    print("\n" + "="*80)
    print("ALS Model Training")
    print("="*80 + "\n")
    
    # 1. 학습 데이터 로드
    print("Step 1: Loading training data...")
    train_matrix = load_train_matrix(config.TRAIN_MATRIX_PATH)
    
    print(f"\nTrain matrix info:")
    print(f"  Shape: {train_matrix.shape}")
    print(f"  Users: {train_matrix.shape[0]:,}")
    print(f"  Movies (TMDB IDs): {train_matrix.shape[1]:,}")
    print(f"  Ratings (>= 3.0): {train_matrix.nnz:,}")
    print(f"  Sparsity: {100 * (1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])):.4f}%")
    
    # 2. 모델 초기화
    print("\nStep 2: Initializing ALS model...")
    print(f"  Factors: {config.ALS_CONFIG['factors']}")
    print(f"  Regularization: {config.ALS_CONFIG['regularization']}")
    print(f"  Iterations: {config.ALS_CONFIG['iterations']}")
    print(f"  GPU: {config.ALS_CONFIG['use_gpu']}")
    
    model = ALSRecommender(
        factors=config.ALS_CONFIG['factors'],
        regularization=config.ALS_CONFIG['regularization'],
        iterations=config.ALS_CONFIG['iterations'],
        use_gpu=config.ALS_CONFIG['use_gpu'],
        dtype=np.float32
    )
    
    # 3. 모델 학습
    print("\n" + "="*80)
    print("Step 3: Training...")
    print("="*80)
    
    training_time = model.train(train_matrix)
    
    # 4. 학습 결과 검증
    print("\n" + "="*80)
    print("Training Results")
    print("="*80)
    
    n_users_expected = train_matrix.shape[0]
    n_items_expected = train_matrix.shape[1]
    n_users_actual = model.model.user_factors.shape[0]
    n_items_actual = model.model.item_factors.shape[0]
    
    print(f"\nExpected shapes:")
    print(f"  Users: {n_users_expected:,}")
    print(f"  Items: {n_items_expected:,}")
    
    print(f"\nActual factor shapes:")
    print(f"  User factors: {model.model.user_factors.shape} ({n_users_actual:,} × {config.ALS_CONFIG['factors']})")
    print(f"  Item factors: {model.model.item_factors.shape} ({n_items_actual:,} × {config.ALS_CONFIG['factors']})")
    
    # Shape 검증
    if n_users_actual == n_users_expected and n_items_actual == n_items_expected:
        print(f"\n✓ Shapes are correct!")
    else:
        print(f"\n✗ WARNING: Shape mismatch detected!")
        if n_users_actual != n_users_expected:
            print(f"  User factors: expected {n_users_expected:,}, got {n_users_actual:,}")
        if n_items_actual != n_items_expected:
            print(f"  Item factors: expected {n_items_expected:,}, got {n_items_actual:,}")
    
    # 5. 모델 저장
    print("\n" + "="*80)
    print("Step 4: Saving model...")
    print("="*80)
    
    model.save_model(config.MODEL_PATH)
    print(f"\n✓ Model saved to: {config.MODEL_PATH}")
    
    # 6. 최종 요약
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"\n[Dataset]")
    print(f"  Users: {train_matrix.shape[0]:,}")
    print(f"  Movies: {train_matrix.shape[1]:,}")
    print(f"  Ratings: {train_matrix.nnz:,}")
    
    print(f"\n[Model]")
    print(f"  Factors: {config.ALS_CONFIG['factors']}")
    print(f"  Regularization: {config.ALS_CONFIG['regularization']}")
    print(f"  Iterations: {config.ALS_CONFIG['iterations']}")
    
    print(f"\n[Training]")
    print(f"  Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"  GPU: {config.ALS_CONFIG['use_gpu']}")
    
    print(f"\n[Output]")
    print(f"  User factors: {model.model.user_factors.shape}")
    print(f"  Item factors: {model.model.item_factors.shape}")
    print(f"  Saved to: {config.MODEL_PATH}")
    
    print("\n" + "="*80)
    print("✓ Training completed successfully!")
    print("="*80 + "\n")
    
    print("Next step:")
    print("  python scripts/run_evaluation.py")


if __name__ == "__main__":
    main()