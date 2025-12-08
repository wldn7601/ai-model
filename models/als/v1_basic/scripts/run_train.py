# scripts/run_train.py
import sys
from pathlib import Path
import time

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
    print("Loading training data...")
    train_matrix = load_train_matrix(config.TRAIN_MATRIX_PATH)
    
    # 2. 모델 초기화
    print("\nInitializing ALS model...")
    model = ALSRecommender(
        factors=config.ALS_CONFIG['factors'],
        regularization=config.ALS_CONFIG['regularization'],
        iterations=config.ALS_CONFIG['iterations'],
        use_gpu=config.ALS_CONFIG['use_gpu'],
        dtype=np.float32
    )
    
    # 3. 모델 학습
    print("\n" + "="*80)
    training_time = model.train(train_matrix)
    print("="*80)
    
    # 4. 모델 저장
    model_save_path = config.MODEL_PATH
    model.save_model(model_save_path)
    
    # 5. 요약 출력
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"Training time:        {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Train matrix shape:   {train_matrix.shape}")
    print(f"Model factors:        {config.ALS_CONFIG['factors']}")
    print(f"Iterations:           {config.ALS_CONFIG['iterations']}")
    print(f"GPU used:             {config.ALS_CONFIG['use_gpu']}")
    print(f"\nModel saved to:       {model_save_path}")
    print("="*80 + "\n")
    
    print("Ready for evaluation!")


if __name__ == "__main__":
    # numpy import 추가
    import numpy as np
    main()