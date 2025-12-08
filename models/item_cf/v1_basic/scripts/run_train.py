"""
학습 실행 스크립트
User-Item 행렬 생성 → Item 유사도 계산 → 모델 저장
"""

import sys
sys.path.append('..')

import pandas as pd
import config
from src.train import (
    build_user_item_matrix,
    compute_item_similarity_cpu,
    compute_item_similarity_gpu,
    save_model
)
import time


def main():
    print("\n" + "="*60)
    print("Item-CF v1_basic - Training")
    print("="*60)
    
    total_start = time.time()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 데이터 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[1/3] Loading train data...")
    train_path = f"{config.OUTPUT_DIR}/train.csv"
    
    print(f"Path: {train_path}")
    train = pd.read_csv(train_path)
    
    print(f"✓ Loaded {len(train):,} ratings")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. User-Item 행렬 생성
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[2/3] Building matrix...")
    matrix, user_map, movie_map, idx_to_user, idx_to_movie = build_user_item_matrix(train)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Item 유사도 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[3/3] Computing similarity...")
    
    if config.USE_GPU:
        print(f"Mode: GPU (Shrinkage={config.SHRINKAGE_PARAM}, IDF={config.IDF_PARAM})")
        try:
            similarity = compute_item_similarity_gpu(
                matrix,
                top_k=config.TOP_K_SIMILAR,
                shrinkage=config.SHRINKAGE_PARAM,
                apply_idf=config.IDF_PARAM
            )
        except Exception as e:
            print(f"GPU failed: {e}")
            print("Falling back to CPU...")
            similarity = compute_item_similarity_cpu(
                matrix,
                top_k=config.TOP_K_SIMILAR,
                shrinkage=config.SHRINKAGE_PARAM,
                apply_idf=config.IDF_PARAM
            )
    else:
        print(f"Mode: CPU (Shrinkage={config.SHRINKAGE_PARAM}, IDF={config.IDF_PARAM})")
        similarity = compute_item_similarity_cpu(
            matrix,
            top_k=config.TOP_K_SIMILAR,
            shrinkage=config.SHRINKAGE_PARAM,
            apply_idf=config.IDF_PARAM
        )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 모델 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\nSaving model...")
    save_model(
        matrix,
        similarity,
        user_map,
        movie_map,
        idx_to_user,
        idx_to_movie,
        config.MODEL_DIR
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 완료
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  - {config.MODEL_DIR}/item_similarity.npy")
    print(f"  - {config.MODEL_DIR}/model.pkl")
    print("\n")


if __name__ == "__main__":
    main()