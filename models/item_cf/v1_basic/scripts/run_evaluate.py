"""
평가 실행 스크립트
모델 로드 → 테스트 데이터로 평가 → 결과 저장
"""

import sys
sys.path.append('..')

import pandas as pd
import json
import config
from src.evaluate import (
    load_model,
    compute_rmse,
    compute_precision_recall_at_k,
    compute_ndcg_at_k,
    compute_coverage
)
import time


def main():
    print("\n" + "="*60)
    print("Item-CF v1_basic - Evaluation")
    print("="*60)
    
    total_start = time.time()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 모델 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[1/3] Loading model...")
    model_path = f"{config.MODEL_DIR}/model.pkl"
    model_data = load_model(model_path)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 테스트 데이터 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[2/3] Loading test data...")
    test_path = f"{config.OUTPUT_DIR}/test.csv"
    test = pd.read_csv(test_path)
    
    print(f"✓ Loaded {len(test):,} test ratings")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. 평가
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n[3/3] Evaluating...")
    
    metrics = {}
    
    # RMSE
    rmse = compute_rmse(
        test, 
        model_data, 
        sample_size=config.RMSE_SAMPLE_SIZE
    )
    metrics['rmse'] = float(rmse)
    
    # Precision & Recall
    precision, recall = compute_precision_recall_at_k(
        test, 
        model_data, 
        k=config.EVAL_K,
        # 4 -> 3.5 변경
        threshold=3.5,
        sample_users=1000
    )
    metrics[f'precision@{config.EVAL_K}'] = float(precision)
    metrics[f'recall@{config.EVAL_K}'] = float(recall)
    
    # NDCG
    ndcg = compute_ndcg_at_k(
        test, 
        model_data, 
        k=config.EVAL_K,
        sample_users=1000
    )
    metrics[f'ndcg@{config.EVAL_K}'] = float(ndcg)
    
    # Coverage
    coverage = compute_coverage(
        model_data, 
        test, 
        k=config.EVAL_K,
        sample_users=1000
    )
    metrics['coverage'] = float(coverage)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 결과 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\nSaving results...")
    
    results_path = f"{config.RESULTS_DIR}/metrics.json"
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved to {results_path}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 완료
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("✓ Evaluation Complete!")
    print("="*60)
    print(f"\nMetrics:")
    print(f"  RMSE:           {metrics['rmse']:.4f}")
    print(f"  Precision@{config.EVAL_K}:   {metrics[f'precision@{config.EVAL_K}']:.4f}")
    print(f"  Recall@{config.EVAL_K}:      {metrics[f'recall@{config.EVAL_K}']:.4f}")
    print(f"  NDCG@{config.EVAL_K}:        {metrics[f'ndcg@{config.EVAL_K}']:.4f}")
    print(f"  Coverage:       {metrics['coverage']:.4f}")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"\n")


if __name__ == "__main__":
    main()