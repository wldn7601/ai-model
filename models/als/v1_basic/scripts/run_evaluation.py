# scripts/run_evaluation.py
import sys
from pathlib import Path
import json
import numpy as np

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import ALSRecommender, load_train_matrix
from src.evaluation import RecommenderEvaluator, load_test_matrix
import config


def main():
    print("\n" + "="*80)
    print("ALS Model Comprehensive Evaluation")
    print("="*80 + "\n")
    
    # 1. 데이터 로드
    print("Loading data...")
    train_matrix = load_train_matrix(config.TRAIN_MATRIX_PATH)
    test_matrix = load_test_matrix(config.TEST_MATRIX_PATH)
    
    # 2. 모델 로드
    print("\nLoading trained model...")
    model = ALSRecommender(
        factors=config.ALS_CONFIG['factors'],
        regularization=config.ALS_CONFIG['regularization'],
        iterations=config.ALS_CONFIG['iterations'],
        use_gpu=config.ALS_CONFIG['use_gpu']
    )
    model.load_model(config.MODEL_PATH)
    
    # 3. 통합 평가 실행
    print("\n")
    evaluator = RecommenderEvaluator(k_values=config.EVALUATION_CONFIG['k_values'])
    
    metrics = evaluator.evaluate(
        model=model,
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        n_recommendations=config.EVALUATION_CONFIG['n_recommendations']
    )
    
    # 4. 결과 출력
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    # Ranking 지표
    print("\n### Ranking Metrics ###")
    for k in config.EVALUATION_CONFIG['k_values']:
        print(f"\n--- Top-{k} ---")
        print(f"Precision@{k}:  {metrics[f'Precision@{k}']:.4f}")
        print(f"Recall@{k}:     {metrics[f'Recall@{k}']:.4f}")
        print(f"MAP@{k}:        {metrics[f'MAP@{k}']:.4f}")
        print(f"NDCG@{k}:       {metrics[f'NDCG@{k}']:.4f}")
    
    print(f"\n--- Overall ---")
    print(f"Coverage:       {metrics['Coverage']:.4f} ({metrics['Coverage']*100:.2f}%)")
    
    # Loss 지표
    print("\n### Loss Metrics ###")
    print(f"Test RMSE:      {metrics['Test_RMSE']:.4f}")
    print(f"Test MSE:       {metrics['Test_MSE']:.4f}")
    
    # RMSE를 원본 rating 스케일로 변환 (참고용)
    alpha = config.IMPLICIT_CONFIG['alpha']
    rmse_rating_scale = metrics['Test_RMSE'] / alpha
    print(f"\nRMSE (rating scale): {rmse_rating_scale:.4f} (≈ {rmse_rating_scale:.2f} points)")
    
    print("="*80 + "\n")
    
    # 5. 성능 해석
    print("Performance Interpretation:")
    print("-" * 80)
    
    # Precision@10 해석
    prec_10 = metrics['Precision@10']
    if prec_10 >= 0.20:
        prec_status = "Excellent ⭐⭐⭐"
    elif prec_10 >= 0.15:
        prec_status = "Good ⭐⭐"
    elif prec_10 >= 0.10:
        prec_status = "Fair ⭐"
    else:
        prec_status = "Needs Improvement"
    print(f"Precision@10: {prec_status}")
    
    # NDCG@10 해석
    ndcg_10 = metrics['NDCG@10']
    if ndcg_10 >= 0.25:
        ndcg_status = "Excellent ⭐⭐⭐"
    elif ndcg_10 >= 0.20:
        ndcg_status = "Good ⭐⭐"
    elif ndcg_10 >= 0.15:
        ndcg_status = "Fair ⭐"
    else:
        ndcg_status = "Needs Improvement"
    print(f"NDCG@10:      {ndcg_status}")
    
    # Coverage 해석
    coverage = metrics['Coverage']
    if coverage >= 0.40:
        cov_status = "Excellent ⭐⭐⭐ (High diversity)"
    elif coverage >= 0.30:
        cov_status = "Good ⭐⭐ (Moderate diversity)"
    elif coverage >= 0.20:
        cov_status = "Fair ⭐ (Limited diversity)"
    else:
        cov_status = "Needs Improvement (Filter bubble risk)"
    print(f"Coverage:     {cov_status}")
    
    # RMSE 해석
    if rmse_rating_scale < 0.5:
        rmse_status = "Excellent ⭐⭐⭐ (<0.5 rating error)"
    elif rmse_rating_scale < 1.0:
        rmse_status = "Good ⭐⭐ (<1.0 rating error)"
    elif rmse_rating_scale < 2.0:
        rmse_status = "Fair ⭐ (<2.0 rating error)"
    else:
        rmse_status = "Needs Improvement (>2.0 rating error)"
    print(f"RMSE:         {rmse_status}")
    
    print("="*80 + "\n")
    
    # scripts/run_evaluation.py

    # 6. 결과 저장
    results_path = config.RESULTS_DIR / "evaluation_results.json"
    
    # NumPy 타입을 Python 기본 타입으로 변환
    def convert_to_python_type(obj):
        """NumPy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_python_type(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_type(item) for item in obj]
        else:
            return obj
    
    # 저장용 metrics (해석 정보 추가)
    save_metrics = {
        **metrics,
        'rmse_rating_scale': rmse_rating_scale,
        'interpretation': {
            'precision@10': prec_status,
            'ndcg@10': ndcg_status,
            'coverage': cov_status,
            'rmse': rmse_status
        }
    }
    
    # NumPy 타입 변환
    save_metrics = convert_to_python_type(save_metrics)
    
    with open(results_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()