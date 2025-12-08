# config.py
import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

# 디렉토리 생성
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 데이터 파일 경로
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# 전처리 출력 파일
USER_MAPPING_PATH = DATA_DIR / "user_id_mapping.pkl"
MOVIE_MAPPING_PATH = DATA_DIR / "movie_id_mapping.pkl"
TRAIN_MATRIX_PATH = DATA_DIR / "train_matrix.npz"
TEST_MATRIX_PATH = DATA_DIR / "test_matrix.npz"

# ALS Hyperparameters
ALS_CONFIG = {
    'factors': 100,
    'regularization': 0.05,
    'iterations': 15,
    'dtype': 'float32',
    'use_gpu': True,
    'calculate_training_loss': False  # 학습 속도 향상
}

# Implicit Feedback 변환 설정
IMPLICIT_CONFIG = {
    'alpha': 15,
    'min_rating_threshold': 3.0  # ← 변경: 4.0 이상만 positive
}

# 평가 설정
EVALUATION_CONFIG = {
    'k_values': [5, 10, 20],  # Precision@K, Recall@K 계산할 K 값들
    'n_recommendations': 100   # 추천 생성할 개수
}

# 모델 체크포인트 경로
MODEL_CHECKPOINT_DIR = OUTPUT_DIR / "model_checkpoints"
MODEL_CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_CHECKPOINT_DIR / "als_model.pkl"