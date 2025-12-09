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

# [수정 1] 모델 복잡도 조정
# 데이터가 희소할 때 factors가 너무 크면 학습이 제대로 안 됩니다. 64로 조정 권장.
ALS_CONFIG = {
    'factors': 100,           # 기존 100
    'regularization': 0.05,  # 0.05 유지
    'iterations': 20,
    'dtype': 'float32',
    'use_gpu': True,
    'calculate_training_loss': False
}

# [수정 2] 데이터 필터링 해제
IMPLICIT_CONFIG = {
    'alpha': 20,              # 15~40 사이 권장. 너무 크면 1.0 평점이 과대평가될 수 있음.
    'min_rating_threshold': 3.0
}

# 평가 설정
EVALUATION_CONFIG = {
    'k_values': [5, 10, 20],
    'n_recommendations': 100
}

# 모델 체크포인트 경로
MODEL_CHECKPOINT_DIR = OUTPUT_DIR / "model_checkpoints"
MODEL_CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_CHECKPOINT_DIR / "als_model.pkl"