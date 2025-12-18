# config.py
import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 원본 데이터 경로
SOURCE_DATA_PATH = Path("/home/ubuntu/ai-model/datas/data/ratings_tmdb.csv")

# 분할된 데이터 파일 경로
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# 전처리 출력 파일
USER_MAPPING_PATH = DATA_DIR / "user_id_mapping.pkl"
MOVIE_MAPPING_PATH = DATA_DIR / "movie_id_mapping.pkl"  # TMDB ID 매핑
TRAIN_MATRIX_PATH = DATA_DIR / "train_matrix.npz"
TEST_MATRIX_PATH = DATA_DIR / "test_matrix.npz"

# ALS 모델 설정
ALS_CONFIG = {
    'factors': 100,
    'regularization': 0.05,
    'iterations': 20,
    'dtype': 'float32',
    'use_gpu': True,
    'calculate_training_loss': False
}

# Implicit Feedback 변환 설정
# ALS는 LightGCN과 달리 threshold 필요!
# 이유: "좋아함/모름" 이진 분류 + confidence
IMPLICIT_CONFIG = {
    'alpha': 20,
    'min_rating_threshold': 3.0  # 3.0 이상만 positive (필수!)
}

# 데이터 분할 설정
SPLIT_CONFIG = {
    'test_size': 0.2,  # 80:20 split
    'random_state': 42,
    'min_ratings_per_user': 5  # Stratify 조건
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