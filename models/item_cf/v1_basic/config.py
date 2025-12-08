"""
Item-CF v1_basic Configuration
모든 설정값을 여기서 관리
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. 경로 설정 (자동)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 현재 파일 (config.py) 위치 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # v1_basic/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))  # 프로젝트 루트


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 원본 데이터 (절대 경로)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "datas/archive/ratings.csv")
MOVIES_PATH = os.path.join(PROJECT_ROOT, "datas/archive/movies.csv")

# 출력 디렉토리 (절대 경로)
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 데이터 전처리 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 필터링 기준
MIN_USER_RATINGS = 20          # 사용자당 최소 평점 수
MIN_MOVIE_RATINGS = 30         # 영화당 최소 평점 수
MAX_ITERATIONS = 10            # 반복 필터링 최대 횟수

# Train/Test 분할
TEST_SIZE = 0.2                # 테스트 데이터 비율 (20%)
RANDOM_STATE = 42              # 재현성을 위한 랜덤 시드


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 모델 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 유사도 계산
SIMILARITY_METHOD = "cosine"   # 유사도 방법: "cosine" or "pearson"
TOP_K_SIMILAR = 1500            # 각 영화당 저장할 유사 영화 수

# 메모리 최적화
USE_SPARSE_MATRIX = True       # 희소 행렬 사용 여부

SHRINKAGE_PARAM = 0
IDF_PARAM = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 평가 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Ranking 평가
EVAL_K = 10                    # Precision@K, Recall@K의 K값

# RMSE 평가
RMSE_SAMPLE_SIZE = 10000       # RMSE 계산용 샘플 크기 (전체는 오래 걸림)

INFERENCE_POP_PENALTY = 0.5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 실행 옵션 (선택)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 샘플링 모드 (빠른 테스트용)
USE_SAMPLE = False             # True면 샘플 데이터만 사용
SAMPLE_SIZE = 200000           # 샘플 크기 (USE_SAMPLE=True일 때만)

# 로깅
VERBOSE = True                 # 진행 상황 출력 여부
LOG_INTERVAL = 1000            # 진행률 출력 간격


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 시스템 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# GPU 사용 (집 환경: RTX 4060ti)
USE_GPU = True                 # GPU 사용 활성화
GPU_DEVICE = 0                 # GPU 디바이스 번호 (보통 0)

# 멀티프로세싱
N_JOBS = -1                    # CPU 코어 수 (-1: 모든 코어)

# 배치 크기 (GPU 사용 시)
BATCH_SIZE = 5000              # 한 번에 처리할 아이템 수 (GPU 메모리에 맞게)