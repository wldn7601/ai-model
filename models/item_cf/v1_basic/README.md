# Item-CF v1_basic

기본 Item-Based Collaborative Filtering 프로토타입

## 개요

MovieLens 25M 데이터셋을 활용한 아이템 기반 협업 필터링 구현.
Cosine Similarity 기반으로 영화 간 유사도를 계산하고, 사용자의 평점 이력을 바탕으로 새로운 영화를 추천합니다.

## 폴더 구조

```
v1_basic/
├── README.md                     # 이 문서
├── config.py                     # 설정 파일
├── requirements.txt              # 의존성
├── .gitignore                    # Git 제외 파일
├── src/                          # 핵심 로직
│   ├── __init__.py
│   ├── preprocess.py            # 데이터 전처리
│   ├── train.py                 # 모델 학습
│   └── evaluate.py              # 평가
├── scripts/                      # 실행 스크립트
│   ├── run_preprocess.py        # 전처리 실행
│   ├── run_train.py             # 학습 실행
│   └── run_evaluate.py          # 평가 실행
├── notebooks/                    # 실험용 노트북
│   ├── 01_explore.ipynb         # 데이터 탐색
│   ├── 02_prototype.ipynb       # 프로토타입 검증
│   └── 03_results.ipynb         # 결과 분석
├── data/                         # 전처리 데이터 (gitignore)
│   ├── train.csv
│   ├── test.csv
│   └── stats.json
├── outputs/                      # 모델 출력 (gitignore)
│   ├── item_similarity.npy
│   └── model.pkl
└── results/                      # 평가 결과
    └── metrics.json
```

## 설정값

### config.py

```python
# 데이터 전처리
MIN_USER_RATINGS = 20      # 사용자당 최소 평점 수
MIN_MOVIE_RATINGS = 30     # 영화당 최소 평점 수
MAX_ITERATIONS = 10        # 반복 필터링 최대 횟수
TEST_SIZE = 0.2            # 테스트 데이터 비율

# 모델
SIMILARITY_METHOD = "cosine"   # 유사도 방법
TOP_K_SIMILAR = 100            # 각 영화당 저장할 유사 영화 수

# GPU
USE_GPU = True             # GPU 사용 (RTX 4060ti)
BATCH_SIZE = 5000          # 배치 크기

# 평가
EVAL_K = 10                # Precision@K, Recall@K
RMSE_SAMPLE_SIZE = 10000   # RMSE 계산용 샘플
```

## 데이터 전처리 결과

### 입력 데이터

- **데이터셋**: MovieLens 25M
- **원본 평점**: 25,000,095개
- **원본 사용자**: 162,541명
- **원본 영화**: 62,423개

### 필터링 과정

```
Iteration 1: 253,263개 제거 → 24,746,832개
Iteration 2: 280개 제거 → 24,746,552개
Iteration 3: 수렴 (변화 없음)
```

### 최종 결과

| 항목              | 값           |
| ----------------- | ------------ |
| **전체 평점**     | 24,746,552개 |
| **데이터 유지율** | 99.0%        |
| **사용자 수**     | 162,357명    |
| **영화 수**       | 15,906개     |
| **희소도**        | 99.04%       |

### Train/Test 분할

| 항목      | 크기         | 비율 |
| --------- | ------------ | ---- |
| **Train** | 19,797,241개 | 80%  |
| **Test**  | 4,949,311개  | 20%  |

### 통계 요약

```json
{
  "original": {
    "total_ratings": 25000095
  },
  "filtered": {
    "total_ratings": 24746552,
    "total_users": 162357,
    "total_movies": 15906,
    "retention_rate": 0.9899
  },
  "split": {
    "train_size": 19797241,
    "test_size": 4949311
  },
  "config": {
    "min_user_ratings": 20,
    "min_movie_ratings": 30
  }
}
```

## 실행 방법

### 1. 환경 설정

```bash
# 가상환경 활성화
cd movigation-ai
source venv311/bin/activate
pip install -r requirements.txt

# v1_basic 폴더로 이동
cd models/item_cf/v1_basic

# 의존성 설치 (필요시)
pip install -r requirements.txt
```

### 2. 전처리 (완료 ✅)

```bash
python scripts/run_preprocess.py
```

**출력**:

- `data/train.csv` (~630 MB)
- `data/test.csv` (~160 MB)
- `data/stats.json` (~1 KB)

### 3. 학습 (진행 예정)

```bash
python scripts/run_train.py
```

**예상 시간**:

- CPU: 2-4시간
- GPU (RTX 4060ti): 30-60분

**출력**:

- `outputs/item_similarity.npy` (~400 MB, Top-K=100)
- `outputs/model.pkl` (~500 MB)

### 4. 평가 (진행 예정)

```bash
python scripts/run_evaluate.py
```

**예상 시간**: 30-60분  
**출력**:

- `results/metrics.json`

## 알고리즘 설명

### Item-Based Collaborative Filtering

**원리**:

1. 영화 간 유사도 계산 (Cosine Similarity)
2. 사용자가 평가한 영화들과 유사한 영화 찾기
3. 유사도 가중 평균으로 평점 예측

**수식**:

```
예측 평점(u, i) = Σ(sim(i, j) × r(u, j)) / Σ|sim(i, j)|
```

- `u`: 사용자
- `i`: 예측할 영화
- `j`: 사용자가 평가한 영화들
- `sim(i, j)`: 영화 i와 j의 유사도
- `r(u, j)`: 사용자 u의 영화 j 평점

### Cosine Similarity

```
sim(i, j) = (r_i · r_j) / (||r_i|| × ||r_j||)
```

- `r_i`: 영화 i의 평점 벡터
- `r_j`: 영화 j의 평점 벡터

### Top-K 최적화

각 영화당 유사도 상위 100개만 저장:

- **메모리**: 20GB → 400MB (50배 감소)
- **속도**: 2배 향상
- **성능**: 거의 동일

## 성능 목표

| 지표             | 목표   | 설명                             |
| ---------------- | ------ | -------------------------------- |
| **RMSE**         | < 1.0  | 평점 예측 오차                   |
| **Precision@10** | > 0.25 | 상위 10개 중 관련성 비율         |
| **Recall@10**    | > 0.15 | 전체 관련 항목 중 상위 10개 비율 |
| **NDCG@10**      | > 0.40 | 순위 품질                        |
| **Coverage**     | > 0.50 | 추천 가능한 영화 비율            |

## 개발 일지

### 2024-12-06

**완료**:

- ✅ 프로젝트 구조 설계
- ✅ config.py 작성
- ✅ .gitignore 설정
- ✅ src/preprocess.py 구현
- ✅ scripts/run_preprocess.py 구현
- ✅ 전체 데이터 전처리 완료

**진행 중**:

- 🚧 src/train.py 구현

**다음 단계**:

- 📋 학습 실행
- 📋 평가 구현
- 📋 결과 분석

## 기술적 도전

### 해결한 문제

1. **경로 오류**

   - 문제: 상대 경로로 인한 FileNotFoundError
   - 해결: `os.path` 기반 절대 경로 사용

2. **데이터 손실 최소화**
   - 문제: 과도한 필터링으로 데이터 손실
   - 해결: min_user=20, min_movie=30으로 조정 (99% 유지)

### 예정된 도전

1. **GPU 메모리 최적화**

   - 16GB GPU에서 효율적인 배치 처리
   - CuPy 활용한 가속

2. **대용량 유사도 계산**
   - 15,906 × 15,906 행렬 처리
   - Top-K 최적화

## 참고 자료

### 라이브러리

- [scikit-learn](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity)
- [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [CuPy](https://docs.cupy.dev/en/stable/)

## 버전 관리

- **버전**: v1_basic
- **생성일**: 2024-12-06
- **상태**: 전처리 완료, 학습 준비 중
- **다음 버전 계획**: v2_optimized (하이퍼파라미터 튜닝)

