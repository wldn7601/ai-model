# src/data_preprocessing.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ratings_data(csv_path: Path) -> pd.DataFrame:
    """
    CSV 파일에서 ratings 데이터 로드
    
    Args:
        csv_path: CSV 파일 경로
        
    Returns:
        DataFrame with userId, tmdbId, rating columns
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 컬럼명 확인 - tmdbId 사용
    if 'tmdbId' in df.columns:
        # tmdbId를 movieId로 rename (코드 일관성 유지)
        df = df.rename(columns={'tmdbId': 'movieId'})
    elif 'movieId' not in df.columns:
        raise ValueError(f"Required column 'tmdbId' or 'movieId' not found in {csv_path}")
    
    required_cols = ['userId', 'movieId', 'rating']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")
    
    logger.info(f"Loaded {len(df):,} ratings")
    logger.info(f"Unique users: {df['userId'].nunique():,}")
    logger.info(f"Unique movies (TMDB IDs): {df['movieId'].nunique():,}")
    
    return df[required_cols]


def create_id_mappings(
    df: pd.DataFrame
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    User/Movie ID를 연속된 정수 인덱스로 매핑
    
    Args:
        df: DataFrame with userId and movieId columns
        
    Returns:
        user_to_idx: {원본 userId: 내부 index}
        idx_to_user: {내부 index: 원본 userId}
        movie_to_idx: {원본 movieId(tmdbId): 내부 index}
        idx_to_movie: {내부 index: 원본 movieId(tmdbId)}
    """
    logger.info("Creating ID mappings")
    
    # User 매핑
    unique_users = sorted(df['userId'].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    
    # Movie 매핑 (TMDB ID)
    unique_movies = sorted(df['movieId'].unique())
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}
    
    logger.info(f"Created mappings: {len(user_to_idx):,} users, {len(movie_to_idx):,} movies")
    
    return user_to_idx, idx_to_user, movie_to_idx, idx_to_movie


def convert_to_implicit_confidence(
    df: pd.DataFrame, 
    alpha: float = 20.0,
    min_rating: float = 3.0
) -> pd.DataFrame:
    """
    Explicit ratings를 Implicit confidence로 변환
    
    Args:
        df: DataFrame with 'rating' column
        alpha: Confidence scaling parameter
        min_rating: 이 값 이상만 positive feedback
        
    Returns:
        DataFrame with 'confidence' column
    """
    logger.info(f"Converting to implicit feedback (alpha={alpha}, threshold={min_rating})")
    
    original_len = len(df)
    
    # min_rating 이상만 유지
    df = df[df['rating'] >= min_rating].copy()
    logger.info(f"Kept ratings >= {min_rating}: {len(df):,}/{original_len:,} ({len(df)/original_len*100:.1f}%)")
    
    # Confidence 계산
    df['confidence'] = 1.0 + alpha * df['rating']
    
    logger.info(f"Confidence range: [{df['confidence'].min():.2f}, {df['confidence'].max():.2f}]")
    
    return df


def create_sparse_matrix(
    df: pd.DataFrame,
    user_to_idx: Dict[int, int],
    movie_to_idx: Dict[int, int],
    value_column: str = 'confidence'
) -> csr_matrix:
    """
    User-Item sparse matrix 생성
    
    Args:
        df: DataFrame with userId, movieId, and value columns
        user_to_idx: User ID to index mapping
        movie_to_idx: Movie ID (TMDB ID) to index mapping
        value_column: 값으로 사용할 컬럼
        
    Returns:
        Sparse matrix of shape (n_users, n_movies)
    """
    logger.info(f"Creating sparse matrix with {value_column} values")
    
    # 내부 인덱스로 변환
    user_indices = df['userId'].map(user_to_idx)
    movie_indices = df['movieId'].map(movie_to_idx)
    values = df[value_column].values
    
    # NaN 체크 (매핑 실패 케이스)
    valid_mask = ~(user_indices.isna() | movie_indices.isna())
    
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.warning(f"Dropping {n_invalid:,} interactions (unmapped IDs)")
        user_indices = user_indices[valid_mask]
        movie_indices = movie_indices[valid_mask]
        values = values[valid_mask]
    
    user_indices = user_indices.astype(int)
    movie_indices = movie_indices.astype(int)
    
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)
    
    # CSR matrix 생성
    matrix = csr_matrix(
        (values, (user_indices, movie_indices)),
        shape=(n_users, n_movies),
        dtype=np.float32
    )
    
    logger.info(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz:,}")
    sparsity = 100 * (1 - matrix.nnz / (n_users * n_movies))
    logger.info(f"Sparsity: {sparsity:.4f}%")
    
    return matrix


def save_mappings(
    user_to_idx: Dict[int, int],
    idx_to_user: Dict[int, int],
    movie_to_idx: Dict[int, int],
    idx_to_movie: Dict[int, int],
    user_mapping_path: Path,
    movie_mapping_path: Path
):
    """ID 매핑 딕셔너리들을 pickle로 저장"""
    logger.info("Saving ID mappings")
    
    user_mappings = {
        'to_idx': user_to_idx,
        'to_id': idx_to_user
    }
    
    movie_mappings = {
        'to_idx': movie_to_idx,
        'to_id': idx_to_movie
    }
    
    with open(user_mapping_path, 'wb') as f:
        pickle.dump(user_mappings, f)
    
    with open(movie_mapping_path, 'wb') as f:
        pickle.dump(movie_mappings, f)
    
    logger.info(f"Saved mappings")


def load_mappings(
    user_mapping_path: Path,
    movie_mapping_path: Path
) -> Tuple[Dict, Dict, Dict, Dict]:
    """저장된 ID 매핑 로드"""
    with open(user_mapping_path, 'rb') as f:
        user_mappings = pickle.load(f)
    
    with open(movie_mapping_path, 'rb') as f:
        movie_mappings = pickle.load(f)
    
    return (
        user_mappings['to_idx'],
        user_mappings['to_id'],
        movie_mappings['to_idx'],
        movie_mappings['to_id']
    )


def save_sparse_matrix(matrix: csr_matrix, path: Path):
    """Sparse matrix를 npz 파일로 저장"""
    logger.info(f"Saving sparse matrix to {path}")
    save_npz(path, matrix)


def preprocess_pipeline(
    train_csv_path: Path,
    test_csv_path: Path,
    user_mapping_path: Path,
    movie_mapping_path: Path,
    train_matrix_path: Path,
    test_matrix_path: Path,
    alpha: float = 20.0,
    min_rating: float = 3.0
):
    """
    전체 전처리 파이프라인 실행
    """
    logger.info("=" * 80)
    logger.info("Starting ALS preprocessing pipeline")
    logger.info("=" * 80)
    
    # 1. Train 데이터 로드
    train_df = load_ratings_data(train_csv_path)
    
    # 2. Implicit confidence 변환
    train_df = convert_to_implicit_confidence(train_df, alpha, min_rating)
    
    # 3. ID 매핑 생성 (train 데이터 기준)
    user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = create_id_mappings(train_df)
    
    # 4. 매핑 저장
    save_mappings(
        user_to_idx, idx_to_user, 
        movie_to_idx, idx_to_movie,
        user_mapping_path, movie_mapping_path
    )
    
    # 5. Train sparse matrix 생성
    train_matrix = create_sparse_matrix(
        train_df, user_to_idx, movie_to_idx, 'confidence'
    )
    save_sparse_matrix(train_matrix, train_matrix_path)
    
    # 6. Test 데이터 처리
    logger.info("\nProcessing test data")
    test_df = load_ratings_data(test_csv_path)
    test_df = convert_to_implicit_confidence(test_df, alpha, min_rating)
    
    # Test 데이터에서 train에 없는 user/movie 제외
    test_df = test_df[
        test_df['userId'].isin(user_to_idx.keys()) & 
        test_df['movieId'].isin(movie_to_idx.keys())
    ].copy()
    logger.info(f"Test data after filtering: {len(test_df):,} ratings")
    
    # 7. Test sparse matrix 생성
    test_matrix = create_sparse_matrix(
        test_df, user_to_idx, movie_to_idx, 'confidence'
    )
    save_sparse_matrix(test_matrix, test_matrix_path)
    
    logger.info("=" * 80)
    logger.info("Preprocessing completed successfully")
    logger.info("=" * 80)
    
    return {
        'n_users': len(user_to_idx),
        'n_movies': len(movie_to_idx),
        'train_ratings': train_matrix.nnz,
        'test_ratings': test_matrix.nnz,
        'train_shape': train_matrix.shape,
        'test_shape': test_matrix.shape
    }