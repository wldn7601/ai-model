# src/model.py
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from implicit.als import AlternatingLeastSquares
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ALSRecommender:
    """ALS 기반 추천 시스템"""
    
    def __init__(
        self,
        factors: int = 100,
        regularization: float = 0.01,
        iterations: int = 15,
        use_gpu: bool = True,
        dtype=np.float32
    ):
        """
        Args:
            factors: Latent factor 차원
            regularization: L2 정규화 계수
            iterations: 학습 반복 횟수
            use_gpu: GPU 사용 여부
            dtype: 데이터 타입
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            calculate_training_loss=False,  # 학습 속도 향상
            random_state=42
        )
        self.use_gpu = use_gpu
        self.dtype = dtype
        
        logger.info(f"Initialized ALS model:")
        logger.info(f"  - Factors: {factors}")
        logger.info(f"  - Regularization: {regularization}")
        logger.info(f"  - Iterations: {iterations}")
        logger.info(f"  - Use GPU: {use_gpu}")
    
    def train(self, train_matrix: csr_matrix):
        """
        모델 학습
        
        Args:
            train_matrix: User-Item matrix (n_users, n_items)
                         값은 confidence scores
        """
        logger.info("Starting model training...")
        logger.info(f"Input matrix shape: {train_matrix.shape}")
        logger.info(f"  Users: {train_matrix.shape[0]:,}")
        logger.info(f"  Items: {train_matrix.shape[1]:,}")
        logger.info(f"  Non-zero: {train_matrix.nnz:,}")
        
        start_time = time.time()
        
        # Convert to CSR format with correct dtype
        matrix = train_matrix.tocsr().astype(self.dtype)
        
        # GPU 학습
        if self.use_gpu:
            logger.info("Training on GPU...")
            # ★ fit 메서드는 user_items matrix를 받음 (users × items)
            self.model.fit(matrix, show_progress=True)
        else:
            logger.info("Training on CPU...")
            self.model.fit(matrix, show_progress=True)
        
        # 검증
        logger.info(f"\nLearned latent factors:")
        logger.info(f"  User factors: {self.model.user_factors.shape}")
        logger.info(f"  Item factors: {self.model.item_factors.shape}")
        
        # Shape 검증
        if self.model.user_factors.shape[0] != train_matrix.shape[0]:
            logger.error(f"ERROR: User factors count mismatch!")
            logger.error(f"  Expected: {train_matrix.shape[0]}")
            logger.error(f"  Got: {self.model.user_factors.shape[0]}")
        
        if self.model.item_factors.shape[0] != train_matrix.shape[1]:
            logger.error(f"ERROR: Item factors count mismatch!")
            logger.error(f"  Expected: {train_matrix.shape[1]}")
            logger.error(f"  Got: {self.model.item_factors.shape[0]}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        return elapsed_time
    
    def recommend(
        self,
        user_idx: int,
        user_items: csr_matrix,
        n: int = 10,
        filter_already_liked: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        사용자에게 영화 추천
        
        Args:
            user_idx: 사용자 내부 인덱스
            user_items: User-Item matrix
            n: 추천할 아이템 개수
            filter_already_liked: 이미 본 영화 제외 여부
            
        Returns:
            movie_indices: 추천 영화 인덱스 배열
            scores: 추천 점수 배열
        """
        movie_indices, scores = self.model.recommend(
            userid=user_idx,
            user_items=user_items[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked
        )
        
        return movie_indices, scores
    
    def recommend_batch(
        self,
        user_indices: np.ndarray,
        user_items: csr_matrix,
        n: int = 10,
        filter_already_liked: bool = True
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        여러 사용자에게 배치 추천
        
        Args:
            user_indices: 사용자 인덱스 배열
            user_items: User-Item matrix
            n: 추천할 아이템 개수
            filter_already_liked: 이미 본 영화 제외 여부
            
        Returns:
            {user_idx: (movie_indices, scores)} 딕셔너리
        """
        recommendations = {}
        
        for user_idx in user_indices:
            movie_indices, scores = self.recommend(
                user_idx, user_items, n, filter_already_liked
            )
            recommendations[user_idx] = (movie_indices, scores)
        
        return recommendations
    
    def save_model(self, path: Path):
        """모델 저장"""
        logger.info(f"Saving model to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'user_factors': self.model.user_factors,
            'item_factors': self.model.item_factors,
            'config': {
                'factors': self.model.factors,
                'regularization': self.model.regularization,
                'iterations': self.model.iterations
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully")
    
    def load_model(self, path: Path):
        """모델 로드"""
        logger.info(f"Loading model from {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model.user_factors = model_data['user_factors']
        self.model.item_factors = model_data['item_factors']
        
        logger.info("Model loaded successfully")
        logger.info(f"  - User factors shape: {self.model.user_factors.shape}")
        logger.info(f"  - Item factors shape: {self.model.item_factors.shape}")


def load_train_matrix(path: Path) -> csr_matrix:
    """학습 데이터 로드"""
    logger.info(f"Loading train matrix from {path}")
    matrix = load_npz(path)
    logger.info(f"Loaded matrix: shape={matrix.shape}, nnz={matrix.nnz:,}")
    return matrix