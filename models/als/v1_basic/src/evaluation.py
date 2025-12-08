# src/evaluation.py
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """추천 시스템 평가 클래스 (Ranking + Loss 지표)"""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Args:
            k_values: 평가할 K 값들 (Precision@K, Recall@K 등)
        """
        self.k_values = k_values
    
    def evaluate(
        self,
        model,
        train_matrix: csr_matrix,
        test_matrix: csr_matrix,
        n_recommendations: int = 100
    ) -> Dict:
        """
        전체 평가 수행 (Ranking 지표 + Loss 지표)
        
        Args:
            model: 학습된 ALSRecommender 모델
            train_matrix: 학습 데이터
            test_matrix: 테스트 데이터
            n_recommendations: 추천 생성할 개수
            
        Returns:
            평가 결과 딕셔너리
        """
        logger.info("="*80)
        logger.info("Starting comprehensive evaluation (Ranking + Loss metrics)")
        logger.info("="*80)
        
        # 1. Ranking 지표 계산
        ranking_metrics = self._evaluate_ranking(
            model, train_matrix, test_matrix, n_recommendations
        )
        
        # 2. Loss 지표 계산
        loss_metrics = self._evaluate_loss(model, train_matrix, test_matrix)
        
        # 3. 통합
        all_metrics = {**ranking_metrics, **loss_metrics}
        
        logger.info("Evaluation completed")
        
        return all_metrics
    
    def _evaluate_ranking(
        self,
        model,
        train_matrix: csr_matrix,
        test_matrix: csr_matrix,
        n_recommendations: int
    ) -> Dict:
        """Ranking 지표 계산"""
        logger.info("\n[1/2] Evaluating Ranking Metrics (Precision, Recall, MAP, NDCG, Coverage)")
        
        n_users = test_matrix.shape[0]
        
        # 평가할 사용자 선택 (test set에 평가 데이터가 있는 사용자만)
        test_users = []
        for user_idx in range(n_users):
            if test_matrix[user_idx].nnz > 0:
                test_users.append(user_idx)
        
        logger.info(f"Evaluating {len(test_users):,} users with test data")
        
        # 각 K에 대한 메트릭 저장
        results = {
            'precision': {k: [] for k in self.k_values},
            'recall': {k: [] for k in self.k_values},
            'map': {k: [] for k in self.k_values},
            'ndcg': {k: [] for k in self.k_values}
        }
        
        all_recommended_items = set()
        
        # 사용자별 평가
        for user_idx in tqdm(test_users, desc="Ranking evaluation"):
            # 추천 생성
            try:
                movie_indices, scores = model.recommend(
                    user_idx=user_idx,
                    user_items=train_matrix,
                    n=n_recommendations,
                    filter_already_liked=True
                )
            except Exception as e:
                logger.warning(f"Failed to recommend for user {user_idx}: {e}")
                continue
            
            # Test set에서 실제로 좋아한 영화
            test_items = test_matrix[user_idx].indices
            
            if len(test_items) == 0:
                continue
            
            # 추천된 영화 기록 (Coverage 계산용)
            all_recommended_items.update(movie_indices)
            
            # 각 K에 대해 메트릭 계산
            for k in self.k_values:
                top_k = movie_indices[:k]
                
                # Precision@K, Recall@K
                precision, recall = self._precision_recall_at_k(
                    top_k, test_items, k
                )
                results['precision'][k].append(precision)
                results['recall'][k].append(recall)
                
                # MAP@K
                ap = self._average_precision_at_k(top_k, test_items, k)
                results['map'][k].append(ap)
                
                # NDCG@K
                ndcg = self._ndcg_at_k(top_k, test_items, k)
                results['ndcg'][k].append(ndcg)
        
        # Coverage 계산 부분
        # 평균 계산
        metrics = {}
        for k in self.k_values:
            metrics[f'Precision@{k}'] = np.mean(results['precision'][k])
            metrics[f'Recall@{k}'] = np.mean(results['recall'][k])
            metrics[f'MAP@{k}'] = np.mean(results['map'][k])
            metrics[f'NDCG@{k}'] = np.mean(results['ndcg'][k])

        # Coverage 계산 수정
        n_items = train_matrix.shape[1]
        n_unique_recommended = len(all_recommended_items)

        # 디버깅 로그
        logger.info(f"Coverage calculation:")
        logger.info(f"  Total items in catalog: {n_items:,}")
        logger.info(f"  Unique items recommended: {n_unique_recommended:,}")

        # Coverage는 0~1 사이여야 함
        if n_unique_recommended > n_items:
            logger.error(f"ERROR: Recommended items ({n_unique_recommended}) > Total items ({n_items})")
            logger.error(f"This indicates a bug in recommendation or counting.")
            coverage = 1.0  # 임시로 1.0 설정
        else:
            coverage = n_unique_recommended / n_items

        metrics['Coverage'] = coverage
        
        return metrics
    
    def _evaluate_loss(
        self,
        model,
        train_matrix: csr_matrix,
        test_matrix: csr_matrix
    ) -> Dict:
        """Loss 지표 계산"""
        logger.info("\n[2/2] Evaluating Loss Metrics (RMSE, MSE)")
        
        # Test RMSE/MSE
        test_rmse = self._calculate_rmse(model, test_matrix)
        test_mse = self._calculate_mse(model, test_matrix)
        
        # Train RMSE/MSE (선택적 - 시간이 오래 걸릴 수 있음)
        # train_rmse = self._calculate_rmse(model, train_matrix)
        # train_mse = self._calculate_mse(model, train_matrix)
        
        metrics = {
            'Test_RMSE': test_rmse,
            'Test_MSE': test_mse,
            # 'Train_RMSE': train_rmse,
            # 'Train_MSE': train_mse
        }
        
        return metrics
    
    def _precision_recall_at_k(
        self,
        recommended: np.ndarray,
        relevant: np.ndarray,
        k: int
    ) -> Tuple[float, float]:
        """Precision@K와 Recall@K 계산"""
        recommended_set = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_set & relevant_set)
        
        precision = hits / k if k > 0 else 0.0
        recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
        
        return precision, recall
    
    def _average_precision_at_k(
        self,
        recommended: np.ndarray,
        relevant: np.ndarray,
        k: int
    ) -> float:
        """Average Precision@K 계산"""
        relevant_set = set(relevant)
        
        if len(relevant_set) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i
        
        return score / min(len(relevant_set), k)
    
    def _ndcg_at_k(
        self,
        recommended: np.ndarray,
        relevant: np.ndarray,
        k: int
    ) -> float:
        """NDCG@K 계산"""
        relevant_set = set(relevant)
        
        if len(relevant_set) == 0:
            return 0.0
        
        # DCG 계산
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG 계산
        idcg = 0.0
        for i in range(min(len(relevant_set), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0

    # src/evaluation.py

    def _calculate_rmse(
        self,
        model,
        matrix: csr_matrix
    ) -> float:
        """RMSE (Root Mean Squared Error) 계산 - 벡터화 버전"""
        rows, cols = matrix.nonzero()
        actual_values = np.array(matrix[rows, cols]).flatten()
        
        # User factors와 Item factors 가져오기
        user_factors = model.model.user_factors
        item_factors = model.model.item_factors
        
        n_users, n_factors = user_factors.shape
        n_items = item_factors.shape[0]
        
        # 유효한 인덱스만 필터링
        valid_mask = (rows < n_users) & (cols < n_items)
        
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"  Found {n_invalid:,} ratings with invalid indices. Filtering them out.")
            rows = rows[valid_mask]
            cols = cols[valid_mask]
            actual_values = actual_values[valid_mask]
        
        logger.info(f"  Calculating predictions for {len(rows):,} ratings...")
        
        # 벡터화된 예측 계산
        predicted_values = np.sum(
            user_factors[rows] * item_factors[cols], 
            axis=1
        )
        
        # RMSE 계산
        mse = np.mean((predicted_values - actual_values) ** 2)
        rmse = np.sqrt(mse)
        
        logger.info(f"  RMSE: {rmse:.4f}")
        
        return rmse
    
    def _calculate_mse(
        self,
        model,
        matrix: csr_matrix
    ) -> float:
        """MSE (Mean Squared Error) 계산 - 벡터화 버전"""
        rows, cols = matrix.nonzero()
        actual_values = np.array(matrix[rows, cols]).flatten()
        
        # User factors와 Item factors 가져오기
        user_factors = model.model.user_factors
        item_factors = model.model.item_factors
        
        n_users, n_factors = user_factors.shape
        n_items = item_factors.shape[0]
        
        # 유효한 인덱스만 필터링
        valid_mask = (rows < n_users) & (cols < n_items)
        
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"  Found {n_invalid:,} ratings with invalid indices. Filtering them out.")
            rows = rows[valid_mask]
            cols = cols[valid_mask]
            actual_values = actual_values[valid_mask]
        
        logger.info(f"  Calculating predictions for {len(rows):,} ratings...")
        
        # 벡터화된 예측 계산
        predicted_values = np.sum(
            user_factors[rows] * item_factors[cols], 
            axis=1
        )
        
        # MSE 계산
        mse = np.mean((predicted_values - actual_values) ** 2)
        
        logger.info(f"  MSE: {mse:.4f}")
        
        return mse


def load_test_matrix(path):
    """Test matrix 로드"""
    from scipy.sparse import load_npz
    logger.info(f"Loading test matrix from {path}")
    matrix = load_npz(path)
    logger.info(f"Loaded test matrix: shape={matrix.shape}, nnz={matrix.nnz:,}")
    return matrix