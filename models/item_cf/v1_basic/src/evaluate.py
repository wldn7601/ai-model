"""
Item-CF 평가 모듈
- 모델 로드
- 평점 예측
- 성능 지표 계산 (RMSE, Precision@K, Recall@K, NDCG@K, Coverage)
"""

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import time
from collections import defaultdict
import config

def load_model(model_path):
    """
    저장된 모델 로드
    
    Args:
        model_path: model.pkl 경로
    
    Returns:
        model_data: dict with keys:
            - user_item_matrix
            - similarity
            - user_map
            - movie_map
            - idx_to_user
            - idx_to_movie
            - n_users
            - n_movies
    """
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✓ Model loaded")
    print(f"  Users:  {model_data['n_users']:,}")
    print(f"  Movies: {model_data['n_movies']:,}")
    print(f"  Similarity shape: {model_data['similarity'].shape}")
    print("="*60 + "\n")
    
    return model_data


def predict_rating(user_idx, movie_idx, user_item_matrix, similarity, top_k=100):
    """
    특정 사용자-영화 쌍의 평점 예측
    
    Args:
        user_idx: 사용자 인덱스
        movie_idx: 영화 인덱스
        user_item_matrix: User-Item 희소 행렬
        similarity: Item-Item 유사도 행렬
        top_k: 사용할 유사 영화 수
    
    Returns:
        predicted_rating: 예측 평점 (1.0 ~ 5.0)

    2차 시도 후 예측 알고리즘 수정
    """
    # 사용자가 평가한 영화들
    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]
    
    if len(rated_indices) == 0:
        return 3.0
    
    # 타겟 영화의 유사도
    sim_scores = similarity[movie_idx][rated_indices]  # ← 간소화!
    user_ratings_for_sim = user_ratings[rated_indices]
    
    # 유사도가 0인 경우 많으므로 필터링
    non_zero_mask = sim_scores > 0
    
    if not np.any(non_zero_mask):
        return 3.0
    
    sim_scores = sim_scores[non_zero_mask]
    user_ratings_for_sim = user_ratings_for_sim[non_zero_mask]
    
    # Top-K
    if top_k and len(sim_scores) > top_k:
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        sim_scores = sim_scores[top_indices]
        user_ratings_for_sim = user_ratings_for_sim[top_indices]
    
    if len(sim_scores) == 0 or np.sum(np.abs(sim_scores)) == 0:
        return 3.0
    
    # 가중 평균
    predicted = np.sum(sim_scores * user_ratings_for_sim) / np.sum(np.abs(sim_scores))
    predicted = np.clip(predicted, 1.0, 5.0)
    
    return predicted


def compute_rmse(test_df, model_data, sample_size=10000):
    """
    RMSE (Root Mean Squared Error) 계산
    
    Args:
        test_df: 테스트 DataFrame
        model_data: 로드된 모델
        sample_size: 샘플 크기 (전체는 시간 오래 걸림)
    
    Returns:
        rmse: RMSE 값
    """
    print("\n" + "="*60)
    print(f"Computing RMSE (sample={sample_size:,})")
    print("="*60)
    
    start_time = time.time()
    
    # 샘플링
    if len(test_df) > sample_size:
        test_sample = test_df.sample(n=sample_size, random_state=42)
    else:
        test_sample = test_df
    
    user_map = model_data['user_map']
    movie_map = model_data['movie_map']
    user_item_matrix = model_data['user_item_matrix']
    similarity = model_data['similarity']
    
    predictions = []
    actuals = []
    
    print(f"Predicting {len(test_sample):,} ratings...")
    
    for idx, row in test_sample.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(test_sample)} ({idx/len(test_sample)*100:.1f}%)")
        
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        # ID → 인덱스 변환
        if user_id not in user_map or movie_id not in movie_map:
            continue
        
        user_idx = user_map[user_id]
        movie_idx = movie_map[movie_id]
        
        # 예측
        pred_rating = predict_rating(
            user_idx, 
            movie_idx, 
            user_item_matrix, 
            similarity,
            top_k=100
        )
        
        predictions.append(pred_rating)
        actuals.append(actual_rating)
    
    # RMSE 계산
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ RMSE: {rmse:.4f}")
    print(f"  Predictions: {len(predictions):,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print("="*60 + "\n")
    
    return rmse


# evaluate.py

def recommend_top_k(user_idx, user_item_matrix, similarity, k=10, pop_penalty=None):
    """
    사용자에게 Top-K 영화 추천 (인기 편향 강제 보정 적용)
    
    Args:
        pop_penalty: 인기 패널티 강도 (0.5 ~ 0.8 추천)
                     - 0.0: 패널티 없음 (Coverage 낮음, RMSE 좋음)
                     - 1.0: 강력한 패널티 (Coverage 높음, 인기 영화 거의 안 나옴)
    """
    # config 값을 기본값으로 사용 (호출 시 값을 안 넣으면 config 값 사용)
    if pop_penalty is None:
        pop_penalty = config.INFERENCE_POP_PENALTY

    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]
    
    # 1. 인기도 계산 (Log 스케일 복귀)
    item_popularity = np.array(user_item_matrix.getnnz(axis=0))
    # log1p를 쓰면 0~10000 -> 0~9.2 범위로 변환되어 값이 튀지 않음
    pop_norm = np.log1p(item_popularity)
    
    # Cold Start
    if len(rated_indices) == 0:
        # 인기도 기반 랜덤 추천
        top_indices = np.argsort(item_popularity)[::-1][:k*3]
        return np.random.choice(top_indices, k, replace=False)
    
    n_movies = user_item_matrix.shape[1]
    scores = np.zeros(n_movies)
    
    # 2. 점수 합산
    for rated_idx in rated_indices:
        similar_scores = similarity[rated_idx].copy()
        user_rating = user_ratings[rated_idx]
        
        # 가중치 (이전과 동일)
        if user_rating >= 4.0: weight = 2.0
        elif user_rating >= 3.0: weight = 1.0
        else: weight = 0.5 # 0.1은 너무 낮아서 0.5로 상향 조정
            
        scores += similar_scores * weight
    
    # 3. [핵심] 패널티 적용
    # 인기 영화(log값 7~8)는 (7~8 ^ 0.5) ≈ 2.6~2.8배 감점
    # 비인기 영화(log값 2~3)는 (2~3 ^ 0.5) ≈ 1.4~1.7배 감점
    # -> 인기 영화가 너무 심하게 깎이지 않도록 방어
    if pop_penalty > 0:
        scores = scores / (np.power(pop_norm, pop_penalty) + 1e-6)
    
    # 이미 본 영화 제외
    scores[rated_indices] = -np.inf
    
    top_indices = np.argsort(scores)[::-1][:k]
    
    return top_indices

def compute_precision_recall_at_k(test_df, model_data, k=10, threshold=4.0, sample_users=1000):
    """
    Precision@K, Recall@K 계산
    
    Args:
        test_df: 테스트 DataFrame
        model_data: 로드된 모델
        k: Top-K
        threshold: 관련 아이템 기준 (평점 >= threshold)
        sample_users: 평가할 사용자 샘플 수
    
    Returns:
        precision: Precision@K
        recall: Recall@K
    """
    print("\n" + "="*60)
    print(f"Computing Precision@{k} and Recall@{k}")
    print("="*60)
    
    start_time = time.time()
    
    user_map = model_data['user_map']
    movie_map = model_data['movie_map']
    user_item_matrix = model_data['user_item_matrix']
    similarity = model_data['similarity']
    
    # 사용자별로 그룹화
    test_grouped = test_df.groupby('userId')
    
    # 샘플 사용자
    all_users = list(test_grouped.groups.keys())
    if len(all_users) > sample_users:
        sampled_users = np.random.choice(all_users, sample_users, replace=False)
    else:
        sampled_users = all_users
    
    precisions = []
    recalls = []
    
    print(f"Evaluating {len(sampled_users):,} users...")
    
    for idx, user_id in enumerate(sampled_users):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(sampled_users)} ({idx/len(sampled_users)*100:.1f}%)")
        
        if user_id not in user_map:
            continue
        
        user_idx = user_map[user_id]
        
        # Ground truth: 테스트셋에서 관련 아이템
        user_test = test_grouped.get_group(user_id)
        relevant_items = set(
            user_test[user_test['rating'] >= threshold]['movieId'].values
        )
        
        # movieId → idx 변환
        relevant_indices = set()
        for mid in relevant_items:
            if mid in movie_map:
                relevant_indices.add(movie_map[mid])
        
        if len(relevant_indices) == 0:
            continue
        
        # Top-K 추천
        recommended_indices = recommend_top_k(
            user_idx, 
            user_item_matrix, 
            similarity, 
            k=k
        )
        recommended_indices = set(recommended_indices)
        
        # Precision & Recall
        hits = len(recommended_indices & relevant_indices)
        
        precision = hits / k if k > 0 else 0
        recall = hits / len(relevant_indices) if len(relevant_indices) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Precision@{k}: {avg_precision:.4f}")
    print(f"✓ Recall@{k}: {avg_recall:.4f}")
    print(f"  Users evaluated: {len(precisions):,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print("="*60 + "\n")
    
    return avg_precision, avg_recall


def compute_ndcg_at_k(test_df, model_data, k=10, sample_users=1000):
    """
    NDCG@K (Normalized Discounted Cumulative Gain) 계산
    
    Args:
        test_df: 테스트 DataFrame
        model_data: 로드된 모델
        k: Top-K
        sample_users: 평가할 사용자 샘플 수
    
    Returns:
        ndcg: NDCG@K
    """
    print("\n" + "="*60)
    print(f"Computing NDCG@{k}")
    print("="*60)
    
    start_time = time.time()
    
    user_map = model_data['user_map']
    movie_map = model_data['movie_map']
    user_item_matrix = model_data['user_item_matrix']
    similarity = model_data['similarity']
    
    test_grouped = test_df.groupby('userId')
    
    all_users = list(test_grouped.groups.keys())
    if len(all_users) > sample_users:
        sampled_users = np.random.choice(all_users, sample_users, replace=False)
    else:
        sampled_users = all_users
    
    ndcgs = []
    
    print(f"Evaluating {len(sampled_users):,} users...")
    
    for idx, user_id in enumerate(sampled_users):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(sampled_users)} ({idx/len(sampled_users)*100:.1f}%)")
        
        if user_id not in user_map:
            continue
        
        user_idx = user_map[user_id]
        
        # Ground truth
        user_test = test_grouped.get_group(user_id)
        
        # movieId → rating 매핑
        true_ratings = {}
        for _, row in user_test.iterrows():
            mid = row['movieId']
            if mid in movie_map:
                true_ratings[movie_map[mid]] = row['rating']
        
        if len(true_ratings) == 0:
            continue
        
        # Top-K 추천
        recommended_indices = recommend_top_k(
            user_idx, 
            user_item_matrix, 
            similarity, 
            k=k
        )
        
        # DCG 계산
        dcg = 0.0
        for i, movie_idx in enumerate(recommended_indices):
            if movie_idx in true_ratings:
                relevance = true_ratings[movie_idx]
                dcg += relevance / np.log2(i + 2)  # i+2 because index starts at 0
        
        # IDCG 계산 (이상적인 순서)
        ideal_ratings = sorted(true_ratings.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_ratings))
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ NDCG@{k}: {avg_ndcg:.4f}")
    print(f"  Users evaluated: {len(ndcgs):,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print("="*60 + "\n")
    
    return avg_ndcg


def compute_coverage(model_data, test_df, k=10, sample_users=1000):
    """
    Coverage 계산 (추천 다양성)
    전체 아이템 중 추천된 아이템 비율
    
    Args:
        model_data: 로드된 모델
        test_df: 테스트 DataFrame
        k: Top-K
        sample_users: 평가할 사용자 수
    
    Returns:
        coverage: Coverage 비율
    """
    print("\n" + "="*60)
    print(f"Computing Coverage (K={k})")
    print("="*60)
    
    start_time = time.time()
    
    user_map = model_data['user_map']
    user_item_matrix = model_data['user_item_matrix']
    similarity = model_data['similarity']
    n_movies = model_data['n_movies']
    
    # 샘플 사용자
    test_users = test_df['userId'].unique()
    if len(test_users) > sample_users:
        sampled_users = np.random.choice(test_users, sample_users, replace=False)
    else:
        sampled_users = test_users
    
    recommended_items = set()
    
    print(f"Generating recommendations for {len(sampled_users):,} users...")
    
    for idx, user_id in enumerate(sampled_users):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(sampled_users)} ({idx/len(sampled_users)*100:.1f}%)")
        
        if user_id not in user_map:
            continue
        
        user_idx = user_map[user_id]
        
        # Top-K 추천
        top_k_items = recommend_top_k(
            user_idx, 
            user_item_matrix, 
            similarity, 
            k=k
        )
        
        recommended_items.update(top_k_items)
    
    coverage = len(recommended_items) / n_movies
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Coverage: {coverage:.4f}")
    print(f"  Recommended items: {len(recommended_items):,} / {n_movies:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print("="*60 + "\n")
    
    return coverage


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트 코드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing evaluate.py")
    print("="*60)
    
    # 간단한 테스트 데이터
    print("\nCreating sample data...")
    
    # 샘플 모델 데이터
    from scipy.sparse import csr_matrix
    
    sample_matrix = csr_matrix([
        [5, 0, 3, 0],
        [4, 0, 0, 2],
        [0, 5, 4, 0],
        [0, 3, 0, 4]
    ], dtype=np.float32)
    
    sample_similarity = np.array([
        [1.0, 0.2, 0.8, 0.1],
        [0.2, 1.0, 0.3, 0.9],
        [0.8, 0.3, 1.0, 0.2],
        [0.1, 0.9, 0.2, 1.0]
    ], dtype=np.float32)
    
    model_data = {
        'user_item_matrix': sample_matrix,
        'similarity': sample_similarity,
        'user_map': {1: 0, 2: 1, 3: 2, 4: 3},
        'movie_map': {10: 0, 20: 1, 30: 2, 40: 3},
        'idx_to_user': {0: 1, 1: 2, 2: 3, 3: 4},
        'idx_to_movie': {0: 10, 1: 20, 2: 30, 3: 40},
        'n_users': 4,
        'n_movies': 4
    }
    
    # 예측 테스트
    print("\nTesting predict_rating()...")
    pred = predict_rating(0, 1, sample_matrix, sample_similarity)
    print(f"  Predicted rating for user 0, movie 1: {pred:.2f}")
    
    # 추천 테스트
    print("\nTesting recommend_top_k()...")
    recs = recommend_top_k(0, sample_matrix, sample_similarity, k=2)
    print(f"  Top-2 recommendations for user 0: {recs}")
    
    print("\n" + "="*60)
    print("✓ evaluate.py works!")
    print("="*60 + "\n")