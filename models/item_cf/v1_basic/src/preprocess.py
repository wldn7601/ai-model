"""
데이터 전처리 모듈
- 반복 필터링
- Train/Test 분할
- 데이터 저장
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os


def filter_data(ratings, min_user, min_movie, max_iter=10):
    """
    반복 필터링
    
    Args:
        ratings: DataFrame (userId, movieId, rating, timestamp)
        min_user: 사용자당 최소 평점 수
        min_movie: 영화당 최소 평점 수
        max_iter: 최대 반복 횟수
    
    Returns:
        filtered DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Filtering (min_user={min_user}, min_movie={min_movie})")
    print(f"{'='*60}")
    
    print(f"Original: {len(ratings):,} ratings")
    
    for i in range(max_iter):
        before = len(ratings)
        
        # 영화 필터링
        movie_counts = ratings.groupby('movieId').size()
        valid_movies = movie_counts[movie_counts >= min_movie].index
        ratings = ratings[ratings['movieId'].isin(valid_movies)]
        
        # 사용자 필터링
        user_counts = ratings.groupby('userId').size()
        valid_users = user_counts[user_counts >= min_user].index
        ratings = ratings[ratings['userId'].isin(valid_users)]
        
        after = len(ratings)
        removed = before - after
        
        print(f"Iteration {i+1:2d}: Removed {removed:,} ratings → {after:,} remaining")
        
        if removed == 0:
            print(f"\n✓ Converged at iteration {i+1}")
            break
    
    # 최종 통계
    n_users = ratings['userId'].nunique()
    n_movies = ratings['movieId'].nunique()
    sparsity = 1 - (len(ratings) / (n_users * n_movies))
    
    print(f"\nFinal Statistics:")
    print(f"  Users:    {n_users:,}")
    print(f"  Movies:   {n_movies:,}")
    print(f"  Ratings:  {len(ratings):,}")
    print(f"  Sparsity: {sparsity:.4%}")
    print(f"{'='*60}\n")
    
    return ratings


def split_data(ratings, test_size=0.2, random_state=42):
    """
    Train/Test 분할
    
    Args:
        ratings: DataFrame
        test_size: 테스트 비율
        random_state: 랜덤 시드
    
    Returns:
        train, test DataFrames
    """
    print(f"\n{'='*60}")
    print(f"Splitting Data (test_size={test_size})")
    print(f"{'='*60}")
    
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"Train: {len(train):,} ratings ({len(train)/len(ratings)*100:.1f}%)")
    print(f"Test:  {len(test):,} ratings ({len(test)/len(ratings)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return train, test


def save_data(train, test, output_dir="./data"):
    """
    데이터 저장
    
    Args:
        train: Train DataFrame
        test: Test DataFrame
        output_dir: 저장 경로
    """
    print(f"\n{'='*60}")
    print(f"Saving Data")
    print(f"{'='*60}")
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 저장
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    # 파일 크기 확인
    train_size = os.path.getsize(train_path) / (1024**2)  # MB
    test_size = os.path.getsize(test_path) / (1024**2)
    
    print(f"✓ Saved train.csv ({train_size:.1f} MB)")
    print(f"✓ Saved test.csv ({test_size:.1f} MB)")
    print(f"{'='*60}\n")


def save_statistics(original_size, filtered_ratings, train, test, 
                   min_user, min_movie, output_dir="./data"):
    """
    전처리 통계 저장
    
    Args:
        original_size: 원본 데이터 크기
        filtered_ratings: 필터링된 DataFrame
        train: Train DataFrame
        test: Test DataFrame
        min_user: 사용자 최소 평점
        min_movie: 영화 최소 평점
        output_dir: 저장 경로
    """
    stats = {
        "original": {
            "total_ratings": original_size
        },
        "filtered": {
            "total_ratings": len(filtered_ratings),
            "total_users": int(filtered_ratings['userId'].nunique()),
            "total_movies": int(filtered_ratings['movieId'].nunique()),
            "retention_rate": len(filtered_ratings) / original_size
        },
        "split": {
            "train_size": len(train),
            "test_size": len(test)
        },
        "config": {
            "min_user_ratings": min_user,
            "min_movie_ratings": min_movie
        }
    }
    
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved statistics to {stats_path}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트 코드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing preprocess.py")
    print("="*60)
    
    # 샘플 데이터 생성
    sample_data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 4],
        'rating': [5, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3],
        'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }
    sample = pd.DataFrame(sample_data)
    
    print("\nSample data:")
    print(sample)
    
    # 필터링 테스트
    filtered = filter_data(sample, min_user=2, min_movie=2, max_iter=10)
    
    print("\nFiltered data:")
    print(filtered)
    
    # 분할 테스트
    train, test = split_data(filtered, test_size=0.2, random_state=42)
    
    print("\nTrain:")
    print(train)
    print("\nTest:")
    print(test)
    
    print("\n" + "="*60)
    print("✓ preprocess.py works!")
    print("="*60 + "\n")