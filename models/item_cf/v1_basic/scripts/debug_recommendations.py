"""
추천 시스템 디버깅
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import config
from src.evaluate import load_model, recommend_top_k

def main():
    print("\n" + "="*60)
    print("Debugging Recommendations")
    print("="*60)
    
    # 모델 로드
    model_data = load_model(f"{config.MODEL_DIR}/model.pkl")
    
    user_item_matrix = model_data['user_item_matrix']
    similarity = model_data['similarity']
    idx_to_movie = model_data['idx_to_movie']
    movie_map = model_data['movie_map']
    user_map = model_data['user_map']
    
    # 테스트 데이터 로드
    test = pd.read_csv(f"{config.OUTPUT_DIR}/test.csv")
    train = pd.read_csv(f"{config.OUTPUT_DIR}/train.csv")
    
    # 영화 정보 로드 (원본)
    movies = pd.read_csv(f"{config.RAW_DATA_PATH}".replace('ratings.csv', 'movies.csv'))
    
    print("\n" + "="*60)
    print("Test 1: 유사도 행렬 확인")
    print("="*60)
    
    # 유사도 확인
    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Non-zero values: {np.count_nonzero(similarity):,}")
    print(f"Total values: {similarity.size:,}")
    print(f"Sparsity: {1 - np.count_nonzero(similarity)/similarity.size:.4%}")
    
    # 샘플 유사도
    print("\nSample similarity (movie 0):")
    print(f"  Top 5 similar movies: {np.argsort(similarity[0])[::-1][1:6]}")
    print(f"  Similarity scores: {sorted(similarity[0], reverse=True)[1:6]}")
    
    print("\n" + "="*60)
    print("Test 2: 랜덤 사용자 추천")
    print("="*60)
    
    # 랜덤 사용자 선택
    test_users = test['userId'].unique()
    sample_user_ids = np.random.choice(test_users, 5, replace=False)
    
    for user_id in sample_user_ids:
        if user_id not in user_map:
            continue
            
        user_idx = user_map[user_id]
        
        print(f"\n━━━ User {user_id} (idx={user_idx}) ━━━")
        
        # 사용자가 본 영화 (train)
        user_train = train[train['userId'] == user_id]
        print(f"\nTrain ratings: {len(user_train)}")
        
        if len(user_train) > 0:
            user_train_sorted = user_train.sort_values('rating', ascending=False).head(5)
            print("\nTop 5 rated movies (train):")
            for _, row in user_train_sorted.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                movie_title = movies[movies['movieId'] == movie_id]['title'].values
                title = movie_title[0] if len(movie_title) > 0 else "Unknown"
                print(f"  {title[:50]:50s} | {rating:.1f}")
        
        # 추천
        recommended_indices = recommend_top_k(
            user_idx, 
            user_item_matrix, 
            similarity, 
            k=10
        )
        
        print(f"\nTop 10 recommendations:")
        for i, movie_idx in enumerate(recommended_indices, 1):
            movie_id = idx_to_movie[movie_idx]
            movie_title = movies[movies['movieId'] == movie_id]['title'].values
            title = movie_title[0] if len(movie_title) > 0 else "Unknown"
            print(f"  {i:2d}. {title[:50]}")
        
        # Ground truth (test)
        user_test = test[test['userId'] == user_id]
        if len(user_test) > 0:
            user_test_sorted = user_test.sort_values('rating', ascending=False).head(5)
            print(f"\nActual top rated movies (test):")
            for _, row in user_test_sorted.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                movie_title = movies[movies['movieId'] == movie_id]['title'].values
                title = movie_title[0] if len(movie_title) > 0 else "Unknown"
                print(f"  {title[:50]:50s} | {rating:.1f}")
        
        # 겹치는지 확인
        recommended_movie_ids = set([idx_to_movie[idx] for idx in recommended_indices])
        test_movie_ids = set(user_test[user_test['rating'] >= 3.5]['movieId'].values)
        
        hits = recommended_movie_ids & test_movie_ids
        print(f"\nHits: {len(hits)} / 10")
        if hits:
            print(f"Matched movies:")
            for movie_id in hits:
                movie_title = movies[movies['movieId'] == movie_id]['title'].values
                title = movie_title[0] if len(movie_title) > 0 else "Unknown"
                print(f"  - {title}")
    
    print("\n" + "="*60)
    print("Test 3: 유사도 품질 확인")
    print("="*60)
    
    # 인기 영화 몇 개 선택
    popular_movies = train['movieId'].value_counts().head(10)
    
    for movie_id in popular_movies.index[:3]:
        if movie_id not in movie_map:
            continue
            
        movie_idx = movie_map[movie_id]
        movie_title = movies[movies['movieId'] == movie_id]['title'].values
        title = movie_title[0] if len(movie_title) > 0 else "Unknown"
        
        print(f"\n━━━ {title[:50]} ━━━")
        
        # Top-5 유사 영화
        sim_scores = similarity[movie_idx].copy()
        sim_scores[movie_idx] = -np.inf
        
        top_5_indices = np.argsort(sim_scores)[::-1][:5]
        
        print("Most similar movies:")
        for i, idx in enumerate(top_5_indices, 1):
            similar_movie_id = idx_to_movie[idx]
            similar_title = movies[movies['movieId'] == similar_movie_id]['title'].values
            similar_title = similar_title[0] if len(similar_title) > 0 else "Unknown"
            score = sim_scores[idx]
            print(f"  {i}. {similar_title[:50]:50s} | {score:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()