"""
Item-CF 학습 모듈
- User-Item 행렬 생성
- Item-Item 유사도 계산
- 모델 저장
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import time
import os


def build_user_item_matrix(train_df):
    """
    User-Item 희소 행렬 생성
    
    Args:
        train_df: Train DataFrame (userId, movieId, rating, timestamp)
    
    Returns:
        matrix: scipy.sparse.csr_matrix (n_users × n_movies)
        user_map: dict {userId → user_idx}
        movie_map: dict {movieId → movie_idx}
    """
    print("\n" + "="*60)
    print("Building User-Item Matrix")
    print("="*60)
    
    # ID 매핑 생성
    user_ids = train_df['userId'].unique()
    movie_ids = train_df['movieId'].unique()
    
    user_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    movie_map = {mid: idx for idx, mid in enumerate(sorted(movie_ids))}
    
    print(f"Users:  {len(user_ids):,}")
    print(f"Movies: {len(movie_ids):,}")
    
    # 역방향 매핑도 저장 (나중에 사용)
    idx_to_user = {idx: uid for uid, idx in user_map.items()}
    idx_to_movie = {idx: mid for mid, idx in movie_map.items()}
    
    # 인덱스 변환
    print("\nConverting IDs to indices...")
    user_indices = train_df['userId'].map(user_map).values
    movie_indices = train_df['movieId'].map(movie_map).values
    ratings = train_df['rating'].values
    
    # 희소 행렬 생성
    print("Creating sparse matrix...")
    matrix = csr_matrix(
        (ratings, (user_indices, movie_indices)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32
    )
    
    # 통계
    n_ratings = matrix.nnz
    sparsity = 1 - (n_ratings / (matrix.shape[0] * matrix.shape[1]))
    
    print(f"\nMatrix Statistics:")
    print(f"  Shape:    {matrix.shape}")
    print(f"  Ratings:  {n_ratings:,}")
    print(f"  Sparsity: {sparsity:.4%}")
    print(f"  Memory:   {matrix.data.nbytes / (1024**2):.1f} MB")
    print("="*60 + "\n")
    
    return matrix, user_map, movie_map, idx_to_user, idx_to_movie


def compute_item_similarity_cpu(matrix, top_k=100, shrinkage=50, apply_idf=True):
    """
    CPU 기반 Item-Item 유사도 계산 (IDF & Shrinkage 적용)
    
    Args:
        matrix: User-Item 행렬 (n_users × n_movies)
        top_k: 각 아이템당 저장할 유사 아이템 수
        shrinkage: 공통 평가 수가 적을 때 유사도를 낮추는 상수 (보통 10~100)
        apply_idf: IDF 가중치 적용 여부 (Coverage 향상)
    
    Returns:
        similarity: numpy array (n_movies × n_movies)
    """
    print("\n" + "="*60)
    print("Computing Item Similarity (Advanced CPU)")
    print("="*60)
    
    start_time = time.time()
    n_users, n_items = matrix.shape
    
    # 1. IDF 가중치 적용 (Coverage 개선 핵심)
    # 유명한 영화(모두가 본 영화)의 영향력을 낮춤
    if apply_idf:
        print("Applying IDF weighting...")
        # 아이템별 평가 횟수
        item_counts = np.array(matrix.getnnz(axis=0))
        # IDF 공식: log(총 유저 수 / (아이템 평가 수 + 1))
        idf = np.log1p(n_users) - np.log1p(item_counts)
        
        # 행렬에 IDF 가중치 곱하기 (Diagonal Matrix 곱셈과 동일 효과)
        # matrix는 (User x Item)이므로, 각 열(Item)에 idf 값을 곱해줌
        from scipy.sparse import diags
        idf_diag = diags(idf)
        weighted_matrix = matrix @ idf_diag
    else:
        weighted_matrix = matrix

    # 2. 코사인 유사도 계산
    print("Calculating cosine similarity...")
    # (Item x User) 형태로 전치
    item_matrix = weighted_matrix.T.tocsr()
    similarity = cosine_similarity(item_matrix, dense_output=True)
    
    elapsed = time.time() - start_time
    print(f"Base similarity calculation: {elapsed/60:.1f} minutes")

    # 3. Shrinkage 적용 (신뢰도/Precision 개선 핵심)
    # 공통 평가자 수(Support)가 적으면 유사도를 0으로 수렴시킴
    # 공식: Sim_new = Sim_old * (Support / (Support + shrinkage))
    if shrinkage > 0:
        print(f"Applying shrinkage (param={shrinkage})...")
        start_time = time.time()
        
        # 이진 행렬 생성 (0보다 크면 1, 아니면 0)
        binary_matrix = (matrix > 0).astype(int)
        
        # 공통 평가자 수 계산: (Item x User) @ (User x Item)
        # item_matrix가 전치된 상태이므로 binary_matrix.T 사용
        common_support = binary_matrix.T @ binary_matrix
        # Dense matrix로 변환 (메모리 주의: 아이템 수가 5만 개 넘으면 주의)
        common_support = common_support.toarray()
        
        # Shrinkage 계수 계산
        shrinkage_factor = common_support / (common_support + shrinkage)
        
        # 유사도에 적용
        similarity = similarity * shrinkage_factor
        
        elapsed = time.time() - start_time
        print(f"Shrinkage application: {elapsed/60:.1f} minutes")

    # 4. Top-K 필터링
    if top_k:
        print(f"\nFiltering top-{top_k} similar items...")
        
        # 대각선(자기 자신) 0으로 처리
        np.fill_diagonal(similarity, 0)
        
        for i in range(n_items):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{n_items} ({i/n_items*100:.1f}%)")
            
            row = similarity[i]
            
            # 상위 K개 인덱스 추출 (속도를 위해 argpartition 사용 권장)
            if len(row) > top_k:
                # 상위 K개가 아닌 것들은 0으로 만듦
                # argpartition은 정렬하지 않고 K번째 큰 값을 기준으로 나눔 (빠름)
                params = np.argpartition(row, -top_k)[-top_k:]
                
                # 마스킹: params에 포함되지 않은 인덱스는 0 처리
                mask = np.ones(n_items, dtype=bool)
                mask[params] = False
                row[mask] = 0
                similarity[i] = row

    print(f"\nFinal Similarity Matrix:")
    print(f"  Shape:  {similarity.shape}")
    print(f"  Memory: {similarity.nbytes / (1024**2):.1f} MB")
    print("="*60 + "\n")
    
    return similarity

def compute_item_similarity_gpu(matrix, top_k=100, shrinkage=50, apply_idf=True):
    """
    GPU 기반 Item-Item 유사도 계산 (IDF & Shrinkage 적용)
    
    Args:
        matrix: User-Item 행렬 (n_users × n_movies)
        top_k: Top-K 필터링
        shrinkage: 수축 파라미터 (공통 평가자 수 보정)
        apply_idf: IDF 가중치 적용 여부
    """
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
        from cupyx.scipy.sparse import diags as diags_gpu
    except ImportError:
        print("CuPy not available. Falling back to CPU.")
        return compute_item_similarity_cpu(matrix, top_k, shrinkage, apply_idf)
    
    print("\n" + "="*60)
    print("Computing Item Similarity (Advanced GPU)")
    print("="*60)
    
    start_time = time.time()
    n_users, n_items = matrix.shape
    
    # 1. 데이터 GPU 전송 및 IDF 적용
    print("Transferring data to GPU...")
    # 원본 행렬 (User x Item)
    matrix_gpu = csr_matrix_gpu(matrix)
    
    if apply_idf:
        print("Applying IDF weighting on GPU...")
        # 열별(아이템별) 평가 수 계산
        # getnnz()가 cupy sparse에 없을 수 있으므로 sum 후 변환
        item_counts = matrix_gpu.getnnz(axis=0) # cupyx 최신버전 지원
        # 또는: item_counts = cp.diff(matrix_gpu.indptr) (CSC인 경우)
        
        # IDF 계산: log(N / (df + 1)) -> log1p(N) - log1p(df)
        idf = cp.log1p(n_users) - cp.log1p(item_counts)
        
        # 대각 행렬로 곱하기
        idf_diag = diags_gpu(idf)
        weighted_matrix_gpu = matrix_gpu.dot(idf_diag)
    else:
        weighted_matrix_gpu = matrix_gpu
        
    # 유사도 계산을 위해 전치 (Item x User)
    item_matrix_gpu = weighted_matrix_gpu.T
    
    # Shrinkage 계산용 이진 행렬 (Item x User)
    if shrinkage > 0:
        binary_matrix_gpu = (matrix_gpu > 0).astype(cp.float32).T
    
    # 결과 담을 행렬 (CPU 메모리) - 크기가 크면 여기서 터질 수 있으니 주의
    similarity = np.zeros((n_items, n_items), dtype=np.float32)
    
    # 배치 처리
    batch_size = 500  # GPU 메모리에 따라 조절
    
    print(f"Processing {n_items} items in batches of {batch_size}...")
    
    # 전체 행렬 (비교 대상) 미리 준비
    # 메모리가 부족하면 루프 안에서 처리해야 함
    try:
        all_items_dense = item_matrix_gpu.toarray() # (Item x User) Dense
        if shrinkage > 0:
            all_binary_dense = binary_matrix_gpu.toarray()
    except cp.cuda.memory.OutOfMemoryError:
        print("⚠ Not enough GPU memory to hold full dense matrix. Switching to CPU mode.")
        return compute_item_similarity_cpu(matrix, top_k, shrinkage, apply_idf)

    # Norm 미리 계산 (분모)
    all_norms = cp.linalg.norm(all_items_dense, axis=1)
    
    for i in range(0, n_items, batch_size):
        end_i = min(i + batch_size, n_items)
        if i % 1000 == 0:
            print(f"  Progress: {i}/{n_items} ({i/n_items*100:.1f}%)")
        
        # 2. 코사인 유사도 계산 (Batch)
        batch_dense = all_items_dense[i:end_i]
        batch_norms = all_norms[i:end_i]
        
        # 분자: 내적 (Batch x All_T)
        numerator = cp.dot(batch_dense, all_items_dense.T)
        
        # 분모: Norm 곱 (Outer product)
        # (Batch_norms x 1) * (1 x All_norms)
        denominator = cp.outer(batch_norms, all_norms)
        
        batch_sim = numerator / (denominator + 1e-8)
        
        # 3. Shrinkage 적용 (Batch)
        if shrinkage > 0:
            batch_binary = all_binary_dense[i:end_i]
            # 공통 평가자 수 계산 (내적)
            support = cp.dot(batch_binary, all_binary_dense.T)
            # Shrinkage Factor
            factor = support / (support + shrinkage)
            # 적용
            batch_sim *= factor
            
        # CPU로 복사
        similarity[i:end_i] = cp.asnumpy(batch_sim)
        
        # 메모리 정리
        del numerator, denominator, batch_sim
        cp.get_default_memory_pool().free_all_blocks()

    elapsed = time.time() - start_time
    print(f"\nSimilarity calculation (GPU): {elapsed/60:.1f} minutes")
    
    # 4. Top-K 필터링 (CPU에서 수행 - 구현 용이성 및 속도)
    if top_k:
        print(f"Filtering top-{top_k} similar items (CPU)...")
        # CPU 함수 로직 재사용 권장 혹은 직접 구현
        # 여기서는 직접 구현
        for i in range(n_items):
            row = similarity[i]
            # 자기 자신 0
            row[i] = 0
            if len(row) > top_k:
                # argpartition으로 상위 k개만 남김
                params = np.argpartition(row, -top_k)[-top_k:]
                mask = np.ones(n_items, dtype=bool)
                mask[params] = False
                row[mask] = 0
                similarity[i] = row

    print(f"\nFinal Similarity Matrix:")
    print(f"  Shape:  {similarity.shape}")
    print(f"  Memory: {similarity.nbytes / (1024**2):.1f} MB")
    print("="*60 + "\n")
    
    return similarity

def save_model(matrix, similarity, user_map, movie_map, 
               idx_to_user, idx_to_movie, output_dir="./outputs"):
    """
    모델 저장
    
    Args:
        matrix: User-Item 행렬
        similarity: Item-Item 유사도 행렬
        user_map: userId → idx 매핑
        movie_map: movieId → idx 매핑
        idx_to_user: idx → userId 매핑
        idx_to_movie: idx → movieId 매핑
        output_dir: 저장 경로
    """
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 유사도 행렬만 저장 (numpy)
    similarity_path = os.path.join(output_dir, "item_similarity.npy")
    np.save(similarity_path, similarity)
    
    similarity_size = os.path.getsize(similarity_path) / (1024**2)
    print(f"✓ Saved item_similarity.npy ({similarity_size:.1f} MB)")
    
    # 2. 전체 모델 저장 (pickle)
    model_data = {
        'user_item_matrix': matrix,
        'similarity': similarity,
        'user_map': user_map,
        'movie_map': movie_map,
        'idx_to_user': idx_to_user,
        'idx_to_movie': idx_to_movie,
        'n_users': matrix.shape[0],
        'n_movies': matrix.shape[1]
    }
    
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    model_size = os.path.getsize(model_path) / (1024**2)
    print(f"✓ Saved model.pkl ({model_size:.1f} MB)")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*60 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트 코드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing train.py")
    print("="*60)
    
    # 샘플 데이터 생성
    sample_data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4] * 3,
        'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4] * 3,
        'rating': [5, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3, 4] * 3,
        'timestamp': list(range(36))
    }
    sample_df = pd.DataFrame(sample_data)
    
    print("\nSample data:")
    print(sample_df.head(10))
    
    # 1. 행렬 생성 테스트
    matrix, user_map, movie_map, idx_to_user, idx_to_movie = build_user_item_matrix(sample_df)
    
    print("\nUser map:", user_map)
    print("Movie map:", movie_map)
    
    # 2. 유사도 계산 테스트 (CPU)
    similarity = compute_item_similarity_cpu(matrix, top_k=2)
    
    print("\nSimilarity matrix (sample):")
    print(similarity[:3, :3])
    
    # 3. 저장 테스트
    save_model(matrix, similarity, user_map, movie_map, 
               idx_to_user, idx_to_movie, "./test_outputs")
    
    print("\n" + "="*60)
    print("✓ train.py works!")
    print("="*60 + "\n")