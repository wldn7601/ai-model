import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix, save_npz
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 경로 설정
DATA_DIR = '/home/ubuntu/ai-model/models/light_gcn/data'
os.makedirs(DATA_DIR, exist_ok=True)

RATINGS_PATH = '/home/ubuntu/ai-model/datas/data/ratings_tmdb.csv'

print("=== LightGCN Data Preprocessing (TMDB) ===\n")

# 1. 데이터 로드 (필요한 컬럼만)
print("Loading data...")
df = pd.read_csv(RATINGS_PATH, usecols=['userId', 'tmdbId', 'rating', 'timestamp'])
print(f"Original data: {len(df):,} ratings")
print(f"Users: {df['userId'].nunique():,}")
print(f"Movies (TMDB): {df['tmdbId'].nunique():,}\n")

# threshold 설정
THRESHOLD = None

# 2. Random Split 함수
def random_split(df, test_ratio=0.2):
    print("Performing Random Split (8:2)...")
    
    # 1. Threshold 적용 (None이면 스킵)
    if THRESHOLD is not None:
        df = df[df['rating'] >= THRESHOLD].copy()
        print(f"Filtered ratings < {THRESHOLD}")
    
    # 2. 유저별로 최소 5개 이상의 로그가 있는 경우만 살리기
    user_counts = df.groupby('userId').size()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['userId'].isin(valid_users)].copy()
    print(f"Filtered users with < 5 interactions. Remaining: {len(df):,}")

    # 3. Stratified Split (유저별 비율 유지하며 랜덤 분할)
    train_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=42, 
        stratify=df['userId']
    )
    
    return train_df, test_df

# 랜덤 분할
train_df, test_df = random_split(df, test_ratio=0.2)

print("=== Random Split (80:20) ===")
print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.2f}%)")
print(f"Test: {len(test_df):,} ({len(test_df)/len(df)*100:.2f}%)\n")

# 원본 분할 데이터 저장
print("Saving split data...")
train_df.to_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), index=False)
test_df.to_csv(os.path.join(DATA_DIR, 'test_ratings.csv'), index=False)

# 3. Implicit Feedback 변환
def convert_to_implicit(df, threshold=None):
    """벡터화된 implicit 변환"""
    if threshold is None:
        return df.copy()
    else:
        return df[df['rating'] >= threshold].copy()

print("Converting to implicit feedback...")
train_implicit = convert_to_implicit(train_df, threshold=THRESHOLD)
test_implicit = convert_to_implicit(test_df, threshold=THRESHOLD)

print("=== Implicit Feedback Conversion ===")
print(f"Threshold: {'All ratings' if THRESHOLD is None else f'>= {THRESHOLD}'}")
print(f"Train: {len(train_df):,} -> {len(train_implicit):,}")
print(f"Test: {len(test_df):,} -> {len(test_implicit):,}\n")

# Implicit 데이터 저장
train_implicit.to_csv(os.path.join(DATA_DIR, 'train_implicit.csv'), index=False)
test_implicit.to_csv(os.path.join(DATA_DIR, 'test_implicit.csv'), index=False)

# 4. User/Item ID 재매핑
class DataProcessor:
    def __init__(self, train_df, test_df):
        print("Creating ID mappings...")
        
        self.train_df = train_df
        self.test_df = test_df
        
        # 모든 unique user와 item 수집 (tmdbId 사용)
        all_users = np.union1d(train_df['userId'].unique(), test_df['userId'].unique())
        all_items = np.union1d(train_df['tmdbId'].unique(), test_df['tmdbId'].unique())
        
        # ID 매핑 생성
        self.user2id = {user: idx for idx, user in enumerate(all_users)}
        self.tmdb2id = {tmdb: idx for idx, tmdb in enumerate(all_items)}
        
        self.id2user = {idx: user for user, idx in self.user2id.items()}
        self.id2tmdb = {idx: tmdb for tmdb, idx in self.tmdb2id.items()}
        
        self.n_users = len(self.user2id)
        self.n_items = len(self.tmdb2id)
    
    def remap_ids(self, df):
        """벡터화된 ID 재매핑 (tmdbId 기반)"""
        df_remapped = df.copy()
        df_remapped['user_idx'] = df_remapped['userId'].map(self.user2id)
        df_remapped['item_idx'] = df_remapped['tmdbId'].map(self.tmdb2id)
        return df_remapped
    
    def create_interaction_matrix(self, df):
        """벡터화된 interaction matrix 생성"""
        df_remapped = self.remap_ids(df)
        
        users = df_remapped['user_idx'].values
        items = df_remapped['item_idx'].values
        interactions = np.ones(len(df_remapped), dtype=np.float32)
        
        matrix = csr_matrix(
            (interactions, (users, items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        return matrix
    
    def create_graph_edge_index(self, interaction_matrix):
        """벡터화된 edge index 생성"""
        print("Creating graph edge index...")
        coo = interaction_matrix.tocoo()
        
        # User -> Item edges
        user_indices = coo.row
        item_indices = coo.col + self.n_users
        
        # Undirected graph (양방향)
        edge_index = np.vstack([
            np.concatenate([user_indices, item_indices]),
            np.concatenate([item_indices, user_indices])
        ])
        
        return torch.LongTensor(edge_index)
    
    def save(self, save_dir):
        """매핑 정보 저장 (tmdb 매핑 포함)"""
        mappings = {
            'user2id': self.user2id,
            'tmdb2id': self.tmdb2id,
            'id2user': self.id2user,
            'id2tmdb': self.id2tmdb,
            'n_users': self.n_users,
            'n_items': self.n_items
        }
        
        with open(os.path.join(save_dir, 'id_mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)

# 데이터 처리
processor = DataProcessor(train_implicit, test_implicit)

print("\n=== ID Remapping ===")
print(f"Total users: {processor.n_users:,}")
print(f"Total items (TMDB): {processor.n_items:,}\n")

# 재매핑된 데이터 생성 및 저장
print("Remapping IDs...")
train_remapped = processor.remap_ids(train_implicit)
test_remapped = processor.remap_ids(test_implicit)

train_remapped.to_csv(os.path.join(DATA_DIR, 'train_remapped.csv'), index=False)
test_remapped.to_csv(os.path.join(DATA_DIR, 'test_remapped.csv'), index=False)

# 5. Interaction Matrix 생성
print("Creating interaction matrices...")
train_matrix = processor.create_interaction_matrix(train_implicit)
test_matrix = processor.create_interaction_matrix(test_implicit)

print("\n=== Interaction Matrices ===")
print(f"Train matrix shape: {train_matrix.shape}")
print(f"Train interactions: {train_matrix.nnz:,}")
print(f"Train sparsity: {(1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])) * 100:.4f}%")
print(f"Test matrix shape: {test_matrix.shape}")
print(f"Test interactions: {test_matrix.nnz:,}\n")

# Sparse matrix 저장
print("Saving sparse matrices...")
save_npz(os.path.join(DATA_DIR, 'train_matrix.npz'), train_matrix)
save_npz(os.path.join(DATA_DIR, 'test_matrix.npz'), test_matrix)

# 6. Graph Edge Index 생성
edge_index = processor.create_graph_edge_index(train_matrix)

print("\n=== Graph Structure ===")
print(f"Total nodes: {processor.n_users + processor.n_items:,}")
print(f"Total edges: {edge_index.shape[1]:,}")
print(f"Average degree: {edge_index.shape[1] / (processor.n_users + processor.n_items):.2f}\n")

# Edge index 저장
print("Saving edge index...")
torch.save(edge_index, os.path.join(DATA_DIR, 'edge_index.pt'))

# 7. 매핑 정보 저장
print("Saving ID mappings...")
processor.save(DATA_DIR)

# 8. 메타데이터 저장
metadata = {
    'threshold': THRESHOLD,
    'n_users': processor.n_users,
    'n_items': processor.n_items,
    'train_interactions': train_matrix.nnz,
    'test_interactions': test_matrix.nnz,
    'total_nodes': processor.n_users + processor.n_items,
    'total_edges': edge_index.shape[1],
    'train_size': len(train_implicit),
    'test_size': len(test_implicit)
}

with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

print("\n=== Files Saved ===")
saved_files = [
    'train_ratings.csv',          # 원본 train 분할
    'test_ratings.csv',           # 원본 test 분할
    'train_implicit.csv',         # Implicit 변환 train
    'test_implicit.csv',          # Implicit 변환 test
    'train_remapped.csv',         # ID 재매핑 train
    'test_remapped.csv',          # ID 재매핑 test
    'train_matrix.npz',           # Train interaction matrix
    'test_matrix.npz',            # Test interaction matrix
    'edge_index.pt',              # Graph edge index
    'id_mappings.pkl',            # ID 매핑 정보 (tmdb 포함)
    'metadata.pkl'                # 메타데이터
]

for file in saved_files:
    filepath = os.path.join(DATA_DIR, file)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ {file:<25} ({size:.2f} MB)")

print(f"\nAll files saved to: {DATA_DIR}")
print("Preprocessing complete!")