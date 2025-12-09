# scripts/check_mapping.py
import pickle
from scipy.sparse import load_npz

# 매핑 로드
with open('../data/user_id_mapping.pkl', 'rb') as f:
    user_mappings = pickle.load(f)

user_to_idx = user_mappings['to_idx']

print(f"Number of users in mapping: {len(user_to_idx):,}")

# Matrix 확인
train_matrix = load_npz('../data/train_matrix.npz')
test_matrix = load_npz('../data/test_matrix.npz')

print(f"\nTrain matrix shape: {train_matrix.shape}")
print(f"Test matrix shape: {test_matrix.shape}")

# User factors 확인
import sys
sys.path.insert(0, '/home/ubuntu/ai-model/models/als/v1_basic')
from src.model import ALSRecommender
import config

model = ALSRecommender()
model.load_model(config.MODEL_PATH)

print(f"\nUser factors shape: {model.model.user_factors.shape}")
print(f"Item factors shape: {model.model.item_factors.shape}")