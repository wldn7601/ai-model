# /home/ubuntu/ai-model/models/light_gcn/scripts/run_train_and_evalute_lightgcn.py

"""
BPR(Bayesian Personalized Ranking) Loss를 사용하여 학습
매 epoch마다 validation 평가 수행
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import load_npz
from tqdm import tqdm

# 경로 설정
DATA_DIR = '/home/ubuntu/ai-model/models/light_gcn/data'
MODEL_DIR = '/home/ubuntu/ai-model/models/light_gcn/checkpoints'
os.makedirs(MODEL_DIR, exist_ok=True)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# 메타데이터 로드
print("Loading metadata...")
with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

n_users = metadata['n_users']
n_items = metadata['n_items']
print(f"Users: {n_users:,}")
print(f"Items: {n_items:,}")
print(f"Train interactions: {metadata['train_interactions']:,}\n")

# Edge index 로드
print("Loading edge index...")
edge_index = torch.load(os.path.join(DATA_DIR, 'edge_index.pt'))
print(f"Edge index shape: {edge_index.shape}\n")

# Train/Test matrix 로드
print("Loading matrices...")
train_matrix = load_npz(os.path.join(DATA_DIR, 'train_matrix.npz'))
test_matrix = load_npz(os.path.join(DATA_DIR, 'test_matrix.npz')).tocsr()
print(f"Train matrix shape: {train_matrix.shape}")
print(f"Test matrix shape: {test_matrix.shape}\n")


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, edge_index, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Embedding 초기화
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        print(f"Model initialized:")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Layers: {n_layers}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        
        # 그래프 구조 미리 계산
        self.graph = self._get_sparse_graph(edge_index)
    
    def _get_sparse_graph(self, edge_index):
        """정규화된 인접 행렬 생성"""
        print("Generating normalized graph structure...")
        num_nodes = self.n_users + self.n_items
        
        device = self.user_embedding.weight.device
        edge_index = edge_index.to(device)
        
        vals = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(edge_index, vals, (num_nodes, num_nodes))
        
        row, col = edge_index
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]
        
        normalized_graph = torch.sparse_coo_tensor(
            edge_index, 
            norm_vals, 
            (num_nodes, num_nodes)
        ).coalesce()
        
        print(f"Graph structure created: {num_nodes:,} nodes\n")
        return normalized_graph
    
    def forward(self):
        """LightGCN forward propagation"""
        if self.graph.device != self.user_embedding.weight.device:
            self.graph = self.graph.to(self.user_embedding.weight.device)

        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        embeddings_list = [all_embeddings]
        
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.graph, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        final_embeddings = torch.stack(embeddings_list, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        
        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]
        
        return user_embeddings, item_embeddings
    
    def bpr_loss(self, users, pos_items, neg_items, user_embeddings, item_embeddings):
        """BPR Loss"""
        user_emb = user_embeddings[users]
        pos_item_emb = item_embeddings[pos_items]
        neg_item_emb = item_embeddings[neg_items]
        
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        return loss
    
    def regularization_loss(self, users, pos_items, neg_items):
        """L2 regularization"""
        user_emb = self.user_embedding(users)
        pos_item_emb = self.item_embedding(pos_items)
        neg_item_emb = self.item_embedding(neg_items)
        
        reg_loss = (1/2) * (
            user_emb.norm(2).pow(2) + 
            pos_item_emb.norm(2).pow(2) + 
            neg_item_emb.norm(2).pow(2)
        ) / float(len(users))
        
        return reg_loss


class BPRSampler:
    """Negative sampling for BPR"""
    def __init__(self, train_matrix, batch_size=1024):
        self.train_matrix = train_matrix.tocsr()
        self.n_users = train_matrix.shape[0]
        self.n_items = train_matrix.shape[1]
        self.batch_size = batch_size
        
        self.user_pos_items = {}
        for user in range(self.n_users):
            pos_items = self.train_matrix[user].indices
            if len(pos_items) > 0:
                self.user_pos_items[user] = pos_items
        
        self.users = list(self.user_pos_items.keys())
        print(f"Sampler initialized: {len(self.users):,} active users")
    
    def sample(self):
        """한 epoch의 모든 배치 생성"""
        np.random.shuffle(self.users)
        
        for i in range(0, len(self.users), self.batch_size):
            batch_users = self.users[i:i + self.batch_size]
            
            users_list = []
            pos_items_list = []
            neg_items_list = []
            
            for user in batch_users:
                pos_items = self.user_pos_items[user]
                pos_item = np.random.choice(pos_items)
                
                while True:
                    neg_item = np.random.randint(0, self.n_items)
                    if neg_item not in pos_items:
                        break
                
                users_list.append(user)
                pos_items_list.append(pos_item)
                neg_items_list.append(neg_item)
            
            yield (
                torch.LongTensor(users_list).to(device),
                torch.LongTensor(pos_items_list).to(device),
                torch.LongTensor(neg_items_list).to(device)
            )


def evaluate(model, train_matrix, test_matrix, top_k=20, batch_size=1024):
    """GPU 가속 평가"""
    model.eval()
    device_eval = next(model.parameters()).device
    
    recalls = []
    ndcgs = []
    
    test_users_idx = np.where(test_matrix.getnnz(axis=1) > 0)[0]
    
    with torch.no_grad():
        user_all_emb, item_all_emb = model()
        
        for start_idx in range(0, len(test_users_idx), batch_size):
            end_idx = min(start_idx + batch_size, len(test_users_idx))
            batch_users = test_users_idx[start_idx:end_idx]
            
            batch_u_emb = user_all_emb[batch_users]
            scores = torch.matmul(batch_u_emb, item_all_emb.t())
            
            for i, user_id in enumerate(batch_users):
                train_pos_items = train_matrix[user_id].indices
                scores[i, train_pos_items] = -float('inf')
            
            _, top_k_items = torch.topk(scores, k=top_k)
            top_k_items = top_k_items.cpu().numpy()
            
            for i, user_id in enumerate(batch_users):
                ground_truth = set(test_matrix[user_id].indices)
                
                if len(ground_truth) == 0:
                    continue
                
                recommended = top_k_items[i]
                hits = [1 if item in ground_truth else 0 for item in recommended]
                
                recall = sum(hits) / min(len(ground_truth), top_k)
                recalls.append(recall)
                
                dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
                idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(ground_truth), top_k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
                
    return np.mean(recalls), np.mean(ndcgs)


def train_epoch(model, sampler, optimizer, epoch, n_epochs, reg_weight=1e-4):
    """한 epoch 학습"""
    model.train()
    
    total_loss = 0
    total_bpr_loss = 0
    total_reg_loss = 0
    n_batches = 0
    
    total_batches = (len(sampler.users) + sampler.batch_size - 1) // sampler.batch_size
    desc_text = f"Epoch {epoch:02d}/{n_epochs}"
    
    for users, pos_items, neg_items in tqdm(sampler.sample(), total=total_batches, desc=desc_text, leave=False):
        optimizer.zero_grad()
        
        user_embeddings, item_embeddings = model()
        
        bpr_loss = model.bpr_loss(users, pos_items, neg_items, user_embeddings, item_embeddings)
        reg_loss = model.regularization_loss(users, pos_items, neg_items)
        
        loss = bpr_loss + reg_weight * reg_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_bpr_loss += bpr_loss.item()
        total_reg_loss += reg_loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'bpr_loss': total_bpr_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches
    }


# 하이퍼파라미터
EMBEDDING_DIM = 64
N_LAYERS = 3
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
REG_WEIGHT = 1e-4
N_EPOCHS = 5
TOP_K = 20

print("=== Training Configuration ===")
print(f"Embedding dim: {EMBEDDING_DIM}")
print(f"Layers: {N_LAYERS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Regularization: {REG_WEIGHT}")
print(f"Epochs: {N_EPOCHS}")
print(f"Evaluation Top-K: {TOP_K}\n")

# 모델 초기화
model = LightGCN(
    n_users=n_users,
    n_items=n_items,
    edge_index=edge_index,
    embedding_dim=EMBEDDING_DIM,
    n_layers=N_LAYERS
).to(device)
print()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Sampler
sampler = BPRSampler(train_matrix, batch_size=BATCH_SIZE)
print()

# 학습
print("=== Training Started ===\n")
history = {
    'loss': [],
    'bpr_loss': [],
    'reg_loss': [],
    'recall': [],
    'ndcg': []
}

best_recall = 0
best_epoch = 0

for epoch in range(1, N_EPOCHS + 1):
    start_time = time.time()
    
    # Train
    metrics = train_epoch(model, sampler, optimizer, epoch, N_EPOCHS, reg_weight=REG_WEIGHT)
    
    # Evaluate
    recall, ndcg = evaluate(model, train_matrix, test_matrix, top_k=TOP_K)
    
    epoch_time = time.time() - start_time
    
    # History 저장
    history['loss'].append(metrics['loss'])
    history['bpr_loss'].append(metrics['bpr_loss'])
    history['reg_loss'].append(metrics['reg_loss'])
    history['recall'].append(recall)
    history['ndcg'].append(ndcg)
    
    # 출력
    print(f"Epoch {epoch:02d}/{N_EPOCHS} | "
          f"Loss: {metrics['loss']:.4f} | "
          f"Recall@{TOP_K}: {recall:.4f} | "
          f"NDCG@{TOP_K}: {ndcg:.4f} | "
          f"Time: {epoch_time:.1f}s")
    
    # Best model 저장 (Recall 기준)
    if recall > best_recall:
        best_recall = recall
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'],
            'recall': recall,
            'ndcg': ndcg,
            'n_users': n_users,
            'n_items': n_items,
            'embedding_dim': EMBEDDING_DIM,
            'n_layers': N_LAYERS
        }, os.path.join(MODEL_DIR, 'best_model.pt'))
    
    # 주기적으로 checkpoint 저장
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'],
            'recall': recall,
            'ndcg': ndcg,
            'history': history,
            'n_users': n_users,
            'n_items': n_items,
            'embedding_dim': EMBEDDING_DIM,
            'n_layers': N_LAYERS
        }, os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch}.pt'))

# 최종 모델 저장
torch.save({
    'epoch': N_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': metrics['loss'],
    'recall': recall,
    'ndcg': ndcg,
    'history': history,
    'n_users': n_users,
    'n_items': n_items,
    'embedding_dim': EMBEDDING_DIM,
    'n_layers': N_LAYERS
}, os.path.join(MODEL_DIR, 'final_model.pt'))

# History 저장
with open(os.path.join(MODEL_DIR, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history, f)

print(f"\n=== Training Complete ===")
print(f"Best Recall@{TOP_K}: {best_recall:.4f} (Epoch {best_epoch})")
print(f"Final Recall@{TOP_K}: {recall:.4f}")
print(f"Final NDCG@{TOP_K}: {ndcg:.4f}")
print(f"Models saved to: {MODEL_DIR}")