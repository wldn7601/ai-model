import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
import time

"""
Simple Hybrid Recommender (No Filtering)
- SBERT (70%) + LightGCN (30%)
- 필터링 제거 (런타임, OTT, 장르 등 X)
- 순수 하이브리드 점수 기반 추천
"""

class SimpleHybridRecommender:
    def __init__(
        self,
        sbert_embeddings_path: str,
        lightgcn_model_path: str,
        lightgcn_data_path: str,
        metadata_path: str,
        sbert_weight: float = 0.7,
        lightgcn_weight: float = 0.3,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sbert_weight = sbert_weight
        self.lightgcn_weight = lightgcn_weight
        
        print("Initializing Simple Hybrid Recommender...")
        
        # 1. 데이터 로드
        self._load_sbert_data(sbert_embeddings_path)
        self._load_lightgcn_data(lightgcn_data_path)
        self._load_lightgcn_model(lightgcn_model_path)
        self._load_metadata(metadata_path)
        
        # 2. Pre-alignment (교집합 사전 정렬)
        print("Pre-aligning models for fast inference...")
        
        # 교집합 ID 추출
        common_ids = set(self.sbert_movie_to_idx.keys()) & set(self.lightgcn_movie_to_idx.keys())
        self.common_movie_ids = sorted(list(common_ids))
        
        # 교집합 영화들의 임베딩 행렬 구축
        self.target_sbert_matrix = []
        self.target_lightgcn_matrix = []
        
        for mid in self.common_movie_ids:
            s_idx = self.sbert_movie_to_idx[mid]
            self.target_sbert_matrix.append(self.sbert_embeddings[s_idx])
            
            l_idx = self.lightgcn_movie_to_idx[mid]
            self.target_lightgcn_matrix.append(self.lightgcn_item_embeddings[l_idx])
        
        self.target_sbert_matrix = np.array(self.target_sbert_matrix)
        self.target_lightgcn_matrix = np.array(self.target_lightgcn_matrix)
        
        # SBERT 정규화 (Cosine Similarity용)
        self.target_sbert_norm = self.target_sbert_matrix / (
            np.linalg.norm(self.target_sbert_matrix, axis=1, keepdims=True) + 1e-10
        )
        
        print(f"Pre-alignment complete. Target movies: {len(self.common_movie_ids)}")
        
        self.scaler = MinMaxScaler()

    def _load_sbert_data(self, embeddings_path: str):
        print(f"Loading SBERT embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        # tmdbId 컬럼 사용
        self.sbert_movie_ids = data['tmdbId'].tolist()
        self.sbert_embeddings = np.array(data['embedding'].tolist(), dtype='float32')
        self.sbert_movie_to_idx = {mid: idx for idx, mid in enumerate(self.sbert_movie_ids)}
        
        print(f"  SBERT movies: {len(self.sbert_movie_ids):,}")
    
    def _load_lightgcn_data(self, data_path: str):
        data_path = Path(data_path)
        with open(data_path / 'id_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        # tmdb2id 매핑 사용
        self.lightgcn_movie_to_idx = mappings['tmdb2id']
        self.lightgcn_idx_to_movie = mappings['id2tmdb']
        
        print(f"  LightGCN movies: {len(self.lightgcn_movie_to_idx):,}")
    
    def _load_lightgcn_model(self, model_path: str):
        print(f"Loading LightGCN model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 키 에러 방지
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.lightgcn_item_embeddings = checkpoint['model_state_dict']['item_embedding.weight'].cpu().numpy()
            elif 'item_embeddings' in checkpoint:
                self.lightgcn_item_embeddings = checkpoint['item_embeddings'].cpu().numpy()
            else:
                self.lightgcn_item_embeddings = checkpoint['item_embedding.weight'].cpu().numpy()
    
    def _load_metadata(self, path: str):
        print(f"Loading metadata from {path}")
        df = pd.read_csv(path)
        
        # tmdbId 인덱스 사용
        if 'tmdbId' in df.columns:
            df['tmdbId'] = df['tmdbId'].astype(int)
            self.metadata_map = df.set_index('tmdbId').to_dict('index')
        else:
            print("Warning: tmdbId column not found in metadata")
            self.metadata_map = {}
        
        print(f"  Metadata loaded: {len(self.metadata_map):,} movies")

    def recommend(
        self,
        user_movie_ids: List[int],
        top_k: int = 20,
        exclude_seen: bool = True
    ) -> List[dict]:
        """
        하이브리드 추천 (필터링 없음)
        
        Args:
            user_movie_ids: 사용자가 본 영화 tmdbId 리스트
            top_k: 추천할 영화 개수
            exclude_seen: 본 영화 제외 여부
        
        Returns:
            추천 영화 리스트 (tmdbId, 점수, 메타데이터 포함)
        """
        print(f"\nStarting hybrid recommendation...")
        print(f"  User movies: {len(user_movie_ids)}")
        print(f"  Top-K: {top_k}")
        
        start_time = time.time()
        
        # 1. 사용자 프로필 생성 (SBERT)
        user_sbert_vecs = []
        for mid in user_movie_ids:
            if mid in self.sbert_movie_to_idx:
                user_sbert_vecs.append(
                    self.sbert_embeddings[self.sbert_movie_to_idx[mid]]
                )
        
        if not user_sbert_vecs:
            print("Warning: No valid SBERT movies found")
            return []
        
        user_sbert_profile = np.mean(user_sbert_vecs, axis=0)
        user_sbert_profile = user_sbert_profile / (np.linalg.norm(user_sbert_profile) + 1e-10)
        
        # 2. 사용자 프로필 생성 (LightGCN)
        user_gcn_vecs = []
        for mid in user_movie_ids:
            if mid in self.lightgcn_movie_to_idx:
                user_gcn_vecs.append(
                    self.lightgcn_item_embeddings[self.lightgcn_movie_to_idx[mid]]
                )
        
        if not user_gcn_vecs:
            print("Warning: No valid LightGCN movies found")
            return []
        
        user_gcn_profile = np.mean(user_gcn_vecs, axis=0)
        
        # 3. 전체 점수 계산 (Pre-aligned Matrix 사용)
        sbert_scores = self.target_sbert_norm @ user_sbert_profile
        lightgcn_scores = self.target_lightgcn_matrix @ user_gcn_profile
        
        # 4. 정규화
        norm_sbert = self.scaler.fit_transform(sbert_scores.reshape(-1, 1)).squeeze()
        norm_lightgcn = self.scaler.fit_transform(lightgcn_scores.reshape(-1, 1)).squeeze()
        
        # 5. 하이브리드 점수
        final_scores = self.sbert_weight * norm_sbert + self.lightgcn_weight * norm_lightgcn
        
        # 6. 본 영화 제외
        if exclude_seen:
            for i, mid in enumerate(self.common_movie_ids):
                if mid in user_movie_ids:
                    final_scores[i] = -np.inf
        
        # 7. Top-K 추출
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # 8. 결과 생성
        recommendations = []
        for idx in top_indices:
            mid = self.common_movie_ids[idx]
            meta = self.metadata_map.get(mid, {})
            
            recommendations.append({
                'tmdbId': mid,
                'hybrid_score': float(final_scores[idx]),
                'sbert_score': float(norm_sbert[idx]),
                'lightgcn_score': float(norm_lightgcn[idx]),
                'title': meta.get('title', 'Unknown'),
                'overview': meta.get('overview', '')
            })
        
        elapsed = time.time() - start_time
        print(f"Recommendation completed in {elapsed:.2f}s")
        
        return recommendations


# ============================================================
# 실행 예시
# ============================================================
if __name__ == "__main__":
    # 경로 설정
    SBERT_EMBEDDINGS_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl"
    LIGHTGCN_MODEL_PATH = "/home/ubuntu/ai-model/models/light_gcn/checkpoints/best_model.pt"
    LIGHTGCN_DATA_PATH = "/home/ubuntu/ai-model/models/light_gcn/data"
    METADATA_PATH = "/home/ubuntu/ai-model/models/cbf/v2/data/pre_final_movies_processed.csv"
    
    # 추천 시스템 초기화
    print("\n" + "="*80)
    print("INITIALIZING SIMPLE HYBRID RECOMMENDER")
    print("="*80)
    
    recommender = SimpleHybridRecommender(
        sbert_embeddings_path=SBERT_EMBEDDINGS_PATH,
        lightgcn_model_path=LIGHTGCN_MODEL_PATH,
        lightgcn_data_path=LIGHTGCN_DATA_PATH,
        metadata_path=METADATA_PATH,
        sbert_weight=0.7,
        lightgcn_weight=0.3
    )
    
    print("\n" + "="*80)
    print("INITIALIZATION COMPLETE")
    print("="*80)
    
    # 테스트: 토이 스토리(862) 좋아하는 사용자
    user_movies = [862, 8844, 15602]  # 토이 스토리, 쥬라기 공원, 포레스트 검프
    
    print("\n" + "="*80)
    print("USER INPUT MOVIES")
    print("="*80)
    for mid in user_movies:
        meta = recommender.metadata_map.get(mid, {})
        print(f"  {mid}: {meta.get('title', 'Unknown')}")
    
    # 추천 실행
    recommendations = recommender.recommend(
        user_movie_ids=user_movies,
        top_k=10,
        exclude_seen=True
    )
    
    # 결과 출력
    print("\n" + "="*120)
    print("RECOMMENDATIONS")
    print("="*120)
    print(f"{'Rank':<4} | {'tmdbId':<8} | {'Score':<8} | {'SBERT':<8} | {'LightGCN':<8} | {'Title':<30} | {'Overview'}")
    print("-" * 120)

    for i, rec in enumerate(recommendations, 1):
        title = rec['title']
        if len(title) > 28:
            title = title[:25] + "..."
        
        overview = rec['overview']
        if len(overview) > 50:
            overview = overview[:47] + "..."
        
        print(f"{i:<4} | {rec['tmdbId']:<8} | {rec['hybrid_score']:.4f} | "
            f"{rec['sbert_score']:.4f} | {rec['lightgcn_score']:.4f} | {title:<30} | {overview}")

    print("\n" + "="*120)