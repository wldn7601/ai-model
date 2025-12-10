"""
개별 임베딩 후 가중 평균 방식
작업 위치: ~/ai-model/models/cbf/v1_basic/scripts/
"""

"""
# 태그
# 장르
# overview
# 각각 임베딩
movie_embeddings_separate_e5 모델 생성
"""

import json
import numpy as np
import pickle
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os

# ==========================================
# 경로 설정
# ==========================================
BASE_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

INPUT_JSON = os.path.join(DATA_DIR, '2019_data_with_tags.json')
OUTPUT_PKL = os.path.join(OUTPUT_DIR, 'movie_embeddings_separate_e5.pkl')

# ==========================================
# 설정
# ==========================================
MODEL_NAME = 'intfloat/multilingual-e5-large'
# BATCH_SIZE = 16
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 가중치 설정 (조정 가능)
WEIGHT_GENRE = 0.2      # 장르 20%
WEIGHT_TAG = 0.7        # 태그 70%
WEIGHT_OVERVIEW = 0.1   # overview 10%

# 태그 설정
TAG_MIN_SCORE = 0.20
TOP_K_TAGS = 25


def load_movies():
    """
    영화 데이터 로드
    """
    print(">>> 1. 영화 데이터 로드 중...")
    
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    
    print(f"   ✓ 총 {len(movies):,}개 영화 로드")
    
    return movies


def prepare_texts_separately(movies):
    """
    장르, 태그, overview를 개별로 준비
    """
    print(">>> 2. 텍스트 생성 중...")
    
    genre_texts = []
    tag_texts = []
    overview_texts = []
    movie_ids = []
    metadata = []
    
    for movie in tqdm(movies, desc="텍스트 변환"):
        # 1. 장르 텍스트
        genres = ", ".join([g['name'] for g in movie.get('genres', [])])
        if not genres:
            genres = "기타"
        genre_text = f"passage: 장르 {genres}"
        
        # 2. 태그 텍스트
        tags = [
            tag_obj['tag'] 
            for tag_obj in movie.get('genome_tags', [])
            if tag_obj.get('score', 0) >= TAG_MIN_SCORE
        ][:TOP_K_TAGS]
        
        if tags:
            tag_text = f"passage: 특징 {', '.join(tags)}"
        else:
            tag_text = f"passage: 일반 영화"
        
        # 3. Overview 텍스트 (축약)
        overview = movie.get('overview', '')[:300]  # 300자로 제한
        if not overview:
            overview = "영화"
        overview_text = f"passage: {overview}"
        
        genre_texts.append(genre_text)
        tag_texts.append(tag_text)
        overview_texts.append(overview_text)
        movie_ids.append(movie['movieId'])
        
        # 메타데이터
        metadata.append({
            'movieId': movie['movieId'],
            'title': movie.get('title', ''),
            'runtime': movie.get('runtime', 0),
            'genres': [g['name'] for g in movie.get('genres', [])],
            'providers': [p['provider_name'] for p in movie.get('providers', [])],
            'vote_average': movie.get('vote_average', 0),
            'popularity': movie.get('popularity', 0)
        })
    
    print(f"   ✓ 텍스트 생성 완료")
    print(f"   ✓ 평균 장르 길이: {sum(len(t) for t in genre_texts) / len(genre_texts):.0f}자")
    print(f"   ✓ 평균 태그 길이: {sum(len(t) for t in tag_texts) / len(tag_texts):.0f}자")
    print(f"   ✓ 평균 overview 길이: {sum(len(t) for t in overview_texts) / len(overview_texts):.0f}자")
    
    return genre_texts, tag_texts, overview_texts, movie_ids, metadata


def create_embeddings_separately(genre_texts, tag_texts, overview_texts):
    """
    개별 임베딩 생성
    """
    print(f"\n>>> 3. 모델 로드 중...")
    model = SentenceTransformer(MODEL_NAME)
    model.to(DEVICE)
    
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"   ✓ 모델 로드 완료")
    
    # 1. 장르 임베딩
    print(f"\n>>> 4-1. 장르 임베딩 생성 중...")
    start = time.time()
    
    genre_embeddings = model.encode(
        genre_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=DEVICE,
        normalize_embeddings=True
    )
    
    print(f"   ✓ 소요 시간: {time.time() - start:.1f}초")
    
    # 2. 태그 임베딩
    print(f"\n>>> 4-2. 태그 임베딩 생성 중...")
    start = time.time()
    
    tag_embeddings = model.encode(
        tag_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=DEVICE,
        normalize_embeddings=True
    )
    
    print(f"   ✓ 소요 시간: {time.time() - start:.1f}초")
    
    # 3. Overview 임베딩
    print(f"\n>>> 4-3. Overview 임베딩 생성 중...")
    start = time.time()
    
    overview_embeddings = model.encode(
        overview_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=DEVICE,
        normalize_embeddings=True
    )
    
    print(f"   ✓ 소요 시간: {time.time() - start:.1f}초")
    
    # GPU 메모리 정리
    if DEVICE == 'cuda':
        del model
        torch.cuda.empty_cache()
    
    return genre_embeddings, tag_embeddings, overview_embeddings


def combine_embeddings(genre_embs, tag_embs, overview_embs):
    """
    가중 평균으로 결합
    """
    print(f"\n>>> 5. 임베딩 결합 중...")
    print(f"   - 장르 가중치: {WEIGHT_GENRE}")
    print(f"   - 태그 가중치: {WEIGHT_TAG}")
    print(f"   - Overview 가중치: {WEIGHT_OVERVIEW}")
    
    combined = (
        genre_embs * WEIGHT_GENRE +
        tag_embs * WEIGHT_TAG +
        overview_embs * WEIGHT_OVERVIEW
    )
    
    # L2 정규화
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / norms
    
    print(f"   ✓ 결합 완료")
    print(f"   ✓ Shape: {combined.shape}")
    
    return combined


def save_embeddings(combined_embs, genre_embs, tag_embs, overview_embs, 
                   movie_ids, metadata):
    """
    모든 임베딩 저장
    """
    print(f"\n>>> 6. 저장 중...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_data = {
        # 최종 결합 임베딩
        'embeddings': combined_embs,
        
        # 개별 임베딩 (가중치 재조정용)
        'genre_embeddings': genre_embs,
        'tag_embeddings': tag_embs,
        'overview_embeddings': overview_embs,
        
        # 메타데이터
        'movie_ids': movie_ids,
        'metadata': metadata,
        'model_name': MODEL_NAME,
        'embedding_dim': combined_embs.shape[1],
        'num_movies': len(movie_ids),
        
        # 설정
        'config': {
            'batch_size': BATCH_SIZE,
            'tag_min_score': TAG_MIN_SCORE,
            'top_k_tags': TOP_K_TAGS,
            'weight_genre': WEIGHT_GENRE,
            'weight_tag': WEIGHT_TAG,
            'weight_overview': WEIGHT_OVERVIEW,
            'normalized': True,
            'device': DEVICE,
            'method': 'separate_embeddings'
        }
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f, protocol=4)
    
    file_size_mb = os.path.getsize(OUTPUT_PKL) / 1024 / 1024
    
    print(f"   ✓ 저장 완료: {OUTPUT_PKL}")
    print(f"   ✓ 파일 크기: {file_size_mb:.1f}MB (개별 임베딩 포함)")


def verify_embeddings():
    """
    임베딩 검증
    """
    print(f"\n>>> 7. 검증 중...")
    
    with open(OUTPUT_PKL, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    
    # 정규화 확인
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "정규화 오류"
    
    # 유사도 분포
    sample_size = 500
    similarities = []
    for _ in range(sample_size):
        i, j = np.random.choice(len(embeddings), 2, replace=False)
        sim = np.dot(embeddings[i], embeddings[j])
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    
    print(f"   ✓ 길이: {len(embeddings):,}개")
    print(f"   ✓ 차원: {embeddings.shape[1]}D")
    print(f"   ✓ 정규화: OK")
    print(f"   ✓ 평균 유사도: {avg_sim:.4f}")
    
    # 평가
    if avg_sim < 0.4:
        print(f"   ✅ 우수 - 영화들 잘 구분됨")
    elif avg_sim < 0.6:
        print(f"   ✅ 양호 - 적당한 다양성")
    else:
        print(f"   ⚠️  개선 필요 - 아직 너무 비슷")


def main():
    """
    메인 실행
    """
    print("="*70)
    print(" 개별 임베딩 + 가중 평균 방식")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CPU 사용 (느림)")
    
    print()
    
    total_start = time.time()
    
    try:
        # 1. 데이터 로드
        movies = load_movies()
        
        # 2. 텍스트 준비
        genre_texts, tag_texts, overview_texts, movie_ids, metadata = \
            prepare_texts_separately(movies)
        
        # 3. 개별 임베딩 (3번)
        genre_embs, tag_embs, overview_embs = \
            create_embeddings_separately(genre_texts, tag_texts, overview_texts)
        
        # 4. 가중 평균
        combined_embs = combine_embeddings(genre_embs, tag_embs, overview_embs)
        
        # 5. 저장
        save_embeddings(combined_embs, genre_embs, tag_embs, overview_embs,
                       movie_ids, metadata)
        
        # 6. 검증
        verify_embeddings()
        
        total_time = time.time() - total_start
        
        print("\n" + "="*70)
        print("🎉 완료!")
        print("="*70)
        print(f"✓ 총 소요 시간: {total_time/60:.1f}분")
        print(f"✓ 임베딩 파일: {OUTPUT_PKL}")
        print(f"\n장점:")
        print(f"  • 정확한 가중치 제어")
        print(f"  • 재임베딩 없이 가중치 조정 가능")
        print(f"  • 더 나은 차별화")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()