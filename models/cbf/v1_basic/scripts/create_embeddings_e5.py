"""
e5-large 영화 임베딩 생성 (SBERT 최적화 버전)
작업 위치: ~/ai-model/models/cbf/v1_basic/scripts/create_embeddings_e5.py
"""

"""
# 장르+태그+overview 전체로 임베딩

movie_embeddings_e5
모델 생성
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
OUTPUT_PKL = os.path.join(OUTPUT_DIR, 'movie_embeddings_e5.pkl')

# ==========================================
# 설정
# ==========================================
MODEL_NAME = 'intfloat/multilingual-e5-large'
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 태그 필터링
TAG_MIN_SCORE = 0.25  # 0.25 이상만 (고품질 태그만)
TOP_K_TAGS = 15       # 상위 15개만


def create_movie_text(movie: dict) -> str:
    """
    영화 데이터를 SBERT에 최적화된 자연어 텍스트로 변환
    
    사용 데이터:
    1. genres (장르)
    2. genome_tags (상위 15개, score >= 0.25)
    3. overview (줄거리)
    """
    # 1. 장르
    genres = ", ".join([g['name'] for g in movie.get('genres', [])])
    
    # 2. 태그 (고품질만 선택)
    tags = [
        tag_obj['tag'] 
        for tag_obj in movie.get('genome_tags', [])
        if tag_obj.get('score', 0) >= TAG_MIN_SCORE
    ][:TOP_K_TAGS]
    
    tag_text = ", ".join(tags)
    
    # 3. Overview
    overview = movie.get('overview', '')
    
    # 4. 자연어 형식으로 결합 (SBERT가 문맥을 더 잘 이해)
    if tag_text:
        combined = f"장르: {genres}. 특징: {tag_text}. 줄거리: {overview}"
    else:
        combined = f"장르: {genres}. 줄거리: {overview}"
    
    return combined.strip()


def load_and_prepare_data():
    """
    영화 데이터 로드 및 텍스트 생성
    """
    print(">>> 1. 영화 데이터 로드 중...")
    
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    
    print(f"   ✓ 총 {len(movies):,}개 영화 로드")
    
    # 텍스트 생성
    print(">>> 2. 텍스트 생성 중...")
    texts = []
    movie_ids = []
    metadata = []
    
    tag_count_stats = []
    
    for movie in tqdm(movies, desc="텍스트 변환"):
        # 영화 텍스트 생성
        text = create_movie_text(movie)
        
        # e5 프롬프트 추가 (passage:)
        text_with_prompt = f"passage: {text}"
        
        texts.append(text_with_prompt)
        movie_ids.append(movie['movieId'])
        
        # 통계
        tag_count = len([
            t for t in movie.get('genome_tags', [])
            if t.get('score', 0) >= TAG_MIN_SCORE
        ])
        tag_count_stats.append(tag_count)
        
        # 메타데이터 저장 (필터링 및 추천용)
        metadata.append({
            'movieId': movie['movieId'],
            'title': movie.get('title', ''),
            'runtime': movie.get('runtime', 0),
            'genres': [g['name'] for g in movie.get('genres', [])],
            'providers': [p['provider_name'] for p in movie.get('providers', [])],
            'vote_average': movie.get('vote_average', 0),
            'popularity': movie.get('popularity', 0)
        })
    
    # 통계 출력
    print(f"   ✓ 텍스트 생성 완료")
    print(f"   ✓ 평균 텍스트 길이: {sum(len(t) for t in texts) / len(texts):.0f}자")
    print(f"   ✓ 평균 태그 수: {sum(tag_count_stats) / len(tag_count_stats):.1f}개")
    print(f"   ✓ 태그 있는 영화: {sum(1 for c in tag_count_stats if c > 0):,}개")
    
    return texts, movie_ids, metadata


def create_embeddings(texts):
    """
    e5-large로 임베딩 생성
    """
    print(f"\n>>> 3. e5-large 모델 로드 중...")
    print(f"   - 모델: {MODEL_NAME}")
    print(f"   - 디바이스: {DEVICE}")
    
    if DEVICE == 'cuda':
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   - 예상 VRAM 사용: ~6GB")
    
    # 모델 로드
    model = SentenceTransformer(MODEL_NAME)
    model.to(DEVICE)
    
    print(f"   ✓ 모델 로드 완료")
    
    # GPU 메모리 초기화
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f"   ✓ GPU 메모리 초기화")
    
    # 임베딩 생성
    print(f"\n>>> 4. 임베딩 생성 중...")
    print(f"   - 배치 크기: {BATCH_SIZE}")
    
    if DEVICE == 'cuda':
        print(f"   - 예상 시간: ~3분")
    else:
        print(f"   - 예상 시간: ~15분 (CPU 느림 주의!)")
    
    start_time = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=DEVICE,
        normalize_embeddings=True  # 코사인 유사도 최적화
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n   ✓ 임베딩 생성 완료!")
    print(f"   ✓ 소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    print(f"   ✓ Shape: {embeddings.shape}")
    print(f"   ✓ Dtype: {embeddings.dtype}")
    print(f"   ✓ 메모리: {embeddings.nbytes / 1024 / 1024:.1f}MB")
    
    # GPU 메모리 정리
    if DEVICE == 'cuda':
        del model
        torch.cuda.empty_cache()
    
    return embeddings


def save_embeddings(embeddings, movie_ids, metadata):
    """
    임베딩 및 메타데이터 저장
    """
    print(f"\n>>> 5. 저장 중...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_data = {
        'embeddings': embeddings,
        'movie_ids': movie_ids,
        'metadata': metadata,
        'model_name': MODEL_NAME,
        'embedding_dim': embeddings.shape[1],
        'num_movies': len(movie_ids),
        'config': {
            'batch_size': BATCH_SIZE,
            'tag_min_score': TAG_MIN_SCORE,
            'top_k_tags': TOP_K_TAGS,
            'normalized': True,
            'device': DEVICE
        }
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f, protocol=4)
    
    file_size_mb = os.path.getsize(OUTPUT_PKL) / 1024 / 1024
    
    print(f"   ✓ 저장 완료: {OUTPUT_PKL}")
    print(f"   ✓ 파일 크기: {file_size_mb:.1f}MB")


def verify_embeddings():
    """
    임베딩 검증
    """
    print(f"\n>>> 6. 검증 중...")
    
    with open(OUTPUT_PKL, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    movie_ids = data['movie_ids']
    
    # 기본 검증
    assert len(embeddings) == len(movie_ids), "길이 불일치"
    assert embeddings.shape[1] == 1024, "차원 오류"
    
    # 정규화 확인 (코사인 유사도를 위해 중요)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "정규화 오류"
    
    # 샘플 유사도 계산
    sample_sims = []
    for i in range(min(5, len(embeddings))):
        for j in range(i+1, min(5, len(embeddings))):
            sim = np.dot(embeddings[i], embeddings[j])
            sample_sims.append(sim)
    
    avg_sim = np.mean(sample_sims)
    
    print(f"   ✓ 길이 검증: {len(embeddings):,}개")
    print(f"   ✓ 차원 검증: {embeddings.shape[1]}D")
    print(f"   ✓ 정규화 검증: OK (norm ≈ 1.0)")
    print(f"   ✓ 평균 유사도: {avg_sim:.4f}")
    
    # 상위 5개 영화 정보
    print(f"\n   영화 샘플 (처음 5개):")
    for i in range(min(5, len(movie_ids))):
        meta = data['metadata'][i]
        print(f"     {i+1}. {meta['title']} (ID: {meta['movieId']})")
        print(f"        장르: {', '.join(meta['genres'])}")


def main():
    """
    메인 실행 함수
    """
    print("="*70)
    print(" e5-large 영화 임베딩 생성 (SBERT 최적화)")
    print("="*70)
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  GPU 없음 - CPU 사용 (약 5배 느림)")
    
    print()
    
    try:
        # 1. 데이터 준비
        texts, movie_ids, metadata = load_and_prepare_data()
        
        # 2. 임베딩 생성
        embeddings = create_embeddings(texts)
        
        # 3. 저장
        save_embeddings(embeddings, movie_ids, metadata)
        
        # 4. 검증
        verify_embeddings()
        
        print("\n" + "="*70)
        print("🎉 완료!")
        print("="*70)
        print(f"✓ 임베딩 파일: {OUTPUT_PKL}")
        print(f"✓ 영화 수: {len(movie_ids):,}개")
        print(f"✓ 차원: {embeddings.shape[1]}D (e5-large)")
        print(f"\n다음 단계:")
        print("  python recommend_e5.py  # 추천 시스템 테스트")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()