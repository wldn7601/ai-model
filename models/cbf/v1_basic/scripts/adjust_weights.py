"""
재임베딩 없이 가중치만 조정
작업 위치: ~/ai-model/models/cbf/v1_basic/scripts/
"""

import pickle
import numpy as np


def recombine_embeddings(
    embeddings_path: str,
    weight_genre: float = 0.2,
    weight_tag: float = 0.7,
    weight_overview: float = 0.1,
    output_path: str = None
):
    """
    저장된 개별 임베딩으로 가중치만 재조정
    
    재임베딩 없이 즉시 테스트 가능!
    """
    print(">>> 가중치 재조정 중...")
    print(f"   장르: {weight_genre}")
    print(f"   태그: {weight_tag}")
    print(f"   Overview: {weight_overview}")
    
    # 로드
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    genre_embs = data['genre_embeddings']
    tag_embs = data['tag_embeddings']
    overview_embs = data['overview_embeddings']
    
    # 재결합
    combined = (
        genre_embs * weight_genre +
        tag_embs * weight_tag +
        overview_embs * weight_overview
    )
    
    # 정규화
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / norms
    
    # 유사도 체크
    sample_size = 500
    similarities = []
    for _ in range(sample_size):
        i, j = np.random.choice(len(combined), 2, replace=False)
        sim = np.dot(combined[i], combined[j])
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    
    print(f"\n   ✓ 평균 유사도: {avg_sim:.4f}")
    
    if avg_sim < 0.4:
        print(f"   ✅ 우수")
    elif avg_sim < 0.6:
        print(f"   ✅ 양호")
    else:
        print(f"   ⚠️  개선 필요")
    
    # 저장
    if output_path:
        data['embeddings'] = combined
        data['config']['weight_genre'] = weight_genre
        data['config']['weight_tag'] = weight_tag
        data['config']['weight_overview'] = weight_overview
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        
        print(f"\n   ✓ 저장: {output_path}")
    
    return combined, avg_sim


def test_different_weights():
    """
    다양한 가중치 조합 테스트
    """
    print("="*70)
    print(" 가중치 조합 테스트")
    print("="*70)
    
    embeddings_path = '../outputs/movie_embeddings_separate_e5.pkl'
    
    # 테스트할 가중치 조합
    weight_combinations = [
        {'genre': 0.2, 'tag': 0.7, 'overview': 0.1, 'name': '태그 중심'},
        {'genre': 0.4, 'tag': 0.5, 'overview': 0.1, 'name': '균형형'},
        {'genre': 0.5, 'tag': 0.4, 'overview': 0.1, 'name': '장르 중심'},
        {'genre': 0.3, 'tag': 0.7, 'overview': 0.0, 'name': '태그만'},
        {'genre': 0.5, 'tag': 0.5, 'overview': 0.0, 'name': '장르+태그'},
    ]
    
    results = []
    
    for combo in weight_combinations:
        print(f"\n>>> {combo['name']}:")
        _, avg_sim = recombine_embeddings(
            embeddings_path,
            weight_genre=combo['genre'],
            weight_tag=combo['tag'],
            weight_overview=combo['overview']
        )
        
        results.append({
            'name': combo['name'],
            'genre': combo['genre'],
            'tag': combo['tag'],
            'overview': combo['overview'],
            'avg_sim': avg_sim
        })
    
    # 결과 요약
    print("\n\n" + "="*70)
    print(" 결과 요약")
    print("="*70)
    print(f"{'조합':<12} {'장르':<6} {'태그':<6} {'Over':<6} {'평균유사도':<10} {'평가'}")
    print("-"*70)
    
    for r in results:
        evaluation = "우수" if r['avg_sim'] < 0.4 else "양호" if r['avg_sim'] < 0.6 else "개선필요"
        print(f"{r['name']:<12} {r['genre']:<6.1f} {r['tag']:<6.1f} {r['overview']:<6.1f} {r['avg_sim']:<10.4f} {evaluation}")
    
    # 최적 조합 추천
    best = min(results, key=lambda x: x['avg_sim'])
    print(f"\n✅ 추천 조합: {best['name']}")
    print(f"   장르: {best['genre']}, 태그: {best['tag']}, Overview: {best['overview']}")
    print(f"   평균 유사도: {best['avg_sim']:.4f}")


if __name__ == "__main__":
    test_different_weights()