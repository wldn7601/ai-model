import pandas as pd
import json
import os
from tqdm import tqdm

SOURCE_ROOT = '/home/ubuntu/ai-model/datas'
ARCHIVE_DIR = os.path.join(SOURCE_ROOT, 'archive')
TARGET_DIR = '/home/ubuntu/ai-model/models/cbf/v1_basic/data'

PATH_JSON = os.path.join(SOURCE_ROOT, '2019_data.json')
PATH_GENOME_SCORES = os.path.join(ARCHIVE_DIR, 'genome-scores.csv')
PATH_GENOME_TAGS = os.path.join(ARCHIVE_DIR, 'genome-tags.csv')
OUTPUT_JSON = os.path.join(TARGET_DIR, '2019_data_with_tags.json')

RELEVANCE_THRESHOLD = 0.20
TOP_K_TAGS = 20

def merge_genome_tags():
    print("🚀 [Genome Tag 통합] 시작...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 1. 태그 매핑 로드
    print(">>> 1. genome-tags.csv 로드...")
    df_tags = pd.read_csv(PATH_GENOME_TAGS)
    tag_map = dict(zip(df_tags['tagId'], df_tags['tag']))
    print(f"   ✓ 태그 정의 {len(tag_map):,}개")
    
    # 2. 2019 영화 데이터 로드
    print(">>> 2. 2019_data.json 로드...")
    with open(PATH_JSON, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    
    target_ids = set([m['movieId'] for m in movies if 'movieId' in m])
    print(f"   ✓ 대상 영화 {len(target_ids):,}개")
    
    # 3. Genome scores 로드 및 필터링
    print(">>> 3. genome-scores.csv 처리 중...")
    df_scores = pd.read_csv(PATH_GENOME_SCORES)
    
    df_scores = df_scores[
        (df_scores['movieId'].isin(target_ids)) &
        (df_scores['relevance'] >= RELEVANCE_THRESHOLD)
    ]
    print(f"   ✓ 필터링 후 {len(df_scores):,}개 행")
    
    # 4. 영화별 Top-K 태그 추출
    print(">>> 4. 영화별 Top 20 태그 선별...")
    top_tags = (
        df_scores.sort_values(['movieId', 'relevance'], ascending=[True, False])
        .groupby('movieId')
        .head(TOP_K_TAGS)
    )
    
    top_tags['tag'] = top_tags['tagId'].map(tag_map)
    
    # 딕셔너리 변환 (score 필드명 유지)
    tag_dict = {}
    for mid, group in tqdm(top_tags.groupby('movieId'), desc="태그 구조화"):
        records = group[['tag', 'relevance']].rename(
            columns={'relevance': 'score'}
        ).to_dict('records')
        tag_dict[mid] = records
    
    print(f"   ✓ 태그 딕셔너리 생성 완료 ({len(tag_dict):,}개 영화)")
    
    # 5. JSON 병합
    print(">>> 5. 데이터 병합 및 저장...")
    tagged_count = 0
    
    for movie in tqdm(movies, desc="병합 중"):
        mid = movie.get('movieId')
        
        if mid in tag_dict:
            # genome 출처 명시 (trend_data의 predicted_tags와 구분)
            movie['genome_tags'] = tag_dict[mid]
            tagged_count += 1
        else:
            movie['genome_tags'] = []
    
    # 저장
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 완료!")
    print(f"   ✓ 출력: {OUTPUT_JSON}")
    print(f"   ✓ 태그 추가: {tagged_count:,}/{len(movies):,} "
          f"({tagged_count/len(movies)*100:.1f}%)")
    print(f"   ✓ 필드명: genome_tags (MovieLens 사용자 평가 기반)")
    print(f"   ✓ Threshold: {RELEVANCE_THRESHOLD}")

if __name__ == "__main__":
    merge_genome_tags()