import pandas as pd
import os

# 경로 설정
DATA_DIR = '/home/ubuntu/ai-model/models/light_gcn/data'

# 1. 분할된 데이터 로드
print("Loading split data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_ratings.csv'))

# 2. Unseen Item 비율 계산
train_items = set(train_df['tmdbId'].unique())  # tmdb_id → tmdbId
test_items = set(test_df['tmdbId'].unique())    # tmdb_id → tmdbId

# 학습에는 없는데 테스트에는 있는 영화들 (차집합)
unseen_items = test_items - train_items

# 비율 계산
ratio = len(unseen_items) / len(test_items) * 100

print(f"\n=== Cold Start Item Analysis ===")
print(f"Train items: {len(train_items):,}")
print(f"Test items:  {len(test_items):,}")
print(f"Unseen items (New in Test): {len(unseen_items):,}")
print(f"Unseen Ratio: {ratio:.2f}%")

# 3. 진단 가이드
print("-" * 40)
if ratio > 20:
    print("🚨 [위험] 비율이 매우 높습니다! (20% 이상)")
    print("   Test 셋 정답의 상당수가 '학습 불가능한 영화'입니다.")
    print("   -> 해결책: Temporal Split 대신 Random Split 사용 권장")
elif ratio > 5:
    print("⚠️ [주의] 비율이 다소 높습니다. (5~20%)")
    print("   Recall 성능의 상한선이 존재할 수 있습니다.")
else:
    print("✅ [양호] 비율이 낮습니다. (5% 미만)")
    print("   Cold Start 문제는 성능 정체의 주원인이 아닐 가능성이 높습니다.")