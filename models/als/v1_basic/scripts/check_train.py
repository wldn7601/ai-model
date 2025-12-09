# 현재 train.csv의 평점 분포 확인
import pandas as pd

train = pd.read_csv('../data/train.csv')

print("Rating distribution:")
print(train['rating'].value_counts().sort_index())

print("\nThreshold별 데이터 비율:")
for threshold in [3.0, 3.5, 4.0, 4.5]:
    ratio = (train['rating'] >= threshold).sum() / len(train)
    print(f"rating >= {threshold}: {ratio*100:.1f}% ({(train['rating'] >= threshold).sum():,} ratings)")