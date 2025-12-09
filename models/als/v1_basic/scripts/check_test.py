# scripts/check_test.py
import pandas as pd

test = pd.read_csv('../data/test.csv')

print("Test Rating distribution:")
print(test['rating'].value_counts().sort_index())

print("\nTest Threshold별 데이터 비율:")
for threshold in [3.0, 3.5, 4.0, 4.5]:
    ratio = (test['rating'] >= threshold).sum() / len(test)
    print(f"rating >= {threshold}: {ratio*100:.1f}%")