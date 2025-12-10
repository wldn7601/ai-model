# 데이터 확인용 스크립트
import pandas as pd

# 경로
BASE_PATH = '/home/ubuntu/ai-model/movielens_data'

# 1. TagDL 확인
tagdl = pd.read_csv(f'{BASE_PATH}/tagdl.csv')
print("TagDL 구조:")
print(tagdl.head())
print(f"- 고유 태그 수: {tagdl['tag'].nunique()}")
print(f"- 고유 영화 수: {tagdl['item_id'].nunique()}")

# 2. Ratings 확인
ratings = pd.read_csv(f'{BASE_PATH}/ratings.csv')
print("\nRatings 구조:")
print(f"- 전체 평점: {len(ratings):,}개")
print(f"- 고유 사용자: {ratings['userId'].nunique():,}명")
print(f"- 고유 영화: {ratings['movieId'].nunique():,}편")

# 3. Metadata 확인
movies = pd.read_csv(f'{BASE_PATH}/movies_metadata_restored.csv')
print("\nMetadata 구조:")
print(movies.head())
print(f"- 전체 영화: {len(movies):,}편")