import pandas as pd
import urllib.parse
import sys

# 1. 파일 경로 설정
ott_data_path = '/home/ubuntu/ai-model/datas/data/movie_ott_ids.csv'
movie_data_path = '/home/ubuntu/ai-model/datas/data/insert_movies_updated.csv'
output_path = '/home/ubuntu/ai-model/datas/data/movie_ott_final.csv'

# 2. 데이터 로드
print(f"Loading files...\n - OTT: {ott_data_path}\n - Movie: {movie_data_path}")
df_ott = pd.read_csv(ott_data_path)
df_movies = pd.read_csv(movie_data_path)

# 3. [핵심 수정 1] 영화 데이터에서 'tmdb_id'를 'tmdbId'로 이름 변경
if 'tmdb_id' in df_movies.columns:
    df_movies.rename(columns={'tmdb_id': 'tmdbId'}, inplace=True)
    print("Renamed column: 'tmdb_id' -> 'tmdbId'")
else:
    print("Warning: 'tmdb_id' column not found. (Maybe already named tmdbId?)")

# 4. [핵심 수정 2] 'movie_id' 기준으로 병합하기 위해 데이터 타입 통일
# 두 파일 모두 'movie_id' 컬럼이 존재하므로 이를 정수형으로 변환
df_ott['movie_id'] = pd.to_numeric(df_ott['movie_id'], errors='coerce')
df_movies['movie_id'] = pd.to_numeric(df_movies['movie_id'], errors='coerce')

# 5. 데이터 병합 (Merge)
# 기준: movie_id == movie_id
print("Merging data on 'movie_id'...")
df_merged = pd.merge(
    df_ott.dropna(subset=['movie_id']), # OTT 데이터에서 ID 없는 건 제외
    df_movies[['movie_id', 'tmdbId', 'title']], # 영화 데이터에서 필요한 컬럼만 가져옴
    on='movie_id', # [중요] movie_id를 기준으로 병합
    how='left'
)

# 매칭 결과 확인
missing_count = df_merged['title'].isna().sum()
print(f"Total OTT Rows: {len(df_merged)}")
print(f"Title Match Failed (Empty Title): {missing_count}")

# 6. OTT 이름 매핑
provider_map = {
    8: 'Netflix', 337: 'Disney+', 356: 'Wavve', 97: 'Watcha', 
    350: 'Apple TV', 3: 'Google Play', 1883: 'TVING'
}
df_merged['ott_name'] = df_merged['provider_id'].map(provider_map)

# 7. URL 생성 로직
def get_ott_url(row):
    provider_id = row['provider_id']
    title = row['title']
    
    # 제목이 비어있으면 URL 생성 불가
    if pd.isna(title):
        return None
        
    encoded_title = urllib.parse.quote(str(title))
    
    if provider_id == 8: return f"https://www.netflix.com/search?q={encoded_title}"
    elif provider_id == 337: return "https://www.disneyplus.com/"
    elif provider_id == 356: return f"https://www.wavve.com/search?searchWord={encoded_title}"
    elif provider_id == 97: return f"https://watcha.com/search?query={encoded_title}"
    elif provider_id == 350: return f"https://tv.apple.com/kr/search?term={encoded_title}"
    elif provider_id == 3: return f"https://play.google.com/store/search?q={encoded_title}&c=movies"
    elif provider_id == 1883: return f"https://www.tving.com/search/main?keyword={encoded_title}"
    return None

print("Generating URLs...")
df_merged['ott_url'] = df_merged.apply(get_ott_url, axis=1)

# 8. 최종 결과 저장
# 컬럼 순서 정리: movie_id, tmdbId, title, ott_name, ott_url, provider_id
final_cols = ['movie_id', 'tmdbId', 'title', 'ott_name', 'ott_url', 'provider_id']
final_df = df_merged[final_cols]

final_df.to_csv(output_path, index=False)
print(f"Saved successfully to: {output_path}")
print("Final Columns:", final_df.columns.tolist())