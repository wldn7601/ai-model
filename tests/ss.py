import pandas as pd
import pickle
import numpy as np

# 파일 경로 설정
file_path = '/home/ubuntu/ai-model/models/cbf/v2/data/movies_with_embeddings.pkl'

print(f"Checking file: {file_path}")
print("-" * 60)

try:
    # 1. Pandas로 로드 시도 (가장 확률 높음)
    data = pd.read_pickle(file_path)
    
    if isinstance(data, pd.DataFrame):
        print("✅ Data Type: Pandas DataFrame")
        print(f"✅ Shape: {data.shape} (Rows, Columns)")
        print("-" * 60)
        
        # 컬럼 정보 출력
        print("[Column Info]")
        print(data.info())
        print("-" * 60)
        
        # 컬럼별 데이터 예시 확인
        print("[Sample Data (First 1 row)]")
        print(data.iloc[0])
        print("-" * 60)
        
        # 임베딩 컬럼 상세 확인 (embedding이라는 컬럼이 있다면)
        if 'embedding' in data.columns:
            emb_sample = data['embedding'].iloc[0]
            print(f"[Embedding Info]")
            print(f"Type: {type(emb_sample)}")
            if isinstance(emb_sample, (list, np.ndarray)):
                print(f"Dimension (Length): {len(emb_sample)}")
                print(f"Sample values: {emb_sample[:5]} ...")
    else:
        # DataFrame이 아닌 경우 (Dictionary 등)
        print(f"✅ Data Type: {type(data)}")
        if isinstance(data, dict):
            print(f"✅ Keys: {data.keys()}")
            # 첫 번째 키의 내용 확인
            first_key = list(data.keys())[0]
            print(f"First Key '{first_key}' Type: {type(data[first_key])}")

except Exception as e:
    print(f"Error loading file: {e}")
    # 일반 pickle 로드로 재시도
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"\nLoaded with standard pickle. Type: {type(data)}")
        print(data)
    except Exception as e2:
        print(f"Standard pickle load also failed: {e2}")