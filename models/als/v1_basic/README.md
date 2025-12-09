# ALS



```bash
# 가상환경 활성화
cd ~/ai-model
source venv311/bin/activate

# 의존성 설치
pip install -r requirements.txt

# als 모델로  이동
cd ~/ai-model/models/als/v1_basic/
```

```bash
# scripts 폴더아래의 run_~~.py 파일을 실행
cd scripts/

# 전처리
python run_preprocess_data.py

# 모델 학습
python run_train.py

# 모델 평가
python run_evaluation.py
```

## 모델 평가 결과 
```bash
# 경로 이동
cd ~/ai-model/models/als/v1_basic/results/

# evaluation_results.json 파일 확인
```