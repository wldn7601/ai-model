import torch

# 경로 설정 (사용자 환경에 맞춤)
path = "/home/ubuntu/ai-model/models/light_gcn/checkpoints/best_model.pt"

try:
    checkpoint = torch.load(path, map_location='cpu')
    print("="*50)
    print(f"Type of checkpoint: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys in checkpoint: {checkpoint.keys()}")
        
        # 만약 'model_state_dict'가 있다면 그 내부도 확인
        if 'model_state_dict' in checkpoint:
            print("\nKeys in model_state_dict (first 5):")
            print(list(checkpoint['model_state_dict'].keys())[:5])
    else:
        print("Checkpoint is not a dictionary (might be raw state_dict).")
        
except Exception as e:
    print(f"Error loading: {e}")