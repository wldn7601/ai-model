# scripts/check_gpu.py
import sys

print("Checking GPU environment...\n")

# 1. CuPy 확인
try:
    import cupy as cp
    print("✓ CuPy installed")
    
    is_available = cp.cuda.is_available()
    print(f"  - CUDA available: {is_available}")
    
    if is_available:
        print(f"  - CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"  - Device count: {cp.cuda.runtime.getDeviceCount()}")
        
        # [수정] 디바이스 이름 가져오는 방식 변경 (안전한 방법)
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            device_name = props['name'].decode('utf-8')
            print(f"  - Device name: {device_name}")
        except Exception as e:
            print(f"  - Device name: (Unknown - {e})")
        
        # 간단한 GPU 연산 테스트
        x = cp.array([1, 2, 3])
        y = cp.array([4, 5, 6])
        z = x + y
        print(f"  - GPU computation test: PASSED (Result: {z})")
        
    else:
        print("✗ CUDA is NOT available. Please check your drivers.")
        sys.exit(1)

except ImportError:
    print("✗ CuPy not installed")
    sys.exit(1)
except Exception as e:
    print(f"✗ CuPy error: {e}")
    sys.exit(1)

# 2. implicit 확인
try:
    import implicit
    print("\n✓ implicit library installed")
    print(f"  - Version: {implicit.__version__}")
except ImportError:
    print("\n✗ implicit library not installed")
    sys.exit(1)

# 3. 메모리 확인
try:
    mempool = cp.get_default_memory_pool()
    free_mem = mempool.free_bytes()
    total_mem = mempool.total_bytes()
    print(f"\n✓ GPU Memory Status")
    # T4 등 클라우드 환경에서는 정확한 전체 메모리 조회가 다를 수 있어 간단히 표시
    print(f"  - Mempool used: {mempool.used_bytes() / 1024**2:.1f} MB")
except Exception as e:
    print(f"\n✗ Memory check failed: {e}")

print("\n" + "="*50)
print("GPU environment is ready for ALS training!")
print("="*50)