import sys
import os

def check_gpu():
    print("=== GPU环境诊断 ===")
    print(f"系统平台: {sys.platform}")
    print(f"Python版本: {sys.version}")
    
    # 检查CUDA
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH: {cuda_path or '未设置'}")
    
    # 检查NVML
    try:
        if sys.platform == 'win32':
            import ctypes
            try:
                ctypes.WinDLL('nvml.dll')
                print("✅ NVML库可用")
            except Exception as e:
                print(f"❌ 无法加载NVML: {str(e)}")
        
        from pynvml import nvmlInit
        nvmlInit()
        print("✅ NVML初始化成功")
        return True
    except Exception as e:
        print(f"❌ NVML测试失败: {str(e)}")
        return False

if __name__ == '__main__':
    if check_gpu():
        print("\n请将上述信息提供给开发人员")
    else:
        print("\n需要安装NVIDIA驱动和CUDA工具包")