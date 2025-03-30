from pynvml import *

nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)
gpu_util = nvmlDeviceGetUtilizationRates(gpu_handle).gpu
print(f"GPU 设备总利用率: {gpu_util}%")