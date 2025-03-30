from pynvml import *
import os

nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

processes = nvmlDeviceGetComputeRunningProcesses(gpu_handle)
pid = os.getpid()
gpu_used = None

for proc in processes:
    if proc.pid == pid:
        gpu_used = proc.usedGpuMemory / 1024 / 1024  # 转 MB

if gpu_used is None:
    print("当前进程未使用 GPU")
else:
    print(f"当前进程 GPU 占用: {gpu_used:.1f}MB")
