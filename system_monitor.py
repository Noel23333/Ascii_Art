# system_monitor.py
import psutil
import platform
from pynvml import *
from tqdm import tqdm
import time

class DynamicSystemMonitor:
    def __init__(self, update_interval=0.5):
        self.cpu_count = psutil.cpu_count()
        self.update_interval = update_interval
        self.gpu_available = False
        
        try:
            nvmlInit()
            self.gpu_available = True
            print("✅ GPU监控初始化成功")
        except Exception as e:
            print(f"❌ GPU监控初始化失败: {str(e)}")
            self.gpu_available = False
            
    def print_gpu_info():
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            print(f"检测到 {device_count} 个GPU设备")
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle)
                print(f"GPU {i}: {name.decode('utf-8')}")
        except Exception as e:
            print(f"无法检测GPU: {str(e)}")
    
    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=self.update_interval)
    
    def get_memory_usage(self):
        return psutil.virtual_memory().percent
    
    def get_gpu_stats(self):
        if not self.gpu_available:
            return 0, 0, "N/A"
        
        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            
            gpu_mem_percent = (mem.used / mem.total) * 100
            
            return (
                util.gpu,  # GPU计算利用率
                gpu_mem_percent,  # 显存使用百分比
                f"{mem.used//1024**2}MB/{mem.total//1024**2}MB",  # 显存用量
            )
        except Exception as e:
            print(f"获取GPU状态失败: {str(e)}")
            return 0, 0, "Error"
    
    def get_dynamic_stats(self):
        cpu = self.get_cpu_usage()
        mem = self.get_memory_usage()
        
        stats = {
            "CPU": f"{cpu:.1f}%",
            "Memory": f"{mem:.1f}%"
        }
        
        if self.gpu_available:
            gpu_usage, gpu_mem, gpu_mem_str = self.get_gpu_stats()
            stats.update({
                "GPU": f"{gpu_usage:.1f}%",
                "GPU Mem": f"{gpu_mem:.1f}% ({gpu_mem_str})"
            })
        
        return stats

def create_live_progress_bar(total, desc="Processing"):
    return tqdm(
        total=total,
        desc=desc,
        unit="row",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )