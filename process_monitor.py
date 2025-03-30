import psutil
import os
from pynvml import *
from tqdm import tqdm
import time

class ProcessMonitor:
    def __init__(self):
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self._last_cpu_time = 0
        self.gpu_available = False
        self._init_gpu()
        
        # 预热CPU监控
        self.get_cpu_usage()
    
    def _init_gpu(self):
        try:
            nvmlInit()
            self.gpu_available = nvmlDeviceGetCount() > 0
            if self.gpu_available:
                self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            self.gpu_available = False
    
    def get_cpu_usage(self):
        """更精确的CPU计算（基于时间差）"""
        try:
            cpu_times = self.process.cpu_times()
            current_time = time.time()
            current_cpu = sum(cpu_times[:4])  # user + system
            
            if self._last_cpu_time > 0:
                usage = (current_cpu - self._last_cpu_time) / \
                       (current_time - self._last_time) * 100
                usage = min(usage / psutil.cpu_count(), 100)  # 多核归一化
            else:
                usage = 0.0
            
            self._last_cpu_time = current_cpu
            self._last_time = current_time
            return round(usage, 1)
        except:
            return 0.0
    
    def get_memory_usage(self):
        """获取当前进程内存占用（MB）"""
        try:
            mem = self.process.memory_info()
            return {
                'rss': round(mem.rss / 1024 / 1024, 1),  # 常驻内存(MB)
                'vms': round(mem.vms / 1024 / 1024, 1),  # 虚拟内存(MB)
                'percent': round(self.process.memory_percent(), 1)
            }
        except psutil.NoSuchProcess:
            return {'rss': 0, 'vms': 0, 'percent': 0}
    
    def get_gpu_usage(self):
        if not self.gpu_available:
            return None
        
        try:
            # 获取设备级利用率
            util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # 获取进程显存
            processes = nvmlDeviceGetComputeRunningProcesses(self.gpu_handle)
            mem_usage = sum(
                p.usedGpuMemory / 1024 / 1024 
                for p in processes 
                if p.pid == self.pid and p.usedGpuMemory
            )
            
            return {
                'gpu_util': util.gpu,
                'mem_util': util.memory,
                'used_memory': round(mem_usage, 1)
            }
        except:
            return {'gpu_util': 0, 'mem_util': 0, 'used_memory': 0.0}
    
    def get_stats(self):
        """获取完整的资源统计（包含异常处理）"""
        stats = {
            'cpu': self.get_cpu_usage(),
            'memory': self.get_memory_usage(),
            'gpu': None
        }
        
        if self.gpu_available:
            gpu_stats = self.get_gpu_usage()
            if gpu_stats:
                stats['gpu'] = gpu_stats
        
        return stats

    def __del__(self):
        """清理NVML资源"""
        if self.gpu_available:
            try:
                nvmlShutdown()
            except:
                pass

def create_process_aware_progress(total, desc="Processing"):
    """创建带资源监控的进度条"""
    return tqdm(
        total=total,
        desc=desc,
        unit="item",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        postfix="等待初始数据..."
    )