import time
import psutil
import gc
import torch

def get_gpu_memory():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        return allocated, reserved
    return 0, 0

def get_cpu_memory():
    """获取CPU内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024**2  # MB

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"