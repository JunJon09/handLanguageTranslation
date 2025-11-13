import psutil
import time
import threading
from datetime import datetime
import torch

class PerformanceMonitor:
    """システムパフォーマンス監視クラス"""
    
    def __init__(self, monitor_interval=1.0):
        self.monitor_interval = monitor_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # メトリクス記録用
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
        
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_memory_usage.clear()
        self.timestamps.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        

        
    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        print("⏹️ パフォーマンス監視を停止しました")
        
    def _monitor_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                # 現在時刻記録
                current_time = datetime.now()
                self.timestamps.append(current_time)
                
                # CPU使用率 (プロセス単位)
                cpu_percent = self.process.cpu_percent()
                self.cpu_usage.append(cpu_percent)
                
                # メモリ使用量 (プロセス単位)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # MB
                self.memory_usage.append(memory_mb)
                
                # GPU メモリ使用量 (PyTorch使用時)
                gpu_memory_mb = 0
                if torch.cuda.is_available():
                    gpu_memory_bytes = torch.cuda.memory_allocated()
                    gpu_memory_mb = gpu_memory_bytes / 1024 / 1024  # MB
                
                self.gpu_memory_usage.append(gpu_memory_mb)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"パフォーマンス監視エラー: {e}")
                
    def get_summary(self):
        """パフォーマンス統計サマリー取得"""
        if not self.cpu_usage:
            return "監視データがありません"
            
        summary = {
            'monitoring_duration': len(self.timestamps) * self.monitor_interval,
            'cpu_usage': {
                'avg': sum(self.cpu_usage) / len(self.cpu_usage),
                'max': max(self.cpu_usage),
                'min': min(self.cpu_usage)
            },
            'memory_usage_mb': {
                'avg': sum(self.memory_usage) / len(self.memory_usage),
                'max': max(self.memory_usage),
                'min': min(self.memory_usage)
            },
            'gpu_memory_mb': {
                'avg': sum(self.gpu_memory_usage) / len(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
                'max': max(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
                'min': min(self.gpu_memory_usage) if self.gpu_memory_usage else 0
            }
        }
        
        return summary
        
    def print_summary(self):
        """パフォーマンス統計を表示"""
        summary = self.get_summary()
        
        if isinstance(summary, str):
            print(summary)
            return
            
        print("\n" + "="*60)
        print("🔍 パフォーマンス監視結果")
        print("="*60)
        print(f"📊 監視時間: {summary['monitoring_duration']:.1f}秒")
        print(f"🖥️  CPU使用率: 平均{summary['cpu_usage']['avg']:.1f}% | 最大{summary['cpu_usage']['max']:.1f}% | 最小{summary['cpu_usage']['min']:.1f}%")
        print(f"💾 メモリ使用量: 平均{summary['memory_usage_mb']['avg']:.0f}MB | 最大{summary['memory_usage_mb']['max']:.0f}MB | 最小{summary['memory_usage_mb']['min']:.0f}MB")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU メモリ: 平均{summary['gpu_memory_mb']['avg']:.0f}MB | 最大{summary['gpu_memory_mb']['max']:.0f}MB | 最小{summary['gpu_memory_mb']['min']:.0f}MB")
        else:
            print("🎮 GPU: 利用不可")
            
        print("="*60)