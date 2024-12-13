import asyncio
from typing import Callable
import time

class AutoSaver:
    def __init__(self, 
                 save_function: Callable, 
                 interval: int = 300):  # 默认5分钟
        self.save_function = save_function
        self.interval = interval
        self.last_save_time = time.time()
        self.running = False
        
    async def start(self):
        """启动自动保存"""
        self.running = True
        while self.running:
            await asyncio.sleep(self.interval)
            if self.running:  # 再次检查，避免在sleep期间被停止
                await self.save_function()
                self.last_save_time = time.time()
                
    def stop(self):
        """停止自动保存"""
        self.running = False 