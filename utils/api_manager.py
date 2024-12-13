import asyncio
import time
from typing import Any, Callable
from functools import wraps

class APIRateLimiter:
    def __init__(self, max_requests: int = 1, time_window: int = 2):
        self.max_requests = max_requests  # 每个时间窗口允许的最大请求数
        self.time_window = time_window    # 时间窗口（秒）
        self.requests = []                # 请求时间戳列表
        self.lock = asyncio.Lock()        # 异步锁

    async def wait_if_needed(self):
        """检查是否需要等待"""
        async with self.lock:
            current_time = time.time()
            
            # 清理过期的请求记录
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.time_window]
            
            # 如果达到限制，等待直到可以发送新请求
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.time_window - current_time
                if wait_time > 0:
                    print(f"API限制，等待 {wait_time:.2f} 秒...")
                    await asyncio.sleep(wait_time)
                
            # 添加新请求
            self.requests.append(current_time)

class APIManager:
    def __init__(self):
        self.rate_limiter = APIRateLimiter(max_requests=1, time_window=3)  # 每3秒1个请求
        self.max_retries = 3
        self.base_delay = 10  # 增加基础延迟到10秒

    async def execute_with_retry(self, 
                               func: Callable, 
                               *args, 
                               **kwargs) -> Any:
        """执行API调用，带重试机制"""
        for attempt in range(self.max_retries):
            try:
                # 等待限流检查
                await self.rate_limiter.wait_if_needed()
                
                # 执行API调用
                return await func(*args, **kwargs)
                
            except Exception as e:
                if "429" in str(e):  # Too Many Requests
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (attempt + 1)
                        print(f"API频率限制，等待 {delay} 秒后重试...")
                        await asyncio.sleep(delay)
                    else:
                        print("达到最大重试次数，操作失败")
                        raise
                else:
                    raise

api_manager = APIManager()  # 创建全局实例 