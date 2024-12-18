import logging
import sys
from pathlib import Path
from config.config import Config

def setup_logger(name: str) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # 确保日志目录存在
    Config.LOG_PATH.mkdir(exist_ok=True)
    
    # 文件处理器
    file_handler = logging.FileHandler(
        Config.LOG_PATH / f"{name}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 