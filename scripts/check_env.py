import os
from pathlib import Path
from dotenv import load_dotenv
import sys

def check_environment():
    # 加载环境变量
    load_dotenv()
    
    # 必需的环境变量（只保留真正必需的）
    required_vars = [
        'ZHIPUAI_API_KEY',  # 只检查 API KEY
    ]
    
    # 检查必需的环境变量
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("错误: 缺少以下必需的环境变量:")
        for var in missing_vars:
            print(f"- {var}")
        return False
        
    # 检查必需的目录
    required_dirs = [
        'db/production',
        'prompts/memory_lane',
        'prompts/dialogue_analysis',
        'prompts/content_generation',
        'logs'
    ]
    
    # 创建必需的目录
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("环境检查通过！")
    return True

if __name__ == "__main__":
    if not check_environment():
        sys.exit(1) 