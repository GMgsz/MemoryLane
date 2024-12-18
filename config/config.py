from pathlib import Path
import os
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

class Config:
    # 项目路径
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 数据库配置
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'db/production/memory_lane.db')
    VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', 'db/production/vector_store')
    
    # API配置
    ZHIPUAI_API_KEY = os.getenv('ZHIPUAI_API_KEY')
    ZHIPUAI_MODEL = os.getenv('ZHIPUAI_MODEL', 'glm-4')
    
    # 智能体配置
    DIALOGUE_WINDOW_SIZE = int(os.getenv('DIALOGUE_WINDOW_SIZE', '5'))
    CLUSTER_ENTITY_THRESHOLD = int(os.getenv('CLUSTER_ENTITY_THRESHOLD', '5'))
    DIALOGUE_CHECK_INTERVAL = int(os.getenv('DIALOGUE_CHECK_INTERVAL', '10'))
    MAX_TOKEN_LIMIT = int(os.getenv('MAX_TOKEN_LIMIT', '4000'))
    
    # 提示词模板路径
    PROMPTS_PATH = PROJECT_ROOT / "prompts"
    MEMORY_LANE_PROMPTS = PROMPTS_PATH / "main"
    DIALOGUE_ANALYSIS_PROMPTS = PROMPTS_PATH / "dialogue_analysis"
    CONTENT_GENERATION_PROMPTS = PROMPTS_PATH / "content_generation"
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_PATH = PROJECT_ROOT / "logs" 