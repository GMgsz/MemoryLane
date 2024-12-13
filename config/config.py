from typing import List

class Config:
    # 主题配置
    TOPICS = [
        "家庭",
        "早年生活",
        "友谊",
        "影响",
        "成就",
        "职业生涯",
        "兴趣",
        "信仰",
        "关键事件",
        "旅行",
        "其他"
    ]
    
    # 对话配置
    MAX_CONTEXT_LENGTH = 2000
    MAX_TURNS_PER_TOPIC = 5
    EMOTION_THRESHOLD = 0.8
    
    # 每个主题的必要元素
    TOPIC_ELEMENTS = {
        "家庭": ["家庭成员", "重要事件", "家庭氛围", "家庭传统"],
        "早年生活": ["童年记忆", "生活环境", "重要人物", "关键事件"],
        "友谊": ["朋友关系", "社交经历", "重要时刻"],
        "影响": ["重要人物", "关键事件", "影响程度"],
        "成就": ["重要成果", "过程经历", "感悟"],
        "职业生涯": ["工作经历", "职业发展", "重要项目"],
        "兴趣": ["爱好类型", "发展历程", "收获"],
        "信仰": ["价值观", "人生观", "精神追求"],
        "关键事件": ["事件描述", "影响", "感悟"],
        "旅行": ["地点", "经历", "感受"],
        "其他": ["基本信息"]
    }
    
    # 内容生成配置
    CONTENT_GENERATION = {
        "MIN_SEGMENTS": 2,          # 最小内容片段数
        "MIN_WORDS": 100,           # 最小字数
        "MIN_TIME_SPAN": 0,         # 最小时间跨度（天）
        "INTEREST_THRESHOLD": 0.7,   # 兴趣度阈值
        "SIMILARITY_THRESHOLD": 0.6  # 内容相似度阈值
    }
    
    # 主题配置扩展
    THEME_STRUCTURE = {
        "家庭": {
            "min_segments": 2,
            "required_aspects": ["成员", "关系", "事件"]
        },
        "兴趣": {
            "min_segments": 3,
            "required_aspects": ["起源", "发展", "现状"]
        },
        "其他": {
            "min_segments": 2,
            "required_aspects": ["基本信息"]
        }
        # ... 其他主题
    }