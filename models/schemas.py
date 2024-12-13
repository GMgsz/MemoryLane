from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class DialogueTurn(BaseModel):
    id: str
    question: str
    answer: str
    topic: str
    emotion_score: float
    interest_score: float
    depth_level: int
    
class AttentionMemory(BaseModel):
    short_term: List[DialogueTurn]
    long_term: Dict[str, List[DialogueTurn]]
    topic_weights: Dict[str, float]
    emotion_history: Dict[str, List[float]] 

class TopicCompletion(BaseModel):
    """话题完整度追踪"""
    topic: str
    required_elements: List[str]        # 必要元素清单
    completed_elements: List[str]       # 已完成元素
    completion_score: float             # 完整度分数
    last_update: datetime              # 最后更新时间

class UserProfile(BaseModel):
    """用户档案"""
    id: str
    name: str
    birth_date: Optional[datetime]
    interests: Dict[str, float]         # 兴趣及其权重
    sensitive_topics: List[str]         # 敏感话题列表
    preferred_depth: int                # 偏好的对话深度

class DialogueContext(BaseModel):
    """对话上下文"""
    current_topic: str
    depth_level: int
    recent_entities: List[str]          # 最近提到的实体
    emotion_state: float                # 当前情感状态
    interest_level: float               # 当前兴趣度
    pending_questions: List[str]        # 待问问题队列
    last_response: str = ""             # 用户最后一次回答