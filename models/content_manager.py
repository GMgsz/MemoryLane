from typing import List, Dict, Optional, Set
from pydantic import BaseModel
from datetime import datetime
from models.schemas import DialogueTurn

class ContentSegment(BaseModel):
    """对话内容片段"""
    id: str
    content: str
    timestamp: datetime
    dialogue_context: List[DialogueTurn]
    entities: Dict[str, List[str]]  # 改为支持多种实体类型
    relations: Optional[List[Dict[str, str]]] = []  # 添加关系字段
    themes: List[str]
    keywords: List[str]

class SubTheme(BaseModel):
    """子主题内容"""
    name: str               # 子主题名称
    content_segments: List[ContentSegment]  # 相关内容片段
    first_mentioned: datetime
    last_updated: datetime
    related_entities: Dict[str, Set[str]]  # 相关实体及其关联

class ThematicContent(BaseModel):
    """主题内容组织"""
    main_theme: str          # 主题（如：兴趣）
    sub_themes: Dict[str, SubTheme]  # 子主题（如：篮球，音乐等）
    last_updated: datetime