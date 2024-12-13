import uuid
from datetime import datetime
import asyncio
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import SystemMessage, HumanMessage
from models.schemas import DialogueTurn
from models.content_manager import ContentSegment
from core.vector_store import VectorStoreManager
from utils.api_manager import api_manager
from config.config import Config

class ContentProcessor:
    def __init__(self, 
                 extract_llm: ChatZhipuAI, 
                 identify_llm: ChatZhipuAI,
                 vector_store: VectorStoreManager):
        self.extract_llm = extract_llm
        self.identify_llm = identify_llm
        self.vector_store = vector_store
        
    async def process_dialogue(self, 
                             dialogue_turn: DialogueTurn,
                             dialogue_context: List[DialogueTurn]) -> ContentSegment:
        """处理对话内容，生成内容片段"""
        try:
            # 1. 提取实体和关键词
            entities_and_keywords = await self._extract_entities_and_keywords(
                dialogue_turn.answer
            )
            
            # 2. 识别可能的主题
            themes = await self._identify_themes(
                dialogue_turn.answer,
                entities_and_keywords
            )
            
            # 3. 创建内容片段
            segment = ContentSegment(
                id=str(uuid.uuid4()),
                content=dialogue_turn.answer,
                timestamp=datetime.now(),
                dialogue_context=dialogue_context[-3:],
                entities=entities_and_keywords['entities'],
                themes=themes,
                keywords=entities_and_keywords['keywords']
            )
            
            # 4. 存储到向量数据库
            metadata = {
                "id": segment.id,
                "dialogue_id": dialogue_turn.id,
                "question": dialogue_turn.question,
                "timestamp": segment.timestamp.isoformat(),
                "themes": segment.themes,
                "entities": segment.entities,
                "relations": segment.relations,
                "keywords": segment.keywords
            }
            
            await self._store_segment(segment, metadata)
            
            return segment
            
        except Exception as e:
            print(f"处理对话时出错: {e}")
            return ContentSegment(
                id=str(uuid.uuid4()),
                content=dialogue_turn.answer,
                timestamp=datetime.now(),
                dialogue_context=dialogue_context[-3:],
                entities={},
                themes=[dialogue_turn.topic],
                keywords=[]
            )
            
    async def _extract_entities_and_keywords(self, text: str) -> Dict:
        """使用LLM提取实体和关键词"""
        system_message = SystemMessage(content="""
            你是一个专业的信息提取助手。请仔细分析文本并提取以下信息：
            
            1. 实体：
               - 人物：包括人称代词、称谓、角色
               - 时间：具体时间点、时期、年代、频率词
               - 地点：具体地点、场所、区域
               - 事件：发生的事情、活动、行为
               - 物品：重要的物件、物品
               
            2. 关系：
               - 人物之间的关系
               - 事件之间的因果关系
               - 时间和事件的关联
               
            3. 关键词：对理解内容重要的词语
            
            示例分析：
            输入："从小父母教育我要努力学习"
            分析：
            {
                "entities": {
                    "人物": ["我", "父母"],
                    "时间": ["从小"],
                    "事件": ["教育", "学习"]
                },
                "relations": [
                    {"from": "父母", "relation": "教育", "to": "我"}
                ],
                "keywords": ["教育", "学习", "从小"]
            }
        """)
        
        human_message = HumanMessage(content=text)
        
        try:
            response = await api_manager.execute_with_retry(
                self.extract_llm.ainvoke,  # 使用专门的extract_llm
                [system_message, human_message]
            )
            return self._parse_response(response.content)
        except Exception as e:
            print(f"API调用失败: {e}")
            return {"entities": {}, "keywords": []}
            
    def _parse_response(self, response_text: str) -> Dict:
        """解析LLM响应"""
        try:
            from utils.json_parser import ResponseParser
            result = ResponseParser.parse_llm_response(response_text)
            if not isinstance(result, dict):
                return {"entities": {}, "keywords": []}
            return {
                "entities": result.get("entities", {}),
                "keywords": result.get("keywords", [])
            }
        except Exception as e:
            print(f"响应解析失败: {e}")
            return {"entities": {}, "keywords": []}
        
    async def _identify_themes(self, text: str, entities_and_keywords: Dict) -> List[str]:
        """识别文本可能属于的主题"""
        system_message = SystemMessage(content=f"""
            你是一个专业的主题分析助手。请仔细分析用户回答涉及的主题。
            
            可选主题仅限于以下选项：
            - 家庭：家庭生活、亲情关系
            - 早年生活：童年、学生时期的经历
            - 友谊：朋友关系、社交经历
            - 影响：生命中的重要影响
            - 成就：个人成就、成功经历
            - 职业生涯：工作、事业相关
            - 兴趣：个人爱好、兴趣发展
            - 信仰：价值观、人生信念
            - 关键事件：人生重要时刻
            - 旅行：旅行经历、见闻
            - 其他：不属于以上类别的内容
            
            分析原则：
            1. 必须从上述主题中选择
            2. 返回1-3个最相关的主题
            3. 按相关程度排序
            4. 如果都不符合，返回"其他"
            
            示例分析：
            输入："父母从小就很关心我的学习"
            分析：涉及家庭关系和早年生活 -> ["家庭", "早年生活"]
            
            输入："大学时认识了一个影响我很深的朋友"
            分析：涉及友谊和人生影响 -> ["友谊", "影响"]
            
            输入："工作后经常出差，去过很多地方"
            分析：涉及职业生涯和旅行 -> ["职业生涯", "旅行"]
            
            当前输入：{text}
            提取的实体：{entities_and_keywords['entities']}
            关键词：{entities_and_keywords['keywords']}
            
            请按相关程度排序返回1-3个最相关的主题。
            返回格式：["主题1", "主题2", "主题3"]
        """)
        
        human_message = HumanMessage(content=f"""
            文本内容：{text}
            提取的实体：{entities_and_keywords['entities']}
            关键词：{entities_and_keywords['keywords']}
        """)
        
        try:
            response = await api_manager.execute_with_retry(
                self.identify_llm.ainvoke,
                [system_message, human_message]
            )
            themes = self._parse_themes(response.content)
            # 确保返回的主题在预定义列表中
            valid_themes = [theme for theme in themes if theme in Config.TOPICS]
            return valid_themes[:3] if valid_themes else ["其他"]
        except Exception as e:
            print(f"主题识别失败: {e}")
            return ["其他"]
        
    async def _store_segment(self, segment: ContentSegment, metadata: Dict):
        """将内容片段存储到向量数据库"""
        await self.vector_store.add_memory(
            text=segment.content,
            metadata=metadata
        )
        
    def _parse_themes(self, response_text: str) -> List[str]:
        """解析主题识别响应"""
        try:
            from utils.json_parser import ResponseParser
            result = ResponseParser.parse_llm_response(response_text)
            if isinstance(result, list):
                return result
            return list(result.values()) if isinstance(result, dict) else ["其他"]
        except Exception as e:
            print(f"主题解析失败: {e}")
            return ["其他"]