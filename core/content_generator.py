from typing import List, Dict
from datetime import datetime
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import SystemMessage, HumanMessage
from models.content_manager import ThematicContent, SubTheme, ContentSegment

class ContentGenerator:
    def __init__(self, llm: ChatZhipuAI):
        self.llm = llm
        
    async def generate_theme_content(self, 
                                   theme_content: ThematicContent) -> str:
        """为主题生成内容"""
        
        # 1. 整理子主题内容
        organized_content = self._organize_content(theme_content)
        
        # 2. 生成内容
        return await self._generate_content(
            theme_content.main_theme,
            organized_content
        )
        
    def _organize_content(self, theme_content: ThematicContent) -> Dict:
        """整理主题内容，按子主题组织"""
        organized = {
            "main_theme": theme_content.main_theme,
            "sub_themes": {},
            "timeline": [],
            "key_entities": {}
        }
        
        # 处理每个子��题
        for sub_name, sub_theme in theme_content.sub_themes.items():
            # 收集子主题内容
            sub_content = {
                "name": sub_name,
                "segments": [seg.content for seg in sub_theme.content_segments],
                "context": [
                    {
                        "question": seg.dialogue_context[-1].question,
                        "answer": seg.dialogue_context[-1].answer
                    }
                    for seg in sub_theme.content_segments
                ],
                "entities": sub_theme.related_entities,
                "first_mentioned": sub_theme.first_mentioned,
                "last_updated": sub_theme.last_updated
            }
            
            organized["sub_themes"][sub_name] = sub_content
            
            # 添加到时间线
            organized["timeline"].append({
                "time": sub_theme.first_mentioned,
                "event": sub_name
            })
            
            # 收集关键实体
            for entity_type, entities in sub_theme.related_entities.items():
                if entity_type not in organized["key_entities"]:
                    organized["key_entities"][entity_type] = set()
                organized["key_entities"][entity_type].update(entities)
        
        # 按时间排序
        organized["timeline"].sort(key=lambda x: x["time"])
        
        return organized
        
    async def _generate_content(self, 
                              theme: str, 
                              organized_content: Dict) -> str:
        """生成最终内容"""
        system_message = SystemMessage(content="""
            你是一个专业的传记作家。请根据提供的信息，生成一段连贯、生动的叙述。
            要求：
            1. 内容要有逻辑性和连贯性
            2. 保持时间线的顺序
            3. 自然地融入关键实体
            4. 语言要生动活泼
            5. 注意情感的表达
            6. 各个子主题之间要有合理的过渡
        """)
        
        # 构建提示
        content_prompt = f"""
        主题：{theme}
        
        子主题内容：
        {self._format_content_for_prompt(organized_content)}
        
        时间线：
        {self._format_timeline(organized_content['timeline'])}
        
        关键实体：
        {self._format_entities(organized_content['key_entities'])}
        
        请生成一段完整的叙述。
        """
        
        human_message = HumanMessage(content=content_prompt)
        
        # 生成��容
        response = await self.llm.ainvoke([system_message, human_message])
        
        return response.content
        
    def _format_content_for_prompt(self, organized_content: Dict) -> str:
        """格式化内容用于提示"""
        formatted = []
        for sub_name, sub_content in organized_content['sub_themes'].items():
            formatted.append(f"\n子主题：{sub_name}")
            formatted.append("相关内容：")
            for segment in sub_content['segments']:
                formatted.append(f"- {segment}")
        return "\n".join(formatted)
        
    def _format_timeline(self, timeline: List[Dict]) -> str:
        """格式化时间线"""
        return "\n".join([
            f"- {event['time'].strftime('%Y-%m-%d')}: {event['event']}"
            for event in timeline
        ])
        
    def _format_entities(self, entities: Dict) -> str:
        """格式化实体信息"""
        formatted = []
        for entity_type, entity_set in entities.items():
            formatted.append(f"\n{entity_type}:")
            for entity in entity_set:
                formatted.append(f"- {entity}")
        return "\n".join(formatted) 