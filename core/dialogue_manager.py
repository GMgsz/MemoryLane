from typing import Dict, List, Optional
from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from models.schemas import DialogueTurn, AttentionMemory, DialogueContext, TopicCompletion
from config.config import Config
import random
from utils.api_manager import api_manager

class DialogueManager:
    def __init__(self, llm: ChatZhipuAI):
        self.llm = llm
        self.attention_memory = AttentionMemory(
            short_term=[],
            long_term={},
            topic_weights={},
            emotion_history={}
        )
        self.current_context = DialogueContext(
            current_topic="家庭",
            depth_level=0,
            recent_entities=[],
            emotion_state=0.0,
            interest_level=0.5,
            pending_questions=[]
        )
        
    async def generate_next_question(self, 
                                   metrics: Dict[str, float],
                                   context: DialogueContext) -> str:
        """基于各项指标生成下一个问题"""
        
        # 基于综合指标决定策略
        strategy = self._determine_question_strategy(metrics, context)
        
        # 生成问题
        prompt = self._create_question_prompt(strategy, context)
        
        system_message = SystemMessage(content="""
            你是一个专业的传记作家助手，负责通过对话的方式收集用户的生平故事。
            你的回复应该包含两个部分：
            1. 对用户刚才回答的简短、温暖的回应（1-2句话）
            2. 根据上下文和策略提出下一个问题
            
            回复要自然、友好，能够引导用户分享更多细节。
            不要直接返回提示词或模板。
        """)
        
        human_message = HumanMessage(content=f"""
            用户刚才的回答是：{context.last_response}
            
            {prompt}
        """)
        
        try:
            response = await api_manager.execute_with_retry(
                self.llm.ainvoke,
                [system_message, human_message]
            )
            return response.content
        except Exception as e:
            print(f"生成问题失败: {e}")
            return "能告诉我更多吗？"  # 返回一个通用的后备问题
    
    def _determine_question_strategy(self, 
                                   metrics: Dict[str, float],
                                   context: DialogueContext) -> Dict:
        """基于指标确定提问策略"""
        
        # 获取当前指标
        emotion_score = metrics['emotion_score']
        interest_score = metrics['interest_score']
        completion_score = metrics['completion_score']
        topic_weight = metrics['topic_weight']
        
        strategy = {
            'action': None,  # 可能的值：deepen, switch, probe, conclude
            'depth_change': 0,
            'focus_aspect': None,
            'new_topic': None
        }
        
        # 话题切换条件判断
        should_switch_topic = (
            # 1. 明确的切换请求
            any(keyword in context.last_response.lower() for keyword in [
                "换个话题", "换一个话题", "聊点别的"
            ]) or
            # 2. 情感指标
            abs(emotion_score) > Config.EMOTION_THRESHOLD or
            # 3. 兴趣度过低
            interest_score < 0.3 or
            # 4. 话题完整度高且兴趣度一般
            (completion_score > 0.8 and interest_score < 0.6) or
            # 5. 简短否定回答模式
            (len(context.last_response) < 10 and 
             any(word in context.last_response.lower() for word in ["没有", "不知道", "不记得"]))
        )
        
        if should_switch_topic:
            strategy['action'] = 'switch'
            strategy['new_topic'] = self._select_new_topic(
                context, metrics, current_topic=context.current_topic
            )
            return strategy
            
        # 如果不需要切换话题，进行其他策略判断
        if interest_score > 0.8:
            strategy['action'] = 'deepen'
            strategy['depth_change'] = 1
        elif completion_score < 0.6:
            strategy['action'] = 'probe'
            strategy['focus_aspect'] = self._get_missing_aspects(context.current_topic)
        else:
            strategy['action'] = 'continue'
            
        return strategy
    
    def _create_question_prompt(self, 
                              strategy: Dict,
                              context: DialogueContext) -> str:
        """根据策略创建提问模板"""
        
        base_prompt = """
        基于以下信息生成后续对话：
        
        当前话题: {current_topic}
        当前深度: {depth_level}
        最近提到的实体: {entities}
        对话���略: {strategy}
        用户最后的回答: {last_response}
        
        要求：
        1. 先对用户的回答做出温暖的回应
        2. 然后自然地引导到下一个问题
        3. 考虑上下文连贯性
        4. 符合当前深度级别
        5. 注意情感适当性
        6. 如果用户要求换话题，必须切换到新话题
        """
        
        if strategy['action'] == 'switch':
            base_prompt += f"""
            - 礼貌地结束当前话题
            - 自然地切换到新话题：{strategy.get('new_topic', '用户指定的话题')}
            - 以开放性问题开始新话题
            """
        
        # 填充模板
        prompt = PromptTemplate(
            template=base_prompt,
            input_variables=["current_topic", "depth_level", "entities", "strategy", "last_response"]
        )
        
        return prompt.format(
            current_topic=context.current_topic,
            depth_level=context.depth_level,
            entities=", ".join(context.recent_entities),
            strategy=strategy['action'],
            last_response=context.last_response
        )
    
    def _get_missing_aspects(self, topic: str) -> List[str]:
        """获取当前话题未覆盖的方面"""
        # 这里需要实现具���的逻辑
        pass 
    
    def _select_new_topic(self, 
                         context: DialogueContext, 
                         metrics: Dict[str, float],
                         current_topic: str) -> str:
        """智能选择新话题"""
        
        # 1. 从用户回答中提取可能的话题偏好
        response = context.last_response.lower()
        explicit_topic = None
        
        topic_keywords = {
            "兴趣": ["兴趣", "爱好", "喜欢", "热爱"],
            "职业生涯": ["工作", "职业", "事业", "公司"],
            "早年生活": ["童年", "小时候", "学生时代"],
            # ... 其他话题关键词映射
        }
        
        # 检查是否有明确提到的话题
        for topic, keywords in topic_keywords.items():
            if any(keyword in response for keyword in keywords):
                explicit_topic = topic
                break
        
        if explicit_topic:
            return explicit_topic
            
        # 2. 如果没有明确话题，基于历史兴趣度选择
        available_topics = [
            topic for topic in Config.TOPICS 
            if topic != current_topic
        ]
        
        # 获取历史兴趣度数据
        topic_interests = self.attention_memory.topic_weights
        
        # 如果有历史数据，选择历史兴趣度最高的话题
        if topic_interests:
            sorted_topics = sorted(
                available_topics,
                key=lambda t: topic_interests.get(t, 0.5),
                reverse=True
            )
            return sorted_topics[0]
            
        # 3. 如果没有历史数据，随机选择
        return random.choice(available_topics)