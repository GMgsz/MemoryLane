from typing import Optional, Dict
from Agent.MemoryLaneAgent import MemoryLaneAgent
from Agent.DialogueAnalysisAgent import DialogueAnalysisAgent, DialogueTurn
from Agent.ContentGenerationAgent import ContentGenerationAgent
from Utils.DatabaseManager import DatabaseManager
from config.config import Config
from Utils.logger import setup_logger
from datetime import datetime
import uuid
from langchain_community.chat_models import ChatZhipuAI

class MemoryLaneController:
    def __init__(self):
        # 设置日志
        self.logger = setup_logger("MemoryLaneController")
        
        try:
            # 初始化LLM
            self.llm = ChatZhipuAI(
                temperature=0.7,
                model_name=Config.ZHIPUAI_MODEL,
                api_key=Config.ZHIPUAI_API_KEY
            )
            
            # 初始化数据库管理器
            self.db_manager = DatabaseManager(
                db_path=Config.DATABASE_PATH
            )
            
            # 初始化智能体
            self.dialogue_agent = MemoryLaneAgent(
                llm=self.llm,
                prompts_path=str(Config.MEMORY_LANE_PROMPTS),
                main_prompt_file="main.json",
                max_token_limit=Config.MAX_TOKEN_LIMIT
            )
            
            self.analysis_agent = DialogueAnalysisAgent(
                llm=self.llm,
                window_size=Config.DIALOGUE_WINDOW_SIZE,
                db_manager=self.db_manager
            )
            
            self.generation_agent = ContentGenerationAgent(
                db_manager=self.db_manager,
                llm_client=self.llm
            )
            
            self.dialogue_count = 0
            self.current_system_query = None
            
            self.logger.info("MemoryLaneController initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MemoryLaneController: {str(e)}")
            raise
    
    async def start_conversation(self) -> str:
        """开始对话，生成第一个系统问题"""
        try:
            self.current_system_query = self.dialogue_agent.start_chat()
            self.logger.info("Started new conversation")
            return self.current_system_query
        except Exception as e:
            self.logger.error(f"Failed to start conversation: {str(e)}")
            raise
    
    async def process_user_input(self, user_input: str) -> Dict[str, str]:
        """处理用户输入并返回下一个系统问题"""
        try:
            if self.current_system_query is None:
                raise ValueError("No current system query. Please start conversation first.")
            
            # 1. 处���当前对话
            self.logger.debug(f"Processing dialogue - System: {self.current_system_query}, User: {user_input}")
            
            # 创建对话记录
            dialogue_turn = DialogueTurn(
                id=str(uuid.uuid4()),
                timestamp=str(datetime.now()),  # 转换为字符串
                system_query=self.current_system_query,
                user_response=user_input
            )
            
            # 处理对话
            self.analysis_agent.process_dialogue(dialogue_turn)
            
            # 2. 生成下一个系统问题
            next_system_query = self.dialogue_agent.chat(user_input)
            
            # 3. 检查是否需要触发内容生成
            self.dialogue_count += 1
            if self.dialogue_count >= Config.DIALOGUE_CHECK_INTERVAL:
                self.dialogue_count = 0
                self.generation_agent.on_dialogue_complete(str(uuid.uuid4()))
            
            # 4. 更新当前系统问题
            result = {
                "previous_turn": {
                    "system_query": self.current_system_query,
                    "user_response": user_input
                },
                "next_query": next_system_query
            }
            self.current_system_query = next_system_query
            
            self.logger.debug(f"Generated next query: {next_system_query}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing user input: {str(e)}")
            raise

    async def get_generated_contents(self, topic: Optional[str] = None):
        """获取已生成的内容"""
        try:
            contents = self.db_manager.get_cluster_contents(topic) if topic else []
            self.logger.debug(f"Retrieved {len(contents)} generated contents for topic: {topic}")
            return contents
        except Exception as e:
            self.logger.error(f"Error retrieving generated contents: {str(e)}")
            raise
            
    async def close(self):
        """关闭资源"""
        try:
            if self.db_manager:
                self.db_manager.close()
            self.logger.info("MemoryLaneController closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing MemoryLaneController: {str(e)}")
            raise