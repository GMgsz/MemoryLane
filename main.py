import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from core.dialogue_manager import DialogueManager
from core.vector_store import VectorStoreManager
from core.content_processor import ContentProcessor
from core.theme_manager import ThemeManager
from core.content_generator import ContentGenerator
from models.schemas import DialogueContext, DialogueTurn
from typing import List, Dict
import uuid

class MemoryLane:
    def __init__(self):
        # 加载环境变量
        _ = load_dotenv(find_dotenv())
        
        # 初始化不同用途的LLM
        self.extract_llm = ChatZhipuAI(
            model=os.getenv('Extract_Model'),
            api_key=os.getenv('Extract_API_key')
        )
        
        self.identify_llm = ChatZhipuAI(
            model=os.getenv('Identify_Model'),
            api_key=os.getenv('Identify_API_key')
        )
        
        self.generate_llm = ChatZhipuAI(
            model=os.getenv('Generate_Model'),
            api_key=os.getenv('Generate_API_key')
        )
        
        self.embeddings = ZhipuAIEmbeddings(
            model=os.getenv('Embedding_model'),
            api_key=os.getenv('Embedding_API_key')
        )
        
        # 初始化组件
        self.dialogue_manager = DialogueManager(self.generate_llm)
        self.vector_store = VectorStoreManager(self.embeddings)
        self.content_processor = ContentProcessor(
            extract_llm=self.extract_llm,
            identify_llm=self.identify_llm,
            vector_store=self.vector_store
        )
        self.theme_manager = ThemeManager()
        self.content_generator = ContentGenerator(self.generate_llm)
        
        # 初始化上下文
        self.context = DialogueContext(
            current_topic="家庭",
            depth_level=0,
            recent_entities=[],
            emotion_state=0.0,
            interest_level=0.5,
            pending_questions=[],
            last_response=""
        )
        
        # 添加对话历史
        self.dialogue_history: List[DialogueTurn] = []
        
        # 添加生成的内容存储
        self.generated_contents: Dict[str, str] = {}
        
        # 初始化last_question
        self.last_question: str = ""
        
    async def start_conversation(self):
        """开始对话"""
        print("欢迎使用 MemoryLane！让我们开始记录您的故事。")
        print("可用命令：")
        print("- 'show content': 显示所有生成的内容")
        print("- 'show content <主题>': 显示特定主题的内容")
        print("- 'exit': 退出程序")
        
        # 第一个问题
        self.last_question = "能告诉我一些关于您家庭的事情吗？"
        print(f"\n系统: {self.last_question}")
        
        while True:
            # 获取用户输入
            user_input = input("\n您: ").lower()
            
            # 处理命令
            if user_input == 'exit':
                break
            elif user_input.startswith('show content'):
                parts = user_input.split()
                theme = parts[2] if len(parts) > 2 else None
                content = await self.show_generated_content(theme)
                print(f"\n{content}")
                continue
            
            # 处理普通对话
            response = await self.process_user_input(user_input)
            print(f"\n系统: {response}")
    
    async def process_user_input(self, user_input: str):
        # 更新上下文中的最后回答
        self.context.last_response = user_input
        
        # 创建当前对话轮次
        current_turn = DialogueTurn(
            id=str(uuid.uuid4()),
            question=self.last_question,
            answer=user_input,
            topic=self.context.current_topic,
            emotion_score=0.5,
            interest_score=0.7,
            depth_level=self.context.depth_level
        )
        
        # 添加到对话历史
        self.dialogue_history.append(current_turn)
        
        # 处理内容
        content_segment = await self.content_processor.process_dialogue(
            current_turn,
            self.dialogue_history
        )
        
        # 处理内容片段，检查是否需要生成内容
        themes_to_generate = await self.theme_manager.process_content(content_segment)
        
        # 如果有主题需要生成内容
        for theme in themes_to_generate:
            theme_content = self.theme_manager.themes[theme]
            generated_content = await self.content_generator.generate_theme_content(
                theme_content
            )
            # 存储生成的内容
            self.generated_contents[theme] = generated_content
            print(f"\n系统: 已经为主题 '{theme}' 生成了新的内容。")
            print(f"要查看生成的内容吗？(yes/no)")
            
            show_content = input("\n您: ").lower()
            if show_content in ['yes', 'y', '是']:
                print(f"\n{generated_content}\n")
        
        # 计算指标
        metrics = {
            'emotion_score': 0.5,
            'interest_score': 0.7,
            'completion_score': 0.3,
            'topic_weight': 0.5
        }
        
        # 生成下一个问题
        next_question = await self.dialogue_manager.generate_next_question(
            metrics,
            self.context
        )
        
        # 保存当前问题
        self.last_question = next_question
        
        return next_question
        
    async def show_generated_content(self, theme: str = None):
        """显示生成的内容"""
        if theme:
            if theme in self.generated_contents:
                return self.generated_contents[theme]
            return f"主题 '{theme}' 还没有生成内容。"
        else:
            # 显示所有主题的内容
            result = []
            for theme, content in self.generated_contents.items():
                result.append(f"\n=== {theme} ===\n{content}\n")
            return "\n".join(result) if result else "还没有生成任何内容。"

async def main():
    memory_lane = MemoryLane()
    await memory_lane.start_conversation()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

